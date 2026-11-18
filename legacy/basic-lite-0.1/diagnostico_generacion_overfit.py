import math
import random
import time
import torch
import torch.nn.functional as F

from config import HybridConfig
from model import TernaryRecursiveDiffusionLM
from data import (
    ProceduralCompositionalDataset,
    apply_blockwise_diffusion_corruption,
    parse_scratchpad_prediction,
)
from generate import generate_block_causal_with_kv_cache


def run_diagnostico():
    cfg = HybridConfig()
    cfg.seed = 42
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)

    print("=== 1. Test de Paridad: Forward Paralelo vs. Forward Incremental con KV-Cache ===")
    model = TernaryRecursiveDiffusionLM(cfg).to(cfg.device)
    model.eval()

    # Crear una secuencia sintética limpia de prueba (longitud 128)
    L = cfg.max_seq_len
    x_test = torch.randint(10, 200, (1, L), dtype=torch.long, device=cfg.device)
    t_zero = torch.zeros((1, L), device=cfg.device)
    dt_zero = torch.zeros((1, L), device=cfg.device)

    # A. Forward Paralelo Completo
    with torch.no_grad():
        logits_paralelo, _, _, _, _ = model(x_test, t_zero, delta_t=dt_zero, num_recursions=2)

    # B. Forward Incremental Bloque a Bloque con KV-Cache
    B_blk = cfg.block_size
    num_blocks = L // B_blk
    logits_kv_list = []
    kv_caches = None
    committed_len = 0

    with torch.no_grad():
        for b in range(num_blocks):
            blk_tokens = x_test[:, b * B_blk : (b + 1) * B_blk]
            t_b = torch.zeros((1, B_blk), device=cfg.device)
            dt_b = torch.zeros((1, B_blk), device=cfg.device)
            l_b, _, _, kv_caches, _ = model(
                blk_tokens,
                t_b,
                delta_t=dt_b,
                num_recursions=2,
                kv_caches_per_r=kv_caches,
                start_pos=committed_len,
            )
            logits_kv_list.append(l_b)
            committed_len += B_blk

    logits_kv = torch.cat(logits_kv_list, dim=1)
    diff = (logits_paralelo - logits_kv).abs().max().item()
    print(f"Diferencia máxima absoluta (Paralelo vs. KV-Cache): {diff:.6e}")
    if diff > 1e-4:
        print("ALERTA CRÍTICA: La KV Cache difiere del forward paralelo!")
    else:
        print("Paridad de KV Cache confirmada (equivalencia matemática exacta).")

    print("\n=== 2. Entrenamiento Rápido de Sobreajuste (250 y 400 pasos) e Inspección Literal ===")
    ds = ProceduralCompositionalDataset(cfg, split="train", num_samples=8)
    fixed_records = ds.unique_records[:8]

    batch_x0, batch_can_mask, batch_digit_mask = [], [], []
    for prompt_b, target_b, _, _ in fixed_records:
        seq = [cfg.bos_token_id] + list(prompt_b) + list(target_b) + [cfg.eos_token_id]
        p_len = 1 + len(prompt_b)
        seq = seq[: cfg.max_seq_len]
        seq = seq + [cfg.pad_token_id] * (cfg.max_seq_len - len(seq))

        c_mask = torch.zeros(cfg.max_seq_len, dtype=torch.bool)
        c_mask[p_len:] = True

        d_mask = torch.zeros(cfg.max_seq_len, dtype=torch.bool)
        t_end = min(cfg.max_seq_len, p_len + len(target_b))
        for p in range(p_len, t_end):
            if 48 <= seq[p] <= 57:
                d_mask[p] = True

        batch_x0.append(torch.tensor(seq, dtype=torch.long))
        batch_can_mask.append(c_mask)
        batch_digit_mask.append(d_mask)

    x_0 = torch.stack(batch_x0, dim=0).to(cfg.device)
    can_mask = torch.stack(batch_can_mask, dim=0).to(cfg.device)
    digit_mask = torch.stack(batch_digit_mask, dim=0).to(cfg.device)

    # Entrenar modelo
    model = TernaryRecursiveDiffusionLM(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.configure_adamw_groups(), lr=2.0e-3, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1.0e-4)

    for step in range(1, 301):
        model.train()
        r_train = cfg.num_recursions

        x_t1, mask_pos1, t1 = apply_blockwise_diffusion_corruption(x_0, can_mask, cfg)
        dt1 = torch.full_like(t1, 1.0 / max(1, cfg.diffusion_steps_per_block))
        logits1, inter1, halt1, _, carried1 = model(x_t1, t1, delta_t=dt1, num_recursions=r_train)

        mask_f1 = mask_pos1.float()
        pos_imp1 = torch.full_like(mask_f1, cfg.template_token_weight)
        pos_imp1 = torch.where(x_0 == cfg.pad_token_id, torch.full_like(mask_f1, cfg.pad_token_weight), pos_imp1)
        pos_imp1 = torch.where(digit_mask, torch.full_like(mask_f1, cfg.digit_token_weight), pos_imp1)
        eff_mask1 = mask_f1 * pos_imp1
        loss1 = (F.cross_entropy(logits1.view(-1, cfg.vocab_size), x_0.view(-1), reduction="none").view(x_0.shape) * eff_mask1).sum() / eff_mask1.sum().clamp(min=1.0)

        preds1 = logits1.detach().argmax(dim=-1)
        frac_keep = torch.empty_like(t1).uniform_(0.30, 0.70)
        keep_mask = (torch.rand_like(x_0, dtype=torch.float32) < frac_keep) & mask_pos1
        for b in range(x_0.size(0)):
            if not keep_mask[b].any() and mask_pos1[b].any():
                idx_m = torch.nonzero(mask_pos1[b], as_tuple=False).squeeze(-1)
                keep_mask[b, idx_m[0]] = True

        x_t2 = x_t1.clone()
        revealed_pos = mask_pos1 & ~keep_mask
        x_t2[revealed_pos] = preds1[revealed_pos]
        x_t2[keep_mask] = cfg.mask_token_id

        t2 = torch.where(keep_mask, (t1 * frac_keep).clamp(min=0.05), torch.zeros_like(t1))
        dt2 = (t1 - t2).clamp(min=0.01)

        logits2, inter2, halt2, _, _ = model(x_t2, t2, delta_t=dt2, prev_states=carried1, num_recursions=r_train)

        mask_f2 = keep_mask.float()
        pos_imp2 = torch.full_like(mask_f2, cfg.template_token_weight)
        pos_imp2 = torch.where(x_0 == cfg.pad_token_id, torch.full_like(mask_f2, cfg.pad_token_weight), pos_imp2)
        pos_imp2 = torch.where(digit_mask, torch.full_like(mask_f2, cfg.digit_token_weight), pos_imp2)
        eff_mask2 = mask_f2 * pos_imp2
        loss2 = (F.cross_entropy(logits2.view(-1, cfg.vocab_size), x_0.view(-1), reduction="none").view(x_0.shape) * eff_mask2).sum() / eff_mask2.sum().clamp(min=1.0)

        total_loss = 0.45 * loss1 + 0.55 * loss2
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if step in (100, 200, 250, 300):
            print(f"\n--- Diagnóstico en Paso {step} (Loss: {total_loss.item():.4f}) ---")
            model.eval()
            for i, (prompt_b, target_b, tw, tms) in enumerate(fixed_records[:3]):
                p_str = prompt_b.decode("utf-8")
                t_str = target_b.decode("utf-8")
                gen_txt, history, n_evals = generate_block_causal_with_kv_cache(
                    model, p_str, cfg, num_recursions_override=2
                )
                is_fmt, p_sw, p_sms, pw, pms = parse_scratchpad_prediction(gen_txt, p_str)
                continuation = gen_txt[len(p_str):] if gen_txt.startswith(p_str) else gen_txt
                print(f"Muestra {i+1}:")
                print(f"  Objetivo Real:    {repr(t_str)}")
                print(f"  Continuación Gen: {repr(continuation)}")
                print(f"  Fmt Ok: {is_fmt} | Preds: sub_w={p_sw}, sub_ms={p_sms}, watt={pw}, ms={pms}")


if __name__ == "__main__":
    run_diagnostico()
