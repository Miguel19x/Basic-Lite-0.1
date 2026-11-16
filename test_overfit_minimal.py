import math
import random
import time
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F

from config import HybridConfig
from model import TernaryRecursiveDiffusionLM
from data import (
    ProceduralCompositionalDataset,
    apply_blockwise_diffusion_corruption,
    parse_watt_ms_prediction,
    parse_scratchpad_prediction,
)
from generate import generate_block_causal_with_kv_cache


def run_minimal_overfit_experiment(num_samples: int = 8, steps: int = 400):
    cfg = HybridConfig()
    cfg.seed = 42
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)

    print(f"=== Experimento de Sobreajuste Mínimo (Overfit Test en {num_samples} Muestras) ===")
    print("Objetivo: Diagnosticar si la arquitectura TRDM puede memorizar a la perfección un conjunto diminuto.")

    # 1. Seleccionar num_samples muestras fijas del conjunto de entrenamiento
    ds = ProceduralCompositionalDataset(cfg, split="train", num_samples=num_samples)
    fixed_records = ds.unique_records[:num_samples]

    # Preparar tensores para las 8 muestras
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

    model = TernaryRecursiveDiffusionLM(cfg).to(cfg.device)
    # Tasa de aprendizaje adaptada para sobreajuste con decaimiento coseno estilo SOTA
    optimizer = torch.optim.AdamW(model.configure_adamw_groups(), lr=2.0e-3, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1.0e-4)

    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        model.train()
        r_train = cfg.num_recursions

        # Paso encadenado 1 (ruido alto)
        x_t1, mask_pos1, t1 = apply_blockwise_diffusion_corruption(x_0, can_mask, cfg)
        dt1 = torch.full_like(t1, 1.0 / max(1, cfg.diffusion_steps_per_block))
        logits1, inter1, halt1, _, carried1 = model(x_t1, t1, delta_t=dt1, num_recursions=r_train)

        # Pérdida ponderada paso 1
        mask_f1 = mask_pos1.float()
        pos_imp1 = torch.full_like(mask_f1, cfg.template_token_weight)
        pos_imp1 = torch.where(x_0 == cfg.pad_token_id, torch.full_like(mask_f1, cfg.pad_token_weight), pos_imp1)
        pos_imp1 = torch.where(digit_mask, torch.full_like(mask_f1, cfg.digit_token_weight), pos_imp1)
        eff_mask1 = mask_f1 * pos_imp1
        loss1 = (F.cross_entropy(logits1.view(-1, cfg.vocab_size), x_0.view(-1), reduction="none").view(x_0.shape) * eff_mask1).sum() / eff_mask1.sum().clamp(min=1.0)

        # Paso encadenado 2 con las PREDICCIONES de la red en posiciones reveladas
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
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                # A. Reconstrucción estática (Mask Acc & Digit Acc)
                masked_digits = keep_mask & digit_mask
                p2 = logits2.argmax(dim=-1)
                all_acc = float(((p2 == x_0) & keep_mask).float().sum().item() / max(1.0, keep_mask.float().sum().item()))
                dig_acc = float(((p2 == x_0) & masked_digits).float().sum().item() / max(1.0, masked_digits.float().sum().item()))

                # B. Generación completa block-causal desde [MASK] en las 8 muestras
                em_count = 0
                sub_em_count = 0
                fmt_count = 0
                has_scratch = False

                for prompt_b, target_b, tw, tms in fixed_records:
                    p_str = prompt_b.decode("utf-8")
                    t_str = target_b.decode("utf-8")
                    gen_txt, _, _ = generate_block_causal_with_kv_cache(model, p_str, cfg, num_recursions_override=2)
                    is_fmt, p_sw, p_sms, pw, pms = parse_scratchpad_prediction(gen_txt, p_str)
                    if is_fmt:
                        fmt_count += 1
                    if pw == tw and pms == tms:
                        em_count += 1

                    _, true_sw, true_sms, _, _ = parse_scratchpad_prediction(t_str, "")
                    if true_sw is not None and true_sms is not None:
                        has_scratch = True
                        if p_sw == true_sw and p_sms == true_sms:
                            sub_em_count += 1

                em_pct = 100.0 * em_count / num_samples
                fmt_pct = 100.0 * fmt_count / num_samples
                sub_str = f" | Gen Sub-EM: {sub_em_count}/{num_samples}" if has_scratch else ""

                print(
                    f"Paso {step:03d}/{steps} | Loss: {total_loss.item():.4f} | "
                    f"GradNorm: {grad_norm.item():.2f} | "
                    f"Reconst AllAcc: {all_acc * 100:.1f}% | Reconst DigAcc: {dig_acc * 100:.1f}% | "
                    f"Gen Fmt: {fmt_pct:.1f}% | Gen Full-EM: {em_count}/{num_samples} ({em_pct:.1f}%){sub_str}"
                )
                if step in (200, 300, 400):
                    p0 = fixed_records[0][0].decode("utf-8")
                    t0_str = fixed_records[0][1].decode("utf-8")
                    g0, _, _ = generate_block_causal_with_kv_cache(model, p0, cfg, num_recursions_override=2)
                    c0 = g0[len(p0):] if g0.startswith(p0) else g0
                    print(f"  [Ejemplo 1] Objetivo: {repr(t0_str)} | Generado: {repr(c0)}")

    total_time = time.perf_counter() - t0
    print(f"\nExperimento finalizado en {total_time:.2f} s.")


if __name__ == "__main__":
    run_minimal_overfit_experiment(num_samples=8, steps=400)
