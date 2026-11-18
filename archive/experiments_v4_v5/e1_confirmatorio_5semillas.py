#!/usr/bin/env python3
# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Experimento Confirmatorio E1 (5 Semillas: 42, 123, 456, 789, 999) — Protocolo Fase 1 (TRDM-v5)
Objetivo:
- Entrenar las 5 semillas a 4000 pasos en GPU Tesla T4 con persistencia en Google Drive.
- Evaluar sobre partición de desarrollo DEV (108 casos).
- Realizar la evaluación confirmatoria final sobre TEST-COMPOSICIONAL (480 casos) y TEST-AZAR (240 casos).
- Generar métricas estadísticas agregadas (media, desviación estándar, IC 95%) según Sección 10 del Protocolo.
"""

import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import HybridConfig
from model import TernaryRecursiveDiffusionLM
from data_v2 import (
    ProceduralCompositionalDatasetV2,
    parse_fixed_prediction,
    parse_fixed_scratchpad,
)
from trdm.diffusion.corruption import apply_blockwise_diffusion_corruption


OUTPUT_DIR = "/content/drive/MyDrive/BasicLite"
if not os.path.exists("/content/drive/MyDrive"):
    print("AVISO: Google Drive no montado en /content/drive/MyDrive. Guardando en './output'.")
    OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 789, 999]


def build_field_ids(prompt_b: bytes, target_b: bytes, max_seq_len: int, use_scratchpad: bool) -> torch.Tensor:
    field_ids = torch.zeros(max_seq_len, dtype=torch.long)
    prompt_len = 1 + len(prompt_b)

    if use_scratchpad:
        lines = target_b.split(b"\n")
        assert len(lines) >= 5, f"Formato inesperado: {lines}"
        len_pw = len(lines[0]) + 1
        len_pms = len(lines[1]) + 1
        len_final = len(lines[2]) + 1 + len(lines[3]) + 1

        p_start = prompt_len
        pw_end = p_start + len_pw
        pms_end = pw_end + len_pms
        final_end = pms_end + len_final

        field_ids[p_start:pw_end] = 1        # Campo 1: Pw
        field_ids[pw_end:pms_end] = 2        # Campo 2: Pms
        field_ids[pms_end:final_end] = 3     # Campo 3: Watt y MS
        if final_end < max_seq_len:
            field_ids[final_end] = 4         # Campo 4: EOS
    else:
        target_end = min(max_seq_len, prompt_len + len(target_b))
        field_ids[prompt_len:target_end] = 3
        if target_end < max_seq_len:
            field_ids[target_end] = 4
    return field_ids


class FieldDatasetWrapper(Dataset):
    def __init__(self, base_ds: ProceduralCompositionalDatasetV2):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x_0, can_mask, digit_mask = self.base_ds[idx]
        p_b, t_b, _, _ = self.base_ds.samples[idx]
        f_ids = build_field_ids(p_b, t_b, self.base_ds.cfg.max_seq_len, self.base_ds.cfg.use_scratchpad)
        return x_0, can_mask, digit_mask, f_ids


def compute_v4_decoupled_loss(
    final_logits: torch.Tensor,
    intermediate_logits: List[torch.Tensor],
    halt_probs_list: List[torch.Tensor],
    x_0: torch.Tensor,
    mask_positions: torch.Tensor,
    digit_mask: torch.Tensor,
    t_noise: torch.Tensor,
    cfg: HybridConfig,
    field_ids: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    x_0_sup = x_0[mask_positions]
    dig_sup = digit_mask[mask_positions]
    field_sup = field_ids[mask_positions]

    pos_importance = torch.full_like(x_0_sup, cfg.template_token_weight, dtype=final_logits.dtype)
    pos_importance = torch.where(
        x_0_sup == cfg.pad_token_id,
        torch.full_like(pos_importance, cfg.pad_token_weight),
        pos_importance,
    )
    pos_importance = torch.where(
        dig_sup,
        torch.full_like(pos_importance, cfg.digit_token_weight),
        pos_importance,
    )

    if t_noise.dim() == 1:
        inv_t_full = (1.0 / t_noise.clamp(min=0.10)).unsqueeze(1).expand_as(x_0)
    else:
        inv_t_full = (1.0 / t_noise.clamp(min=0.10)).expand_as(x_0)
    inv_t_sup = inv_t_full[mask_positions]

    def _field_weighted_ce(lg: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        lg_sup = lg[mask_positions] if lg.ndim == 3 else lg
        token_loss = F.cross_entropy(lg_sup, x_0_sup, reduction="none")
        w_loss = token_loss * pos_importance * inv_t_sup

        m_pw = (field_sup == 1)
        m_pms = (field_sup == 2)
        m_fin = (field_sup == 3)
        m_eos = (field_sup == 4)

        loss_pw = (w_loss[m_pw].sum() / pos_importance[m_pw].sum().clamp(min=1.0)) if m_pw.any() else torch.tensor(0.0, device=lg_sup.device)
        loss_pms = (w_loss[m_pms].sum() / pos_importance[m_pms].sum().clamp(min=1.0)) if m_pms.any() else torch.tensor(0.0, device=lg_sup.device)
        loss_fin = (w_loss[m_fin].sum() / pos_importance[m_fin].sum().clamp(min=1.0)) if m_fin.any() else torch.tensor(0.0, device=lg_sup.device)
        loss_eos = (w_loss[m_eos].sum() / pos_importance[m_eos].sum().clamp(min=1.0)) if m_eos.any() else torch.tensor(0.0, device=lg_sup.device)

        w_pw = getattr(cfg, "loss_weight_pw", 1.0)
        w_pms = getattr(cfg, "loss_weight_pms", 1.0)
        w_fin = getattr(cfg, "loss_weight_final", 3.5)
        w_eos = getattr(cfg, "loss_weight_eos", 0.2)

        num = (
            w_pw * loss_pw * float(m_pw.any())
            + w_pms * loss_pms * float(m_pms.any())
            + w_fin * loss_fin * float(m_fin.any())
            + w_eos * loss_eos * float(m_eos.any())
        )
        den = (
            w_pw * float(m_pw.any())
            + w_pms * float(m_pms.any())
            + w_fin * float(m_fin.any())
            + w_eos * float(m_eos.any())
        )
        comb_loss = num / max(den, 1e-6)

        preds_sup = lg_sup.argmax(dim=-1)
        correct_dig = ((preds_sup == x_0_sup) & dig_sup).sum().item()
        total_dig = dig_sup.sum().item()
        dig_acc = correct_dig / max(1.0, total_dig)

        metrics_dict = {
            "loss_pw": float(loss_pw.item()) if m_pw.any() else 0.0,
            "loss_pms": float(loss_pms.item()) if m_pms.any() else 0.0,
            "loss_final": float(loss_fin.item()) if m_fin.any() else 0.0,
            "loss_eos": float(loss_eos.item()) if m_eos.any() else 0.0,
            "dig_acc": float(dig_acc),
        }
        return comb_loss, metrics_dict

    loss_final, metrics_dict = _field_weighted_ce(final_logits)

    loss_inter = torch.tensor(0.0, device=x_0.device)
    if len(intermediate_logits) > 1 and cfg.intermediate_loss_weight > 0:
        inter_terms = [_field_weighted_ce(lg)[0] for lg in intermediate_logits[:-1]]
        loss_inter = sum(inter_terms) / len(inter_terms)

    total_loss = loss_final + cfg.intermediate_loss_weight * loss_inter
    return total_loss, metrics_dict


def update_bitnet_schedule(optimizer: torch.optim.Optimizer, step: int, max_steps: int, base_lr: float) -> float:
    warmup_steps = max(10, int(0.10 * max_steps))
    second_stage_start = int(0.80 * max_steps)

    if step <= warmup_steps:
        lr_mult = step / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        lr_mult = 0.10 + 0.90 * 0.50 * (1.0 + math.cos(math.pi * progress))

    current_lr = base_lr * lr_mult
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
        if param_group.get("weight_decay", 0.0) > 0 and step >= second_stage_start:
            param_group["weight_decay"] = 0.0
    return current_lr


@torch.no_grad()
def sample_diffusion(
    model: nn.Module,
    prompt_b: bytes,
    target_b: bytes,
    cfg: HybridConfig,
    num_steps: int,
    device: torch.device,
) -> Tuple[str, bool, Optional[int], Optional[int], bool]:
    p_len = 1 + len(prompt_b)
    t_len = len(target_b)

    full_seq = [cfg.bos_token_id] + list(prompt_b) + [cfg.mask_token_id] * t_len + [cfg.eos_token_id]
    seq = (full_seq + [cfg.pad_token_id] * (cfg.max_seq_len - len(full_seq)))[: cfg.max_seq_len]

    x = torch.tensor([seq], dtype=torch.long, device=device)
    target_mask = torch.zeros(cfg.max_seq_len, dtype=torch.bool, device=device)
    target_mask[p_len : p_len + t_len] = True

    target_indices = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)
    num_target = int(target_mask.sum().item())
    carried = None
    prev_t = 1.0

    for s in range(num_steps):
        progress = s / float(max(1, num_steps))
        t_val = math.cos(progress * math.pi / 2.0)
        dt_val = max(0.01, prev_t - t_val)
        prev_t = t_val

        t_vec = torch.zeros((1, cfg.max_seq_len), device=device)
        is_masked = (x[0] == cfg.mask_token_id) & target_mask
        t_vec[0, is_masked] = max(0.05, t_val)
        dt_vec = torch.full((1, cfg.max_seq_len), dt_val, device=device)

        logits, _, _, _, carried = model(
            x, t_vec, delta_t=dt_vec, prev_states=carried, num_recursions=2
        )

        logits[0, :, cfg.mask_token_id] = -1e9
        logits[0, :, cfg.bos_token_id] = -1e9

        probs = F.softmax(logits[0], dim=-1)
        confidences, preds = probs.max(dim=-1)

        next_progress = (s + 1) / float(num_steps)
        if s == num_steps - 1:
            tokens_to_mask = 0
        else:
            tokens_to_mask = int(math.floor(num_target * math.cos(next_progress * math.pi / 2.0)))

        num_to_keep = max(0, len(target_indices) - tokens_to_mask)
        conf_target = confidences[target_indices]
        _, topk_indices = torch.topk(conf_target, k=min(num_to_keep, len(target_indices)), largest=True)
        keep_positions = target_indices[topk_indices]

        x[0, target_indices] = cfg.mask_token_id
        x[0, keep_positions] = preds[keep_positions]

    pred_chars = "".join([chr(tok) if 32 <= tok <= 126 or tok == 10 else "?" for tok in x[0, p_len : p_len + t_len].tolist()])
    ok_pred, pred_w, pred_ms = parse_fixed_prediction(pred_chars)
    ok_sp, _, _, _, _ = parse_fixed_scratchpad(pred_chars)
    return pred_chars, ok_pred, pred_w, pred_ms, ok_sp


def evaluate_split(
    model: nn.Module,
    records: List[Tuple[bytes, bytes, int, int]],
    cfg: HybridConfig,
    num_steps: int,
    device: torch.device,
    split_name: str,
) -> Dict[str, float]:
    n = len(records)
    correct_em = 0
    correct_digits = 0
    total_digits = 0
    format_valid = 0
    scratchpad_valid = 0
    t0 = time.perf_counter()

    for prompt_b, target_b, watt_true, ms_true in records:
        pred_text, ok_pred, w_p, ms_p, ok_sp = sample_diffusion(
            model=model,
            prompt_b=prompt_b,
            target_b=target_b,
            cfg=cfg,
            num_steps=num_steps,
            device=device,
        )

        if ok_pred:
            format_valid += 1
            if w_p == watt_true and ms_p == ms_true:
                correct_em += 1
        if ok_sp:
            scratchpad_valid += 1

        target_str = target_b.decode("utf-8")
        for idx, ch in enumerate(target_str):
            if ch.isdigit():
                total_digits += 1
                if idx < len(pred_text) and pred_text[idx] == ch:
                    correct_digits += 1

    dt = time.perf_counter() - t0
    return {
        "exact_match_pct": 100.0 * correct_em / n,
        "correct_em": correct_em,
        "total_cases": n,
        "digit_acc_pct": 100.0 * correct_digits / max(1, total_digits),
        "format_valid_pct": 100.0 * format_valid / n,
        "scratchpad_valid_pct": 100.0 * scratchpad_valid / n,
        "ms_per_case": (dt / n) * 1000.0,
    }


def train_seed(seed: int, device: torch.device) -> Tuple[nn.Module, HybridConfig]:
    ckpt_path = os.path.join(OUTPUT_DIR, f"e1_checkpoint_seed_{seed}.pt")
    pilot_ckpt = os.path.join(OUTPUT_DIR, "trdm_v5_piloto42_checkpoint.pt")

    cfg = HybridConfig(
        seed=seed,
        cpu_threads=2,
        max_seq_len=160,
        use_scratchpad=True,
        use_explicit_operands=False,
        max_steps=4000,
        batch_size=16,
        learning_rate=2.0e-3,
        dev_mode=False,
        loss_weight_pw=1.0,
        loss_weight_pms=1.0,
        loss_weight_final=3.5,
        loss_weight_eos=0.2,
    )

    model = TernaryRecursiveDiffusionLM(cfg).to(device)

    # Si la semilla 42 ya fue entrenada en el piloto calibrado a 4000 pasos
    if seed == 42 and os.path.exists(pilot_ckpt) and not os.path.exists(ckpt_path):
        print(f"[Semilla {seed}] Reutilizando checkpoint calibrado del piloto: {pilot_ckpt}")
        ckpt = torch.load(pilot_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        torch.save({"model_state": model.state_dict(), "step": 4000, "seed": 42, "config": cfg}, ckpt_path)
        return model, cfg

    if os.path.exists(ckpt_path):
        print(f"[Semilla {seed}] Encontrado checkpoint existente en {ckpt_path}. Cargando...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        return model, cfg

    print(f"\n{'='*70}\n  ENTRENANDO SEMILLA {seed} (4000 pasos)\n{'='*70}")
    torch.manual_seed(seed)
    random.seed(seed)

    raw_train_ds = ProceduralCompositionalDatasetV2(cfg, split="train", num_samples=cfg.max_steps * cfg.batch_size)
    train_ds = FieldDatasetWrapper(raw_train_ds)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.configure_adamw_groups(), lr=cfg.learning_rate, betas=(0.9, 0.95))
    step = 0
    step_times = []
    t_start = time.perf_counter()

    for x_0, can_mask, digit_mask, field_ids in train_loader:
        t_s0 = time.perf_counter()
        step += 1
        x_0 = x_0.to(device)
        can_mask = can_mask.to(device)
        digit_mask = digit_mask.to(device)
        field_ids = field_ids.to(device)

        lr_now = update_bitnet_schedule(optimizer, step, cfg.max_steps, cfg.learning_rate)

        # Paso encadenado 1
        x_t1, mask_pos1, t1 = apply_blockwise_diffusion_corruption(x_0, can_mask, cfg)
        logits1, inter1, halt1, _, carried1 = model(x_t1, t1, num_recursions=cfg.num_recursions)
        loss1, met1 = compute_v4_decoupled_loss(logits1, inter1, halt1, x_0, mask_pos1, digit_mask, t1, cfg, field_ids)

        # Paso encadenado 2
        preds1 = logits1.detach().argmax(dim=-1)
        frac_keep = torch.empty_like(t1).uniform_(0.35, 0.85)
        keep_cond = (torch.rand_like(x_0, dtype=torch.float32) < frac_keep.unsqueeze(1)) if t1.dim() == 1 else (torch.rand_like(x_0, dtype=torch.float32) < frac_keep)
        keep_mask = keep_cond & mask_pos1

        for b in range(x_0.size(0)):
            if not keep_mask[b].any() and mask_pos1[b].any():
                idx_m = torch.nonzero(mask_pos1[b], as_tuple=False).squeeze(-1)
                keep_mask[b, idx_m[0]] = True

        x_t2 = x_t1.clone()
        revealed_pos = mask_pos1 & ~keep_mask
        x_t2[revealed_pos] = preds1[revealed_pos]
        x_t2[keep_mask] = cfg.mask_token_id

        t2 = (t1 * frac_keep).clamp(min=0.05)
        dt = (t1 - t2).clamp(min=0.01)

        logits2, inter2, halt2, _, _ = model(x_t2, t2, delta_t=dt, prev_states=carried1, num_recursions=cfg.num_recursions)
        loss2, met2 = compute_v4_decoupled_loss(logits2, inter2, halt2, x_0, keep_mask, digit_mask, t2, cfg, field_ids)

        total_loss = 0.45 * loss1 + 0.55 * loss2
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        dt_step = time.perf_counter() - t_s0
        if step > 5:
            step_times.append(dt_step * 1000.0)

        if step % 500 == 0 or step == cfg.max_steps:
            ms_p = (sum(step_times) / max(1, len(step_times))) if step_times else dt_step * 1000.0
            print(
                f"  [Semilla {seed} | Paso {step:04d}/{cfg.max_steps}] Loss: {total_loss.item():.3f} | "
                f"DigAcc(tr): {met2['dig_acc']*100:.1f}% | {ms_p:.1f} ms/p",
                flush=True,
            )

        if step >= cfg.max_steps:
            break

    t_total = time.perf_counter() - t_start
    print(f"  Semilla {seed} finalizada en {t_total:.1f} s. Guardando en Drive...")
    torch.save({"model_state": model.state_dict(), "step": step, "seed": seed, "config": cfg}, ckpt_path)
    return model, cfg


def main():
    print("=" * 80)
    print("  EXPERIMENTO CONFIRMATORIO E1 (5 SEMILLAS) — PROTOCOLO FASE 1")
    print(f"  Directorio persistente Drive: {OUTPUT_DIR}")
    print(f"  Semillas oficiales: {SEEDS}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    base_cfg = HybridConfig()
    print("Cargando particiones para evaluación confirmatoria...")
    dev_ds = ProceduralCompositionalDatasetV2(base_cfg, split="dev")
    test_comp_ds = ProceduralCompositionalDatasetV2(base_cfg, split="test-composicional")
    test_azar_ds = ProceduralCompositionalDatasetV2(base_cfg, split="test-azar")

    print(f"  Dev:   {len(dev_ds.unique_records)} combinaciones")
    print(f"  Comp:  {len(test_comp_ds.unique_records)} combinaciones")
    print(f"  Azar:  {len(test_azar_ds.unique_records)} combinaciones")

    per_seed_results = {}

    for seed in SEEDS:
        model, cfg = train_seed(seed, device)
        model.eval()

        print(f"\n[Semilla {seed}] Evaluando en DEV, TEST-COMPOSICIONAL y TEST-AZAR...")
        res_seed = {}

        # 1. Dev (C1 y C2 K=4)
        res_seed["dev_c1"] = evaluate_split(model, dev_ds.unique_records, cfg, num_steps=1, device=device, split_name="dev_c1")
        res_seed["dev_c2_k4"] = evaluate_split(model, dev_ds.unique_records, cfg, num_steps=4, device=device, split_name="dev_c2_k4")

        # 2. Test-Composicional (C1 y C2 K=4)
        res_seed["test_comp_c1"] = evaluate_split(model, test_comp_ds.unique_records, cfg, num_steps=1, device=device, split_name="test_comp_c1")
        res_seed["test_comp_c2_k4"] = evaluate_split(model, test_comp_ds.unique_records, cfg, num_steps=4, device=device, split_name="test_comp_c2_k4")

        # 3. Test-Azar (C1 y C2 K=4)
        res_seed["test_azar_c1"] = evaluate_split(model, test_azar_ds.unique_records, cfg, num_steps=1, device=device, split_name="test_azar_c1")
        res_seed["test_azar_c2_k4"] = evaluate_split(model, test_azar_ds.unique_records, cfg, num_steps=4, device=device, split_name="test_azar_c2_k4")

        per_seed_results[seed] = res_seed

        print(
            f"  [Semilla {seed} Resultados]\n"
            f"    DEV:       C1 EM={res_seed['dev_c1']['exact_match_pct']:.2f}% | C2(K=4) EM={res_seed['dev_c2_k4']['exact_match_pct']:.2f}%\n"
            f"    TEST-COMP: C1 EM={res_seed['test_comp_c1']['exact_match_pct']:.2f}% | C2(K=4) EM={res_seed['test_comp_c2_k4']['exact_match_pct']:.2f}%\n"
            f"    TEST-AZAR: C1 EM={res_seed['test_azar_c1']['exact_match_pct']:.2f}% | C2(K=4) EM={res_seed['test_azar_c2_k4']['exact_match_pct']:.2f}%\n"
        )

    # Estadísticas agregadas (media ± std sobre las 5 semillas)
    def compute_stats(key: str, metric: str = "exact_match_pct") -> Dict[str, float]:
        vals = [per_seed_results[s][key][metric] for s in SEEDS]
        mean_v = sum(vals) / len(vals)
        std_v = (sum((v - mean_v) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
        ci95 = 2.776 * (std_v / (len(vals) ** 0.5))  # t-Student con 4 g.l.
        return {
            "mean": mean_v,
            "std": std_v,
            "ci95": ci95,
            "min": min(vals),
            "max": max(vals),
            "values": vals,
        }

    summary_stats = {
        "dev_c1_em": compute_stats("dev_c1"),
        "dev_c2_k4_em": compute_stats("dev_c2_k4"),
        "test_comp_c1_em": compute_stats("test_comp_c1"),
        "test_comp_c2_k4_em": compute_stats("test_comp_c2_k4"),
        "test_azar_c1_em": compute_stats("test_azar_c1"),
        "test_azar_c2_k4_em": compute_stats("test_azar_c2_k4"),
        "test_comp_c1_digacc": compute_stats("test_comp_c1", "digit_acc_pct"),
        "test_comp_c2_k4_digacc": compute_stats("test_comp_c2_k4", "digit_acc_pct"),
        "test_comp_c1_format": compute_stats("test_comp_c1", "format_valid_pct"),
        "test_comp_c2_k4_format": compute_stats("test_comp_c2_k4", "format_valid_pct"),
    }

    print("\n" + "=" * 80)
    print("  RESUMEN ESTADÍSTICO AGREGADO E1 (5 SEMILLAS — CONFIRMATORIO)")
    print("=" * 80)
    for k, v in summary_stats.items():
        print(f"  {k:25s}: {v['mean']:6.2f}% ± {v['std']:5.2f}% (IC 95%: [{v['mean']-v['ci95']:.2f}%, {v['mean']+v['ci95']:.2f}%])")
    print("=" * 80)

    final_payload = {
        "experiment": "E1_Confirmatorio_5Semillas",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": SEEDS,
        "per_seed_results": per_seed_results,
        "summary_stats": summary_stats,
    }

    out_file = os.path.join(OUTPUT_DIR, "e1_confirmatorio_5semillas_results.json")
    with open(out_file, "w") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\nResultados confirmatorios guardados en Google Drive: {out_file}")


if __name__ == "__main__":
    main()
