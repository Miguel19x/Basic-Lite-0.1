#!/usr/bin/env python3
# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Script Maestro para Google Colab:
1. Entrenamiento Piloto (Semilla 42) a 3000 pasos con persistencia directa en Google Drive.
2. Evaluación Multietapa C1 vs C2 (K=4, 8, 16) y C2-Oráculo sobre partición DEV (108 casos).
3. Resguardo de checkpoints y métricas en /content/drive/MyDrive/BasicLite/.
REGLA ESTRICTA DE PRE-REGISTRO: Los conjuntos de test (test-comp y test-azar) NO se abren.
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


# Configuración del directorio de salida persistente
OUTPUT_DIR = "/content/drive/MyDrive/BasicLite"
if not os.path.exists("/content/drive/MyDrive"):
    print("AVISO: Google Drive no está montado en /content/drive/MyDrive. Guardando en directorio local './output'.")
    OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
        w_fin = getattr(cfg, "loss_weight_final", 1.0)
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
def sample_diffusion_k_steps(
    model: nn.Module,
    prompt_b: bytes,
    target_b: bytes,
    cfg: HybridConfig,
    num_steps: int,
    device: torch.device,
    oracle_scratchpad: bool = False,
) -> Tuple[str, bool, Optional[int], Optional[int], bool]:
    p_len = 1 + len(prompt_b)
    t_len = len(target_b)

    if oracle_scratchpad:
        lines = target_b.split(b"\n")
        pw_pms_len = len(lines[0]) + 1 + len(lines[1]) + 1
        mask_start = p_len + pw_pms_len

        full_seq = [cfg.bos_token_id] + list(prompt_b) + list(target_b) + [cfg.eos_token_id]
        seq = (full_seq + [cfg.pad_token_id] * (cfg.max_seq_len - len(full_seq)))[: cfg.max_seq_len]

        x = torch.tensor([seq], dtype=torch.long, device=device)
        target_mask = torch.zeros(cfg.max_seq_len, dtype=torch.bool, device=device)
        target_mask[mask_start : p_len + t_len] = True
        x[0, target_mask] = cfg.mask_token_id
    else:
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


def evaluate_c2_suite(model: nn.Module, cfg: HybridConfig, dev_ds: ProceduralCompositionalDatasetV2, device: torch.device) -> Dict:
    records = dev_ds.unique_records
    n = len(records)
    print(f"\n{'='*80}\n  EVALUACIÓN MULTIETAPA C1 vs C2 SOBRE PARTICIÓN DEV ({n} casos)\n{'='*80}")

    conditions = [
        ("C1 (1 paso, t=1.0)", 1, False),
        ("C2 (K=4 pasos)", 4, False),
        ("C2 (K=8 pasos)", 8, False),
        ("C2 (K=16 pasos)", 16, False),
        ("C2-Oracle-Scratchpad (K=8)", 8, True),
    ]

    eval_results = {}

    for cond_name, k_steps, use_oracle in conditions:
        t0 = time.perf_counter()
        correct_em = 0
        correct_digits = 0
        total_digits = 0
        format_valid = 0
        scratchpad_valid = 0
        sample_preds = []

        for idx_case, (prompt_b, target_b, watt_true, ms_true) in enumerate(records):
            pred_text, ok_pred, w_p, ms_p, ok_sp = sample_diffusion_k_steps(
                model=model,
                prompt_b=prompt_b,
                target_b=target_b,
                cfg=cfg,
                num_steps=k_steps,
                device=device,
                oracle_scratchpad=use_oracle,
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

            if idx_case < 3:
                sample_preds.append({
                    "target": target_str.replace("\n", " | "),
                    "pred": pred_text.replace("\n", " | "),
                    "match": (ok_pred and w_p == watt_true and ms_p == ms_true),
                })

        dt = time.perf_counter() - t0
        em_pct = 100.0 * correct_em / n
        dig_pct = 100.0 * correct_digits / max(1, total_digits)
        fmt_pct = 100.0 * format_valid / n
        sp_pct = 100.0 * scratchpad_valid / n
        ms_per_case = (dt / n) * 1000.0

        eval_results[cond_name] = {
            "exact_match_pct": em_pct,
            "digit_acc_pct": dig_pct,
            "format_valid_pct": fmt_pct,
            "scratchpad_valid_pct": sp_pct,
            "ms_per_case": ms_per_case,
            "correct_em": correct_em,
            "total_cases": n,
            "sample_predictions": sample_preds,
        }

        print(
            f"[{cond_name:28s}] EM: {em_pct:6.2f}% ({correct_em}/{n}) | "
            f"DigAcc: {dig_pct:6.2f}% | FmtValid: {fmt_pct:6.2f}% | "
            f"SPValid: {sp_pct:6.2f}% | {ms_per_case:5.1f} ms/caso"
        )

    print("=" * 80)
    return eval_results


def main():
    print("=" * 80)
    print("  BASIC-LITE (TRDM-v5) — ENTRENAMIENTO PILOTO + EVALUACIÓN C2 EN DEV")
    print(f"  Directorio persistente: {OUTPUT_DIR}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de cómputo detectado: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(42)
    random.seed(42)

    cfg = HybridConfig(
        seed=42,
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

    print("Cargando datos...")
    raw_train_ds = ProceduralCompositionalDatasetV2(cfg, split="train", num_samples=cfg.max_steps * cfg.batch_size)
    dev_ds = ProceduralCompositionalDatasetV2(cfg, split="dev")
    train_ds = FieldDatasetWrapper(raw_train_ds)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = TernaryRecursiveDiffusionLM(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo M-HIB inicializado: {total_params:,} parámetros.")

    ckpt_path = os.path.join(OUTPUT_DIR, "trdm_v5_piloto42_checkpoint.pt")

    print(f"\nIniciando entrenamiento a {cfg.max_steps} pasos (loss_weight_final={cfg.loss_weight_final})...")
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

        # Paso encadenado 1 (ruido alto t1)
        x_t1, mask_pos1, t1 = apply_blockwise_diffusion_corruption(x_0, can_mask, cfg)
        logits1, inter1, halt1, _, carried1 = model(x_t1, t1, num_recursions=cfg.num_recursions)
        loss1, met1 = compute_v4_decoupled_loss(logits1, inter1, halt1, x_0, mask_pos1, digit_mask, t1, cfg, field_ids)

        # Predicción desacoplada para paso 2
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

        # Paso encadenado 2
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

        if step % 200 == 0 or step == 1 or step == cfg.max_steps:
            ms_p = (sum(step_times) / max(1, len(step_times))) if step_times else dt_step * 1000.0
            print(
                f"  [Paso {step:04d}/{cfg.max_steps}] Loss: {total_loss.item():.3f} | "
                f"DigAcc(tr): {met2['dig_acc']*100:.1f}% | {ms_p:.1f} ms/p",
                flush=True,
            )

        if step >= cfg.max_steps:
            break

    total_wall_s = time.perf_counter() - t_start
    print(f"\nEntrenamiento completado en {total_wall_s:.1f} s (~{total_wall_s/60.0:.1f} min). Guardando checkpoint en Drive...")

    torch.save({
        "model_state": model.state_dict(),
        "step": step,
        "seed": 42,
        "config": cfg,
    }, ckpt_path)
    print(f"Checkpoint persistido con éxito en: {ckpt_path}")

    # Evaluación Multietapa C1 vs C2 sobre Dev
    eval_results = evaluate_c2_suite(model, cfg, dev_ds, device)

    out_json = os.path.join(OUTPUT_DIR, "eval_c2_dev_results.json")
    with open(out_json, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Resultados de evaluación persistidos en: {out_json}")


if __name__ == "__main__":
    main()
