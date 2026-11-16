#!/usr/bin/env python3
"""
Corrida Piloto Técnica (Semilla 42) — Protocolo Fase 1 (TRDM-v5)
Objetivo:
- Medir telemetría real (ms/paso, tiempo total, memoria pico RSS) en Intel Core i3.
- Entrenar modelo M-HIB a 3000 pasos con formato de salida de ancho fijo (watt: 087, ms: 045).
- Evaluar periódicamente sobre el conjunto de desarrollo (dev, 108 casos) y train (200 casos).
- REGLA ESTRICTA DE PRE-REGISTRO: Los conjuntos de test (test-comp y test-azar) NO se abren.
"""

import json
import math
import os
import random
import re
import resource
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


def get_peak_ram_mib() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024.0


def compute_isolated_digit_acc(pred_str: str, target_str: str) -> Tuple[int, int]:
    correct, total = 0, 0
    for idx, ch in enumerate(target_str):
        if ch.isdigit():
            total += 1
            if idx < len(pred_str) and pred_str[idx] == ch:
                correct += 1
    return correct, total


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

        # Exactitud de dígitos en entrenamiento
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
def evaluate_dev_single_pass(model: nn.Module, dev_ds: ProceduralCompositionalDatasetV2, cfg: HybridConfig) -> Dict[str, float]:
    """Condición C1: Una sola pasada con todo el bloque objetivo enmascarado (t = 1.0)."""
    model.eval()
    device = next(model.parameters()).device
    correct_digits = 0
    total_digits = 0
    correct_em = 0
    total_samples = len(dev_ds.unique_records)

    for prompt_b, target_b, watt_true, ms_true in dev_ds.unique_records:
        target_str = target_b.decode("utf-8")
        seq = [cfg.bos_token_id] + list(prompt_b) + list(target_b) + [cfg.eos_token_id]
        p_len = 1 + len(prompt_b)
        seq = (seq[: cfg.max_seq_len] + [cfg.pad_token_id] * (cfg.max_seq_len - len(seq)))[: cfg.max_seq_len]

        x_0 = torch.tensor([seq], dtype=torch.long, device=device)
        can_mask = torch.zeros(1, cfg.max_seq_len, dtype=torch.bool, device=device)
        can_mask[0, p_len:] = True

        digit_mask = torch.zeros(1, cfg.max_seq_len, dtype=torch.bool, device=device)
        t_end = min(cfg.max_seq_len, p_len + len(target_b))
        for p in range(p_len, t_end):
            if 48 <= seq[p] <= 57:
                digit_mask[0, p] = True

        x_t = x_0.clone()
        x_t[can_mask] = cfg.mask_token_id
        t_vec = torch.full((1, cfg.max_seq_len), 1.0, device=device)

        logits, _, _, _, _ = model(x_t, t_vec, num_recursions=2)
        preds = logits[0].argmax(dim=-1)

        # Digits accuracy
        dig_correct = ((preds == x_0[0]) & digit_mask[0]).sum().item()
        dig_total = digit_mask[0].sum().item()
        correct_digits += dig_correct
        total_digits += dig_total

        # Parse text
        pred_chars = "".join([chr(preds[pos].item()) for pos in range(p_len, t_end)])
        ok, w_p, ms_p = parse_fixed_prediction(pred_chars)
        if ok and w_p == watt_true and ms_p == ms_true:
            correct_em += 1

    model.train()
    return {
        "c1_digit_acc": 100.0 * correct_digits / max(1, total_digits),
        "c1_exact_match": 100.0 * correct_em / max(1, total_samples),
    }


def main():
    print("=" * 80)
    print("  CORRIDA PILOTO TÉCNICA (SEMILLA 42) — PROTOCOLO FASE 1 (TRDM-v5)")
    print("=" * 80)

    torch.set_num_threads(2)
    torch.manual_seed(42)
    random.seed(42)

    cfg = HybridConfig(
        seed=42,
        cpu_threads=2,
        max_seq_len=160,
        use_scratchpad=True,
        use_explicit_operands=False,
        max_steps=3000,
        batch_size=16,
        learning_rate=2.0e-3,
        dev_mode=False,
    )

    print(f"Configuración: max_seq_len={cfg.max_seq_len}, pasos={cfg.max_steps}, lote={cfg.batch_size}, hilos=2")

    # 1. Datasets de desarrollo y entrenamiento
    print("Cargando particiones de datos (ProceduralCompositionalDatasetV2)...")
    raw_train_ds = ProceduralCompositionalDatasetV2(cfg, split="train", num_samples=cfg.max_steps * cfg.batch_size)
    dev_ds = ProceduralCompositionalDatasetV2(cfg, split="dev")
    train_ds = FieldDatasetWrapper(raw_train_ds)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    print(f"Dataset train cargado: {len(raw_train_ds.unique_records)} combinaciones únicas (total muestras: {len(train_ds)})")
    print(f"Dataset dev cargado:   {len(dev_ds.unique_records)} combinaciones únicas (test permanece cerrado)")

    # 2. Modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de cómputo seleccionado: {device}")
    if device.type == "cuda":
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

    model = TernaryRecursiveDiffusionLM(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo M-HIB inicializado: {total_params:,} parámetros.")

    optimizer = torch.optim.AdamW(model.configure_adamw_groups(), lr=cfg.learning_rate, betas=(0.9, 0.95))

    step = 0
    step_times = []
    t_start = time.perf_counter()
    train_time_accum = 0.0

    eval_history = []

    print("\nIniciando bucle de entrenamiento piloto a 3000 pasos...")

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
        assert torch.isfinite(total_loss), f"Pérdida no finita en paso {step}: {total_loss.item()}"

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(grad_norm):
            optimizer.step()

        dt_step = time.perf_counter() - t_s0
        train_time_accum += dt_step
        if step > 5:
            step_times.append(dt_step * 1000.0)

        if step % 100 == 0 or step == 1:
            ms_p = (train_time_accum / step) * 1000.0
            rss_now = get_peak_ram_mib()
            print(
                f"  [Piloto 42] Paso {step:04d}/{cfg.max_steps} | LR: {lr_now:.2e} | "
                f"Loss: {total_loss.item():.3f} [Pw: {met2['loss_pw']:.2f}, Pms: {met2['loss_pms']:.2f}, Fin: {met2['loss_final']:.2f}] | "
                f"DigAcc(tr): {met2['dig_acc']*100:.1f}% | GradNorm: {grad_norm.item():.2f} | "
                f"Tiempo: {ms_p:.1f} ms/p | RAM: {rss_now:.1f} MiB",
                flush=True
            )

        # Evaluación periódica en dev
        if step % 500 == 0 or step == cfg.max_steps:
            dev_metrics = evaluate_dev_single_pass(model, dev_ds, cfg)
            print(
                f"\n  ---> EVALUACIÓN DEV (Paso {step}): C1-DigAcc = {dev_metrics['c1_digit_acc']:.2f}% | "
                f"C1-EM = {dev_metrics['c1_exact_match']:.2f}%\n",
                flush=True
            )
            eval_history.append({
                "step": step,
                "dev_metrics": dev_metrics,
            })

        if step >= cfg.max_steps:
            break

    total_wall_s = time.perf_counter() - t_start
    steady_ms_p = sum(step_times) / max(1, len(step_times))

    # Guardar checkpoint de control
    ckpt_path = "trdm_v5_piloto42_checkpoint.pt"
    torch.save({
        "model_state": model.state_dict(),
        "step": step,
        "seed": 42,
        "config": cfg,
    }, ckpt_path)
    print(f"\nCheckpoint guardado en: {ckpt_path}")

    report = {
        "experiment": "E1_Piloto_Semilla42",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_wall_seconds": total_wall_s,
        "steady_ms_per_step": steady_ms_p,
        "peak_ram_mib": get_peak_ram_mib(),
        "steps_completed": step,
        "final_loss": float(total_loss.item()),
        "eval_history": eval_history,
        "test_set_opened": False,
    }

    out_json = "piloto_e1_semilla42_results.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 80)
    print("  CORRIDA PILOTO COMPLETADA")
    print(f"  Tiempo total: {total_wall_s:.1f} s (~{total_wall_s/60.0:.1f} min)")
    print(f"  Régimen estable: {steady_ms_p:.1f} ms/paso")
    print(f"  Pico RAM: {get_peak_ram_mib():.1f} MiB")
    print(f"  Resultados guardados en: {out_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
