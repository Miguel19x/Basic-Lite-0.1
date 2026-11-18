#!/usr/bin/env python3
# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Evaluador Multietapa C1 vs C2 (K=4, 8, 16) sobre partición DEV (108 casos)
Utiliza el checkpoint de calibración de la semilla 42 (trdm_v5_piloto42_checkpoint.pt).
REGLA ESTRICTA DE PRE-REGISTRO: Los conjuntos de test (test-comp y test-azar) NO se abren.
"""

import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import HybridConfig
from model import TernaryRecursiveDiffusionLM
from data_v2 import (
    ProceduralCompositionalDatasetV2,
    parse_fixed_prediction,
    parse_fixed_scratchpad,
)


def load_model_and_config(ckpt_path: str = "trdm_v5_piloto42_checkpoint.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cargando checkpoint '{ckpt_path}' en dispositivo: {device}")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_path}")
        
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", HybridConfig())
    cfg.device = device
    
    model = TernaryRecursiveDiffusionLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Modelo cargado con éxito. Paso de checkpoint: {ckpt.get('step', 'N/A')}")
    return model, cfg, device


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
    """
    Muestreador iterativo de difusión discreta para K pasos.
    oracle_scratchpad: si es True, revela Pw y Pms y solo pide predecir watt y ms.
    """
    p_len = 1 + len(prompt_b)
    t_len = len(target_b)
    
    if oracle_scratchpad:
        # Revelar Pw y Pms, solo enmascarar watt y ms
        lines = target_b.split(b"\n")
        pw_pms_len = len(lines[0]) + 1 + len(lines[1]) + 1
        mask_start = p_len + pw_pms_len
        mask_end = p_len + t_len
        
        full_seq = [cfg.bos_token_id] + list(prompt_b) + list(target_b) + [cfg.eos_token_id]
        seq = (full_seq + [cfg.pad_token_id] * (cfg.max_seq_len - len(full_seq)))[: cfg.max_seq_len]
        
        x = torch.tensor([seq], dtype=torch.long, device=device)
        target_mask = torch.zeros(cfg.max_seq_len, dtype=torch.bool, device=device)
        target_mask[mask_start:mask_end] = True
        x[0, target_mask] = cfg.mask_token_id
    else:
        # Difusión completa desde ruido sobre todo el bloque objetivo
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


def run_evaluation_suite(model: nn.Module, cfg: HybridConfig, dev_ds: ProceduralCompositionalDatasetV2, device: torch.device):
    records = dev_ds.unique_records
    n = len(records)
    print(f"\nIniciando Evaluación Multietapa sobre partición DEV ({n} casos)...")
    print("=" * 80)
    
    conditions = [
        ("C1 (1 paso, t=1.0)", 1, False),
        ("C2 (K=4 pasos)", 4, False),
        ("C2 (K=8 pasos)", 8, False),
        ("C2 (K=16 pasos)", 16, False),
        ("C2-Oracle-Scratchpad (K=8)", 8, True),
    ]
    
    results = {}
    
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
        
        results[cond_name] = {
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
    return results


def main():
    ckpt_file = "trdm_v5_piloto42_checkpoint.pt"
    if not os.path.exists(ckpt_file):
        print(f"Error: {ckpt_file} no encontrado en directorio actual.")
        sys.exit(1)
        
    model, cfg, device = load_model_and_config(ckpt_file)
    dev_ds = ProceduralCompositionalDatasetV2(cfg, split="dev")
    results = run_evaluation_suite(model, cfg, dev_ds, device)
    
    out_file = "eval_c2_dev_results.json"
    import json
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados en: {out_file}")


if __name__ == "__main__":
    main()
