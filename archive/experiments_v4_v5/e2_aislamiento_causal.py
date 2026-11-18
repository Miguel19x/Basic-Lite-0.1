#!/usr/bin/env python3
# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Experimento E2: Aislamiento Causal (Decodificación frente a Estructura) — Protocolo Fase 1 (TRDM-v5)
Matriz Factorial 2x2 evaluada sobre las 5 semillas confirmatorias en Google Colab:
- Factor 1 (Estructura): Plantilla Oculta (Sin pistas) vs. Plantilla Visible (Solo dígitos enmascarados)
- Factor 2 (Dinámica): 1 Pasada Feed-forward (t=1.0) vs. Muestreo Iterativo (K=4 pasos)

Condiciones:
- Condición A: Plantilla Oculta, 1 Pasada (C1 estándar)
- Condición B: Plantilla Oculta, Iterativa K=4 (C2 estándar)
- Condición C: Plantilla Visible, 1 Pasada (Solo dígitos libres)
- Condición D: Plantilla Visible, Iterativa K=4 (Iteración sobre casillas de dígitos)
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

from config import HybridConfig
from model import TernaryRecursiveDiffusionLM
from data_v2 import (
    ProceduralCompositionalDatasetV2,
    parse_fixed_prediction,
    parse_fixed_scratchpad,
)


OUTPUT_DIR = "/content/drive/MyDrive/BasicLite"
if not os.path.exists("/content/drive/MyDrive"):
    print("AVISO: Google Drive no montado en /content/drive/MyDrive. Guardando en './output'.")
    OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 789, 999]


@torch.no_grad()
def evaluate_sample_e2(
    model: nn.Module,
    prompt_b: bytes,
    target_b: bytes,
    cfg: HybridConfig,
    condition: str,
    device: torch.device,
) -> Tuple[bool, bool, bool, int, int]:
    """
    Evalúa una muestra bajo una de las 4 condiciones factoriales de E2.
    Devuelve (exact_match, scratchpad_exact, final_exact, dig_correct, dig_total).
    """
    p_len = 1 + len(prompt_b)
    t_len = len(target_b)
    full_seq = [cfg.bos_token_id] + list(prompt_b) + list(target_b) + [cfg.eos_token_id]
    seq = (full_seq + [cfg.pad_token_id] * (cfg.max_seq_len - len(full_seq)))[: cfg.max_seq_len]

    x = torch.tensor([seq], dtype=torch.long, device=device)
    target_mask = torch.zeros(cfg.max_seq_len, dtype=torch.bool, device=device)

    # Identificar posiciones de dígitos en el bloque objetivo
    digit_positions = []
    for pos in range(p_len, p_len + t_len):
        if 48 <= seq[pos] <= 57:
            digit_positions.append(pos)

    # Configuración de máscaras según el Factor de Estructura
    if condition in ("A", "B"):
        # Plantilla Oculta: todo el bloque objetivo enmascarado
        target_mask[p_len : p_len + t_len] = True
        x[0, target_mask] = cfg.mask_token_id
    elif condition in ("C", "D"):
        # Plantilla Visible: SOLO las posiciones de los dígitos son enmascaradas
        for pos in digit_positions:
            target_mask[pos] = True
        x[0, target_mask] = cfg.mask_token_id
    else:
        raise ValueError(f"Condición desconocida: {condition}")

    num_target = int(target_mask.sum().item())
    target_indices = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)

    # Configuración según el Factor de Dinámica
    num_steps = 1 if condition in ("A", "C") else 4
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

        if condition in ("C", "D"):
            # En plantilla visible, restringir logits estrictamente al vocabulario de dígitos '0'..'9'
            non_digit_mask = torch.ones(logits.size(-1), dtype=torch.bool, device=device)
            non_digit_mask[48:58] = False
            logits[0, target_mask, non_digit_mask] = -1e9

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

    # Decodificar texto generado
    pred_chars = "".join([chr(tok) if 32 <= tok <= 126 or tok == 10 else "?" for tok in x[0, p_len : p_len + t_len].tolist()])
    target_str = target_b.decode("utf-8")

    # Contar exactitud de dígitos
    dig_correct, dig_total = 0, 0
    for idx, ch in enumerate(target_str):
        if ch.isdigit():
            dig_total += 1
            if idx < len(pred_chars) and pred_chars[idx] == ch:
                dig_correct += 1

    ok_sp, pw_p, pms_p, w_p, ms_p = parse_fixed_scratchpad(pred_chars)
    ok_tgt_sp, pw_t, pms_t, w_t, ms_t = parse_fixed_scratchpad(target_str)

    exact_match = (ok_sp and ok_tgt_sp and pw_p == pw_t and pms_p == pms_t and w_p == w_t and ms_p == ms_t)
    sp_exact = (ok_sp and ok_tgt_sp and pw_p == pw_t and pms_p == pms_t)
    final_exact = (ok_sp and ok_tgt_sp and w_p == w_t and ms_p == ms_t)

    return exact_match, sp_exact, final_exact, dig_correct, dig_total


def evaluate_condition_on_split(
    model: nn.Module,
    records: List[Tuple[bytes, bytes, int, int]],
    cfg: HybridConfig,
    condition: str,
    device: torch.device,
) -> Dict[str, float]:
    n = len(records)
    em_count = 0
    sp_count = 0
    fin_count = 0
    total_dig_corr = 0
    total_dig_cnt = 0
    t0 = time.perf_counter()

    for prompt_b, target_b, _, _ in records:
        em, sp_ex, fin_ex, d_cor, d_tot = evaluate_sample_e2(
            model=model,
            prompt_b=prompt_b,
            target_b=target_b,
            cfg=cfg,
            condition=condition,
            device=device,
        )
        if em:
            em_count += 1
        if sp_ex:
            sp_count += 1
        if fin_ex:
            fin_count += 1
        total_dig_corr += d_cor
        total_dig_cnt += d_tot

    dt = time.perf_counter() - t0
    return {
        "exact_match_pct": 100.0 * em_count / n,
        "scratchpad_exact_pct": 100.0 * sp_count / n,
        "final_exact_pct": 100.0 * fin_count / n,
        "digit_acc_pct": 100.0 * total_dig_corr / max(1, total_dig_cnt),
        "ms_per_case": (dt / n) * 1000.0,
        "em_count": em_count,
        "total_cases": n,
    }


def main():
    print("=" * 80)
    print("  EXPERIMENTO E2: AISLAMIENTO CAUSAL (MATRIZ FACTORIAL 2x2)")
    print(f"  Directorio persistente Drive: {OUTPUT_DIR}")
    print(f"  Semillas a evaluar: {SEEDS}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    base_cfg = HybridConfig()
    print("Cargando particiones para evaluación causal E2...")
    test_comp_ds = ProceduralCompositionalDatasetV2(base_cfg, split="test-composicional")
    test_azar_ds = ProceduralCompositionalDatasetV2(base_cfg, split="test-azar")

    print(f"  Test-Composicional: {len(test_comp_ds.unique_records)} combinaciones")
    print(f"  Test-Azar:          {len(test_azar_ds.unique_records)} combinaciones")

    conditions = ["A", "B", "C", "D"]
    cond_labels = {
        "A": "Condición A (Plantilla Oculta, 1 Pasada)",
        "B": "Condición B (Plantilla Oculta, K=4 Pasos)",
        "C": "Condición C (Plantilla Visible, 1 Pasada)",
        "D": "Condición D (Plantilla Visible, K=4 Pasos)",
    }

    per_seed_e2 = {}

    for seed in SEEDS:
        ckpt_path = os.path.join(OUTPUT_DIR, f"e1_checkpoint_seed_{seed}.pt")
        if not os.path.exists(ckpt_path):
            # Fallback para semilla 42
            alt_path = os.path.join(OUTPUT_DIR, "trdm_v5_piloto42_checkpoint.pt")
            if seed == 42 and os.path.exists(alt_path):
                ckpt_path = alt_path
            else:
                print(f"Error: Checkpoint no encontrado para semilla {seed}: {ckpt_path}")
                continue

        print(f"\n[Semilla {seed}] Cargando modelo desde: {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt.get("config", HybridConfig())
        model = TernaryRecursiveDiffusionLM(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        seed_res = {"test_comp": {}, "test_azar": {}}

        for cond in conditions:
            # Evaluar en Test-Composicional
            res_comp = evaluate_condition_on_split(
                model=model,
                records=test_comp_ds.unique_records,
                cfg=cfg,
                condition=cond,
                device=device,
            )
            seed_res["test_comp"][cond] = res_comp

            # Evaluar en Test-Azar
            res_azar = evaluate_condition_on_split(
                model=model,
                records=test_azar_ds.unique_records,
                cfg=cfg,
                condition=cond,
                device=device,
            )
            seed_res["test_azar"][cond] = res_azar

            print(
                f"  [{cond_labels[cond]}]\n"
                f"    TEST-COMP: EM={res_comp['exact_match_pct']:5.2f}% | SP_EM={res_comp['scratchpad_exact_pct']:5.2f}% | Fin_EM={res_comp['final_exact_pct']:5.2f}% | DigAcc={res_comp['digit_acc_pct']:5.2f}%\n"
                f"    TEST-AZAR: EM={res_azar['exact_match_pct']:5.2f}% | SP_EM={res_azar['scratchpad_exact_pct']:5.2f}% | Fin_EM={res_azar['final_exact_pct']:5.2f}% | DigAcc={res_azar['digit_acc_pct']:5.2f}%"
            )

        per_seed_e2[seed] = seed_res

    # Calcular estadísticas agregadas
    def aggregate_e2_metric(split_key: str, cond: str, metric: str) -> Dict[str, float]:
        vals = [per_seed_e2[s][split_key][cond][metric] for s in per_seed_e2]
        mean_v = sum(vals) / len(vals)
        std_v = (sum((v - mean_v) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
        ci95 = 2.776 * (std_v / (len(vals) ** 0.5))
        return {
            "mean": mean_v,
            "std": std_v,
            "ci95": ci95,
            "min": min(vals),
            "max": max(vals),
            "values": vals,
        }

    stats_e2 = {"test_comp": {}, "test_azar": {}}
    for sp in ["test_comp", "test_azar"]:
        for cond in conditions:
            stats_e2[sp][cond] = {
                "exact_match": aggregate_e2_metric(sp, cond, "exact_match_pct"),
                "scratchpad_exact": aggregate_e2_metric(sp, cond, "scratchpad_exact_pct"),
                "final_exact": aggregate_e2_metric(sp, cond, "final_exact_pct"),
                "digit_acc": aggregate_e2_metric(sp, cond, "digit_acc_pct"),
                "ms_per_case": aggregate_e2_metric(sp, cond, "ms_per_case"),
            }

    # Desacoplamiento de Factores (§145-161):
    em_A = stats_e2["test_comp"]["A"]["exact_match"]["mean"]
    em_B = stats_e2["test_comp"]["B"]["exact_match"]["mean"]
    em_C = stats_e2["test_comp"]["C"]["exact_match"]["mean"]
    em_D = stats_e2["test_comp"]["D"]["exact_match"]["mean"]

    delta_iter = em_B - em_A       # Efecto neto de la iteración
    delta_struct = em_C - em_A     # Efecto neto de la estructura de plantilla
    interaction = em_D - em_B      # Interacción estructura-iteración

    print("\n" + "=" * 80)
    print("  DESACOPLAMIENTO FORMAL DE FACTORES E2 (TEST-COMPOSICIONAL)")
    print("=" * 80)
    print(f"  Condición A (Oculta, 1-paso):   EM = {em_A:6.2f}%")
    print(f"  Condición B (Oculta, K=4):      EM = {em_B:6.2f}%")
    print(f"  Condición C (Visible, 1-paso):  EM = {em_C:6.2f}%")
    print(f"  Condición D (Visible, K=4):     EM = {em_D:6.2f}%")
    print("-" * 80)
    print(f"  Efecto Neto Iteración (EM_B - EM_A):        {delta_iter:+6.2f} puntos")
    print(f"  Efecto Neto Estructura (EM_C - EM_A):       {delta_struct:+6.2f} puntos")
    print(f"  Interacción Estructura-Iteración (EM_D - EM_B): {interaction:+6.2f} puntos")
    print("=" * 80)

    # Diagnóstico Pre-Registrado:
    if em_C >= 60.0 and (em_C - em_B) >= 30.0:
        veredicto = "FALLO_CONTROL_ALINEAMIENTO_Y_DELIMITADORES (El núcleo calcula bien cuando tiene la plantilla fija)"
    elif em_C < 30.0:
        veredicto = "INCAPACIDAD_ARITMETICA_INTRINSECA (El núcleo no alcanza 30% EM incluso con plantilla fija)"
    else:
        veredicto = "ERROR_COMPUESTO_MIXTO (Capacidad parcial con pérdidas combinadas de aritmética y estructura)"

    print(f"\n>>> VEREDICTO DIAGNÓSTICO FORMAL E2: {veredicto} <<<\n")

    payload = {
        "experiment": "E2_Aislamiento_Causal",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": list(per_seed_e2.keys()),
        "per_seed_results": per_seed_e2,
        "aggregate_stats": stats_e2,
        "factor_decomposition": {
            "delta_iter_B_minus_A": delta_iter,
            "delta_struct_C_minus_A": delta_struct,
            "interaction_D_minus_B": interaction,
            "diagnostic_verdict": veredicto,
        },
    }

    out_file = os.path.join(OUTPUT_DIR, "e2_aislamiento_causal_results.json")
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Resultados E2 guardados en Google Drive: {out_file}")


if __name__ == "__main__":
    main()
