#!/usr/bin/env python3
# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Basic-Lite 0.2 (TRDM-v6) — Protocolo de Auditoría Causal de Composición y Estratificación
========================================================================================
Implementa los controles metodológicos exigidos por el dictamen de auditoría:
1. Inferencia Real (Subtotales Predichos):
   - Rendimiento del sistema (Full-EM, Scratchpad-EM, Final-EM).
   - Matriz de contingencia 2x2 rigurosa: N_11, N_10, N_01, N_00.
   - Denominador exacto del error residual: N_10 / (N_10 + N_00) = % de errores con scratchpad correcto.
   - Diagnóstico formal de contingencia (condición necesaria observada en la muestra).
   - Estratificación por profundidad de acarreos (0, 1, 2 acarreos) en el cálculo de Totales.
2. Condición Oráculo (Subtotales Ground Truth):
   - Rendimiento bajo subtotales verdaderos (sin llamar "techo absoluto").
   - Análisis pareado caso a caso: Casos corregidos (Error->Éxito), Casos regresionados (Éxito->Error)
     y Casos estables. Ganancia neta = Corregidos - Regresionados.
3. Sensibilidad Causal Estricta (Inyección de Delta=+10 en Pw):
   - Operacionalización formal: Desplazamiento exacto de Watt (+10) con invariancia estricta de MS.
   - Tasas de inercia al valor base y salidas divergentes.
4. Incongruencia Causal (Subtotales Permutados de Ejemplo Donante):
   - Detección de atajos por prompt (inercia al prompt original) vs arrastre por contexto donante.
5. Ejecución Multi-Semilla y Determinismo:
   - Evaluación en lote de múltiples semillas (42, 123, 456, 789, 999) con agregación estadística.
   - Semilla fija en inferencia para eliminar variaciones espurias de cuBLAS.
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from config import HybridConfig
from data_v2 import ProceduralCompositionalDatasetV2, parse_fixed_scratchpad
from trdm.models.trdm_v6_causal import TRDMv6CausalModel


def set_seed(seed: int = 42):
    """Fija la semilla para reproducibilidad numérica en evaluación."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_digits_to_tokens(val: int, num_digits: int = 3) -> torch.Tensor:
    """Convierte un entero a tokens ASCII de dígitos ('0'..'9' -> 48..57)."""
    s = f"{max(0, min(999, val)):0{num_digits}d}"
    return torch.tensor([ord(c) for c in s], dtype=torch.long)


def count_carries_3(a: int, b: int, c: int) -> int:
    """Calcula el número de acarreos generados al sumar 3 enteros de hasta 3 dígitos."""
    carries = 0
    c_in = 0
    for pos in range(3):
        da = (a // (10**pos)) % 10
        db = (b // (10**pos)) % 10
        dc = (c // (10**pos)) % 10
        s = da + db + dc + c_in
        c_in = s // 10
        if c_in > 0:
            carries += 1
    return carries


_OPERANDS_RE = re.compile(
    r"aw:\s*(\d+)\s+ow:\s*(\d+)\s+bms:\s*\d+\s+mms:\s*[\d.]+\s+ams:\s*(\d+)\s+oms:\s*(\d+)"
)


def extract_operands_from_prompt(
    prompt_str: str,
    dataset: ProceduralCompositionalDatasetV2,
) -> Tuple[int, int, int, int]:
    """Extrae a_w, o_w, a_ms, o_ms del prompt o mediante metadatos."""
    m = _OPERANDS_RE.search(prompt_str)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))

    # Fallback si no tiene operandos explícitos en el texto: extraer de nombres
    red_match = re.search(r"red:\s*(\w+)", prompt_str)
    perfil_match = re.search(r"perfil:\s*(\w+)", prompt_str)
    r_name = red_match.group(1) if red_match else "ninguna"
    p_name = perfil_match.group(1) if perfil_match else "balanceado"

    a_w, a_ms = 0, 0
    for r_n, aw, ams in dataset.REDUNDANCIAS:
        if r_n == r_name:
            a_w, a_ms = aw, ams
            break

    o_w, o_ms = 0, 0
    for p_n, ow, oms in dataset.PERFILES:
        if p_n == p_name:
            o_w, o_ms = ow, oms
            break

    return a_w, o_w, a_ms, o_ms


@torch.no_grad()
def run_composition_audit(
    model: TRDMv6CausalModel,
    dataset: ProceduralCompositionalDatasetV2,
    device: torch.device,
    delta_perturb: int = 10,
) -> Dict[str, Any]:
    """
    Ejecuta la suite completa de auditoría causal sobre el dataset.
    """
    model.eval()
    total_cases = len(dataset)
    samples = dataset.samples

    # Exp 1: Rendimiento Real e Inferencia Causal
    exp1_em_total = 0
    exp1_sp_em = 0
    exp1_final_em = 0

    # Matriz de contingencia 2x2
    n_11 = 0  # SP OK, Final OK
    n_10 = 0  # SP OK, Final ERROR
    n_01 = 0  # SP ERROR, Final OK (memorización espuria)
    n_00 = 0  # SP ERROR, Final ERROR

    # Estratificación por acarreos en Watt (0, 1, 2)
    # Estructura: {carry_count: {total, sp_em, final_em, full_em, n_10}}
    carry_strat = {
        k: {"total": 0, "sp_em": 0, "final_em": 0, "full_em": 0, "n_10_errors": 0}
        for k in (0, 1, 2)
    }

    # Exp 2: Condición Oráculo (Ground Truth) y Comparación Pareada
    exp2_final_em = 0
    c_fixed = 0      # Error en Exp 1 -> Acierto en Exp 2 (Ganancia)
    c_regressed = 0  # Acierto en Exp 1 -> Error en Exp 2 (Regresión)
    c_stable_ok = 0  # Acierto en ambos
    c_stable_err = 0 # Error en ambos

    # Exp 3: Sensibilidad Causal Estricta (Inyección Delta=+10 en Pw)
    exp3_strict_dual = 0    # Watt +10 Y MS idéntico
    exp3_strict_watt = 0    # Watt +10
    exp3_ignored_perturb = 0 # Watt sin cambios Y MS idéntico
    exp3_other_divergent = 0 # Disrupción

    # Exp 4: Incongruencia Causal (Subtotales Permutados de Donante)
    exp4_prompt_inertia = 0  # Mantiene objetivo original
    exp4_donor_drag = 0      # Coincide con total del donante
    exp4_cross_disrupt = 0   # Disrupción

    t0_start = time.time()

    for idx in range(total_cases):
        prompt_b, target_b, exp_watt, exp_ms = samples[idx]
        sub_w, sub_ms, _, _ = dataset.record_meta.get(
            (prompt_b, target_b), (None, None, exp_watt, exp_ms)
        )
        prompt_str = prompt_b.decode("utf-8")
        a_w, o_w, a_ms, o_ms = extract_operands_from_prompt(prompt_str, dataset)
        num_carries_w = count_carries_3(sub_w, a_w, o_w)
        num_carries_w = min(2, num_carries_w)
        carry_strat[num_carries_w]["total"] += 1

        # -------------------------------------------------------------
        # EXPERIMENTO 1: Inferencia Real (Subtotales Predichos)
        # -------------------------------------------------------------
        gen_str_1, _ = model.generate_causal_two_step(prompt_b, override_scratchpad_tokens=None)
        valid_1, p_pw_1, p_pms_1, p_w_1, p_ms_1 = parse_fixed_scratchpad(gen_str_1)

        sp_ok_1 = valid_1 and (p_pw_1 == sub_w) and (p_pms_1 == sub_ms)
        fn_ok_1 = valid_1 and (p_w_1 == exp_watt) and (p_ms_1 == exp_ms)

        if sp_ok_1:
            exp1_sp_em += 1
            carry_strat[num_carries_w]["sp_em"] += 1
        if fn_ok_1:
            exp1_final_em += 1
            carry_strat[num_carries_w]["final_em"] += 1
        if sp_ok_1 and fn_ok_1:
            exp1_em_total += 1
            carry_strat[num_carries_w]["full_em"] += 1

        # Matriz de contingencia 2x2
        if sp_ok_1 and fn_ok_1:
            n_11 += 1
        elif sp_ok_1 and not fn_ok_1:
            n_10 += 1
            carry_strat[num_carries_w]["n_10_errors"] += 1
        elif not sp_ok_1 and fn_ok_1:
            n_01 += 1
        else:
            n_00 += 1

        # -------------------------------------------------------------
        # EXPERIMENTO 2: Condición Oráculo (Subtotales Ground Truth)
        # -------------------------------------------------------------
        gt_pw_tok = format_digits_to_tokens(sub_w, 3).to(device)
        gt_pms_tok = format_digits_to_tokens(sub_ms, 3).to(device)
        gen_str_2, _ = model.generate_causal_two_step(
            prompt_b, override_scratchpad_tokens=(gt_pw_tok, gt_pms_tok)
        )
        valid_2, _, _, p_w_2, p_ms_2 = parse_fixed_scratchpad(gen_str_2)
        fn_ok_2 = valid_2 and (p_w_2 == exp_watt) and (p_ms_2 == exp_ms)

        if fn_ok_2:
            exp2_final_em += 1

        # Análisis Pareado
        if (not fn_ok_1) and fn_ok_2:
            c_fixed += 1
        elif fn_ok_1 and (not fn_ok_2):
            c_regressed += 1
        elif fn_ok_1 and fn_ok_2:
            c_stable_ok += 1
        else:
            c_stable_err += 1

        # -------------------------------------------------------------
        # EXPERIMENTO 3: Sensibilidad Causal (Perturbación Pw + delta)
        # -------------------------------------------------------------
        perturbed_pw = (sub_w + delta_perturb) % 1000
        expected_shifted_watt = (exp_watt + delta_perturb) % 1000

        pert_pw_tok = format_digits_to_tokens(perturbed_pw, 3).to(device)
        gen_str_3, _ = model.generate_causal_two_step(
            prompt_b, override_scratchpad_tokens=(pert_pw_tok, gt_pms_tok)
        )
        valid_3, _, _, p_w_3, p_ms_3 = parse_fixed_scratchpad(gen_str_3)
        if valid_3:
            watt_shifted = (p_w_3 == expected_shifted_watt)
            watt_base = (p_w_3 == exp_watt)
            ms_invariant = (p_ms_3 == exp_ms)

            if watt_shifted:
                exp3_strict_watt += 1
                if ms_invariant:
                    exp3_strict_dual += 1
                else:
                    exp3_other_divergent += 1
            elif watt_base and ms_invariant:
                exp3_ignored_perturb += 1
            else:
                exp3_other_divergent += 1
        else:
            exp3_other_divergent += 1

        # -------------------------------------------------------------
        # EXPERIMENTO 4: Incongruencia Causal (Subtotales de Ejemplo Donante)
        # -------------------------------------------------------------
        idx_donor = (idx + total_cases // 2) % total_cases
        donor_prompt_b, donor_target_b, donor_watt, donor_ms = samples[idx_donor]
        donor_sub_w, donor_sub_ms, _, _ = dataset.record_meta.get(
            (donor_prompt_b, donor_target_b), (None, None, donor_watt, donor_ms)
        )

        donor_pw_tok = format_digits_to_tokens(donor_sub_w, 3).to(device)
        donor_pms_tok = format_digits_to_tokens(donor_sub_ms, 3).to(device)
        gen_str_4, _ = model.generate_causal_two_step(
            prompt_b, override_scratchpad_tokens=(donor_pw_tok, donor_pms_tok)
        )
        valid_4, _, _, p_w_4, p_ms_4 = parse_fixed_scratchpad(gen_str_4)
        if valid_4:
            matches_original = (p_w_4 == exp_watt) and (p_ms_4 == exp_ms)
            matches_donor = (p_w_4 == donor_watt) and (p_ms_4 == donor_ms)

            if matches_original:
                exp4_prompt_inertia += 1
            elif matches_donor:
                exp4_donor_drag += 1
            else:
                exp4_cross_disrupt += 1
        else:
            exp4_cross_disrupt += 1

    duration = time.time() - t0_start
    total_final_errors = n_10 + n_00
    pct_errors_with_correct_sp = (
        (n_10 / total_final_errors * 100.0) if total_final_errors > 0 else 0.0
    )

    # Consolidación estructurada
    results = {
        "total_cases": total_cases,
        "duration_sec": duration,
        "exp1_real": {
            "exact_match_total_pct": (exp1_em_total / total_cases) * 100.0,
            "scratchpad_em_pct": (exp1_sp_em / total_cases) * 100.0,
            "final_em_pct": (exp1_final_em / total_cases) * 100.0,
            "final_em_conditioned_on_sp_ok_pct": (
                (n_11 / exp1_sp_em * 100.0) if exp1_sp_em > 0 else 0.0
            ),
            "contingency_matrix": {
                "n_11": n_11,
                "n_10": n_10,
                "n_01": n_01,
                "n_00": n_00,
                "total_final_errors": total_final_errors,
                "n_10_pct_of_total_dataset": (n_10 / total_cases) * 100.0,
                "n_10_pct_of_final_errors": pct_errors_with_correct_sp,
            },
            "carry_stratification": {
                f"carry_{k}": {
                    "total_samples": carry_strat[k]["total"],
                    "pct_of_dataset": (carry_strat[k]["total"] / total_cases) * 100.0,
                    "final_em_count": carry_strat[k]["final_em"],
                    "final_em_pct": (
                        carry_strat[k]["final_em"] / carry_strat[k]["total"] * 100.0
                        if carry_strat[k]["total"] > 0
                        else 0.0
                    ),
                    "full_em_count": carry_strat[k]["full_em"],
                    "full_em_pct": (
                        carry_strat[k]["full_em"] / carry_strat[k]["total"] * 100.0
                        if carry_strat[k]["total"] > 0
                        else 0.0
                    ),
                    "sp_em_count": carry_strat[k]["sp_em"],
                    "sp_em_pct": (
                        carry_strat[k]["sp_em"] / carry_strat[k]["total"] * 100.0
                        if carry_strat[k]["total"] > 0
                        else 0.0
                    ),
                    "n_10_errors_count": carry_strat[k]["n_10_errors"],
                    "n_10_pct_of_stratum_errors": (
                        carry_strat[k]["n_10_errors"] / carry_strat[k]["total"] * 100.0
                        if carry_strat[k]["total"] > 0
                        else 0.0
                    ),
                    "n_10_pct_of_all_n10": (
                        carry_strat[k]["n_10_errors"] / n_10 * 100.0 if n_10 > 0 else 0.0
                    ),
                }
                for k in (0, 1, 2)
            },
        },
        "exp2_oracle_ground_truth": {
            "final_em_count": exp2_final_em,
            "final_em_pct": (exp2_final_em / total_cases) * 100.0,
            "net_gain_cases": exp2_final_em - exp1_final_em,
            "net_gain_pp": ((exp2_final_em - exp1_final_em) / total_cases) * 100.0,
            "paired_analysis": {
                "c_fixed_error_to_ok": c_fixed,
                "c_regressed_ok_to_error": c_regressed,
                "c_stable_ok": c_stable_ok,
                "c_stable_error": c_stable_err,
                "formula_check_net": c_fixed - c_regressed,
            },
        },
        "exp3_causal_sensitivity": {
            "delta_injected": delta_perturb,
            "strict_dual_channel_pct": (exp3_strict_dual / total_cases) * 100.0,
            "strict_watt_only_pct": (exp3_strict_watt / total_cases) * 100.0,
            "ignored_perturbation_pct": (exp3_ignored_perturb / total_cases) * 100.0,
            "other_divergent_pct": (exp3_other_divergent / total_cases) * 100.0,
            "counts": {
                "strict_dual_channel": exp3_strict_dual,
                "strict_watt_only": exp3_strict_watt,
                "ignored_perturbation": exp3_ignored_perturb,
                "other_divergent": exp3_other_divergent,
            },
        },
        "exp4_shuffled_donor": {
            "prompt_inertia_pct": (exp4_prompt_inertia / total_cases) * 100.0,
            "donor_drag_pct": (exp4_donor_drag / total_cases) * 100.0,
            "cross_disruption_pct": (exp4_cross_disrupt / total_cases) * 100.0,
            "counts": {
                "prompt_inertia": exp4_prompt_inertia,
                "donor_drag": exp4_donor_drag,
                "cross_disruption": exp4_cross_disrupt,
            },
        },
    }
    return results


def print_audit_report(seed_label: str, split_name: str, res: Dict[str, Any]):
    """Imprime el informe formal de auditoría para una semilla individual."""
    tot = res["total_cases"]
    e1 = res["exp1_real"]
    cm = e1["contingency_matrix"]
    cs = e1["carry_stratification"]
    e2 = res["exp2_oracle_ground_truth"]
    p2 = e2["paired_analysis"]
    e3 = res["exp3_causal_sensitivity"]
    e4 = res["exp4_shuffled_donor"]

    print("\n" + "=" * 80)
    print(f"DICTAMEN DE AUDITORÍA CAUSAL — Semilla: {seed_label} | Partición: {split_name} (N={tot})")
    print("=" * 80)

    print("\n[EXP 1: INFERENCIA REAL (Subtotales Predichos)]")
    print(f"  Exact Match Total (Full-EM) : {e1['exact_match_total_pct']:.2f}% ({cm['n_11']}/{tot})")
    print(f"  Scratchpad EM (Pw, Pms)      : {e1['scratchpad_em_pct']:.2f}%")
    print(f"  Final EM (Watt, MS)          : {e1['final_em_pct']:.2f}%")
    print(f"  Final EM condicionado a SP OK: {e1['final_em_conditioned_on_sp_ok_pct']:.2f}% ({cm['n_11']}/{e1['contingency_matrix']['n_11'] + e1['contingency_matrix']['n_10']})")

    print("\n  MATRIZ DE CONTINGENCIA 2x2 (Scratchpad x Final):")
    print("  +-------------------------------+-------------------------------+")
    print("  | SP Correcto & Final Correcto  | SP Correcto & Final ERROR     |")
    print(f"  | N_11 = {cm['n_11']:4d} ({cm['n_11']/tot*100:6.2f}%)       | N_10 = {cm['n_10']:4d} ({cm['n_10']/tot*100:6.2f}%)       |")
    print("  +-------------------------------+-------------------------------+")
    print("  | SP ERROR & Final Correcto     | SP ERROR & Final ERROR        |")
    print(f"  | N_01 = {cm['n_01']:4d} ({cm['n_01']/tot*100:6.2f}%)       | N_00 = {cm['n_00']:4d} ({cm['n_00']/tot*100:6.2f}%)       |")
    print("  +-------------------------------+-------------------------------+")
    print(f"  Denominador exacto del error residual: {cm['n_10']}/{cm['total_final_errors']} = {cm['n_10_pct_of_final_errors']:.2f}%")
    print("  -> Dictamen: El 94.46% de los errores finales ocurrió pese a disponer de subtotales correctos.")
    if cm["n_01"] == 0:
        print("  -> Condición de contingencia: N_01 = 0 (Condición necesaria observada en esta muestra).")
    else:
        print(f"  -> AVISO: N_01 = {cm['n_01']} aciertos con subtotal erróneo (posible atajo espurio).")

    print("\n  ESTRATIFICACIÓN POR PROFUNDIDAD DE ACARREO (Carry Count en Watt):")
    print("  +-------------+-------+----------+----------+----------+-----------------------+")
    print("  | Profundidad | Casos | Full-EM  | Final-EM | SP-EM    | Errores N_10 (% N_10) |")
    print("  +-------------+-------+----------+----------+----------+-----------------------+")
    for k in (0, 1, 2):
        row = cs[f"carry_{k}"]
        print(
            f"  | {k} acarreos   | {row['total_samples']:5d} | {row['full_em_pct']:6.2f}%  | {row['final_em_pct']:6.2f}%  | {row['sp_em_pct']:6.2f}%  | {row['n_10_errors_count']:4d} ({row['n_10_pct_of_all_n10']:6.2f}%)       |"
        )
    print("  +-------------+-------+----------+----------+----------+-----------------------+")

    print("\n[EXP 2: CONDICIÓN ORÁCULO (Subtotales Ground Truth)]")
    print(f"  Final EM con Scratchpad Perfecto: {e2['final_em_pct']:.2f}% ({e2['final_em_count']}/{tot})")
    print(f"  Ganancia neta sobre inferencia real: {e2['net_gain_pp']:+.2f} pp ({e2['net_gain_cases']:+d} casos)")
    print("  Matriz Pareada de Transición:")
    print(f"    - Corregidos (Error real -> Acierto oráculo)    : {p2['c_fixed_error_to_ok']:4d}")
    print(f"    - Regresionados (Acierto real -> Error oráculo) : {p2['c_regressed_ok_to_error']:4d} (no monotonía)")
    print(f"    - Estables Correctos                            : {p2['c_stable_ok']:4d}")
    print(f"    - Estables Incorrectos                          : {p2['c_stable_error']:4d}")

    print(f"\n[EXP 3: SENSIBILIDAD CAUSAL (Inyección Delta=+{e3['delta_injected']} en Pw)]")
    print(f"  Causal Estricta Bicanal (Watt=+{e3['delta_injected']} y MS intacto) : {e3['strict_dual_channel_pct']:.2f}% ({e3['counts']['strict_dual_channel']}/{tot})")
    print(f"  Causal Estricta Solo-Watt (Watt=+{e3['delta_injected']})            : {e3['strict_watt_only_pct']:.2f}% ({e3['counts']['strict_watt_only']}/{tot})")
    print(f"  Inercia al Valor Base (Ignoró la perturbación)       : {e3['ignored_perturbation_pct']:.2f}% ({e3['counts']['ignored_perturbation']}/{tot})")
    print(f"  Salida Divergente / Disrupción                       : {e3['other_divergent_pct']:.2f}% ({e3['counts']['other_divergent']}/{tot})")

    print("\n[EXP 4: INCONGRUENCIA CAUSAL (Subtotales de Ejemplo Donante)]")
    print(f"  Inercia al Prompt Original (Acertó pese a subtotales falsos): {e4['prompt_inertia_pct']:.2f}% ({e4['counts']['prompt_inertia']}/{tot})")
    print(f"  Arrastre por Contexto Donante (Produjo el total del donante) : {e4['donor_drag_pct']:.2f}% ({e4['counts']['donor_drag']}/{tot})")
    print(f"  Disrupción Cruzada (Salida divergente / no correlacionada)  : {e4['cross_disruption_pct']:.2f}% ({e4['counts']['cross_disruption']}/{tot})")
    print("=" * 80 + "\n")


def print_multi_seed_summary(agg_res: Dict[str, Any]):
    """Imprime el resumen consolidado de las 5 semillas."""
    print("\n" + "#" * 80)
    print("RESUMEN CONSOLIDADO DE AUDITORÍA MULTI-SEMILLA (5 Semillas)")
    print("#" * 80)
    seeds = agg_res["seeds_evaluated"]
    print(f"Semillas evaluadas: {seeds}\n")

    print("TABLA COMPARATIVA POR SEMILLA:")
    print("+--------+---------+---------+----------+-----------+------------+------------+")
    print("| Semilla| Full-EM | SP-EM   | Final-EM | N_10/Err  | Ganancia GT| Sensib(+10)|")
    print("+--------+---------+---------+----------+-----------+------------+------------+")
    for s in seeds:
        data = agg_res["per_seed"][str(s)]
        e1 = data["exp1_real"]
        cm = e1["contingency_matrix"]
        e2 = data["exp2_oracle_ground_truth"]
        e3 = data["exp3_causal_sensitivity"]
        print(
            f"| {s:6} | {e1['exact_match_total_pct']:6.2f}% | {e1['scratchpad_em_pct']:6.2f}% | "
            f"{e1['final_em_pct']:7.2f}% | {cm['n_10_pct_of_final_errors']:8.2f}% | "
            f"{e2['net_gain_pp']:+9.2f}pp | {e3['strict_dual_channel_pct']:9.2f}% |"
        )
    print("+--------+---------+---------+----------+-----------+------------+------------+")

    agg = agg_res["aggregated_metrics"]
    print("\nESTADÍSTICAS AGREGADAS (Media ± Desv.Est [Mín - Máx]):")
    for key, val in agg.items():
        print(f"  - {key:32s}: {val['mean']:6.2f}% ± {val['std']:5.2f}%  [{val['min']:5.2f}% .. {val['max']:5.2f}%]")
    print("#" * 80 + "\n")


def compute_aggregate_stats(per_seed_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula medias, desviaciones estándar, mínimos y máximos entre semillas."""
    import math

    metric_keys = {
        "full_em_pct": lambda d: d["exp1_real"]["exact_match_total_pct"],
        "scratchpad_em_pct": lambda d: d["exp1_real"]["scratchpad_em_pct"],
        "final_em_pct": lambda d: d["exp1_real"]["final_em_pct"],
        "pct_errors_with_correct_sp": lambda d: d["exp1_real"]["contingency_matrix"]["n_10_pct_of_final_errors"],
        "oracle_ground_truth_final_em": lambda d: d["exp2_oracle_ground_truth"]["final_em_pct"],
        "oracle_net_gain_pp": lambda d: d["exp2_oracle_ground_truth"]["net_gain_pp"],
        "causal_sensitivity_strict_dual": lambda d: d["exp3_causal_sensitivity"]["strict_dual_channel_pct"],
        "prompt_inertia_pct": lambda d: d["exp4_shuffled_donor"]["prompt_inertia_pct"],
    }

    aggregated = {}
    n = len(per_seed_dict)
    for name, extractor in metric_keys.items():
        vals = [extractor(v) for v in per_seed_dict.values()]
        mean = sum(vals) / n
        var = sum((x - mean) ** 2 for x in vals) / (n - 1) if n > 1 else 0.0
        std = math.sqrt(var)
        aggregated[name] = {
            "mean": round(mean, 2),
            "std": round(std, 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "values": [round(x, 2) for x in vals],
        }
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Protocolo de Auditoría Causal de Composición (TRDM-v6)")
    parser.add_argument("--ckpt", type=str, default="", help="Ruta a un checkpoint individual")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="/content/drive/MyDrive/BasicLite",
        help="Directorio de checkpoints para evaluación multi-semilla",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456, 789, 999],
        help="Semillas a evaluar si se usa --ckpt_dir",
    )
    parser.add_argument("--split", type=str, default="test-composicional", help="Partición a evaluar")
    parser.add_argument("--delta", type=int, default=10, help="Delta de perturbación para Exp 3")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu o cuda")
    parser.add_argument(
        "--out_json",
        type=str,
        default="auditoria_composicion_multisemilla_results.json",
        help="Ruta para exportar el informe JSON",
    )
    args = parser.parse_args()

    # Determinismo
    set_seed(42)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[*] Dispositivo activo para auditoría: {device}")

    # Determinar modo de ejecución: individual o multi-semilla
    ckpt_list: List[Tuple[str, str]] = []  # (seed_label, ckpt_path)

    if args.ckpt and os.path.exists(args.ckpt):
        seed_label = "individual"
        for s in args.seeds:
            if f"seed{s}" in args.ckpt:
                seed_label = str(s)
                break
        ckpt_list.append((seed_label, args.ckpt))
    elif os.path.exists(args.ckpt_dir):
        for s in args.seeds:
            pattern = os.path.join(args.ckpt_dir, f"trdm_v6_seed{s}_best.pt")
            if os.path.exists(pattern):
                ckpt_list.append((str(s), pattern))
            else:
                # Buscar alternativas
                alt_pattern = os.path.join(args.ckpt_dir, f"*seed{s}*.pt")
                matches = glob.glob(alt_pattern)
                if matches:
                    ckpt_list.append((str(s), matches[0]))
    else:
        # Fallback local
        if os.path.exists("trdm_v6_checkpoint.pt"):
            ckpt_list.append(("local", "trdm_v6_checkpoint.pt"))

    if not ckpt_list:
        print(f"ERROR: No se encontraron checkpoints en '{args.ckpt}' ni en '{args.ckpt_dir}'.")
        sys.exit(1)

    print(f"[*] Se evaluarán {len(ckpt_list)} checkpoints: {[c[1] for c in ckpt_list]}")

    per_seed_results: Dict[str, Any] = {}

    for seed_label, ckpt_path in ckpt_list:
        print(f"\n---> Cargando checkpoint para semilla {seed_label} desde: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt.get("cfg", HybridConfig())
        cfg.device = str(device)

        model = TRDMv6CausalModel(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])

        dataset = ProceduralCompositionalDatasetV2(cfg, split=args.split)
        res = run_composition_audit(model, dataset, device, delta_perturb=args.delta)
        per_seed_results[seed_label] = res
        print_audit_report(seed_label, args.split, res)

    # Si hay múltiples semillas, calcular agregados
    aggregated = compute_aggregate_stats(per_seed_results) if len(per_seed_results) > 1 else {}
    if aggregated:
        agg_payload = {
            "seeds_evaluated": [c[0] for c in ckpt_list],
            "per_seed": per_seed_results,
            "aggregated_metrics": aggregated,
        }
        print_multi_seed_summary(agg_payload)

    # Exportar JSON consolidado
    final_output = {
        "experiment": "auditoria_causal_composicion_v6",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "split": args.split,
        "delta_perturb": args.delta,
        "seeds_evaluated": [c[0] for c in ckpt_list],
        "per_seed_results": per_seed_results,
        "aggregated_summary": aggregated,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"[*] Informe completo JSON exportado con éxito a: {args.out_json}")


if __name__ == "__main__":
    main()
