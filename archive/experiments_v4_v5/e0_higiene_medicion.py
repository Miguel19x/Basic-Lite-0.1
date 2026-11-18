#!/usr/bin/env python3
# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Suite Formal de Higiene de Medición y Contratos del Sistema (E0)
Protocolo pre-registrado — Basic-Lite (TRDM-v5), Fase 1

Pruebas:
- E0.1 Causalidad temporal diferenciable: d(logits_i) / d(x_embed_j) == 0 (<= 1e-9) para j > i.
- E0.2 Equivalencia numérica de caché: Inferencia incremental con KV-cache vs. recálculo paralelo (<= 1e-4 en FP32).
- E0.3 Independencia estricta de lote: Invarianza de logits (<= 1e-5) de S_0 al variar el resto del lote.
- E0.4 Integridad de analizadores y formato: Prueba de ida y vuelta determinista UTF-8 y analizador estricto de ancho fijo.
- E0.5 Validación criptográfica de datos: Disyunción SHA-256 de particiones y cálculo formal de techos triviales sobre train.
"""

import collections
import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

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


def print_banner(text: str) -> None:
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def run_e0_1_causality(model: nn.Module, cfg: HybridConfig) -> Dict[str, Any]:
    """E0.1 Causalidad temporal diferenciable wrt continuous embeddings."""
    print("\n[E0.1] Verificando Causalidad Temporal Diferenciable...")
    results = {}
    B_blk = cfg.block_size
    num_blocks = cfg.max_seq_len // B_blk

    for mode_name, is_train in [("train", True), ("eval", False)]:
        if is_train:
            model.train()
        else:
            model.eval()

        saved_emb = []

        def emb_hook(module, inp, out):
            out.retain_grad()
            saved_emb.append(out)

        handle = model.token_emb.register_forward_hook(emb_hook)

        torch.manual_seed(42)
        x = torch.randint(0, cfg.vocab_size - 4, (1, cfg.max_seq_len))
        t = torch.full((1, cfg.max_seq_len), 0.50)

        model.zero_grad(set_to_none=True)
        saved_emb.clear()

        # Forward pass (enable grad even if eval mode)
        with torch.enable_grad():
            logits, _, _, _, _ = model(x, t, num_recursions=2)
            emb = saved_emb[0]

            max_leakage_overall = 0.0
            block_leakages = {}

            # Para cada bloque temporal i (0 .. num_blocks - 2)
            for i in range(num_blocks - 1):
                i_start = i * B_blk
                i_end = (i + 1) * B_blk
                loss_i = logits[0, i_start:i_end].sum()

                model.zero_grad(set_to_none=True)
                loss_i.backward(retain_graph=True)

                # Verificar gradiente respecto a bloques futuros j > i
                future_start = (i + 1) * B_blk
                future_grad = emb.grad[0, future_start:]
                max_leak = float(future_grad.abs().max().item()) if future_grad.numel() > 0 else 0.0
                block_leakages[f"block_{i}_to_future"] = max_leak
                if max_leak > max_leakage_overall:
                    max_leakage_overall = max_leak

        handle.remove()
        passed = bool(max_leakage_overall <= 1e-9)
        results[mode_name] = {
            "max_future_gradient": max_leakage_overall,
            "passed": passed,
            "block_leakages": block_leakages,
        }
        status_str = "PASS" if passed else "FAIL"
        print(f"  - Modo {mode_name:5s}: max gradiente futuro = {max_leakage_overall:.2e} -> [{status_str}]")

    return results


def run_e0_2_cache_equivalence(model: nn.Module, cfg: HybridConfig) -> Dict[str, Any]:
    """E0.2 Equivalencia numérica entre inferencia paralela y KV-cache incremental."""
    print("\n[E0.2] Verificando Equivalencia Numérica de Caché KV...")
    model.eval()
    results = {}
    B_blk = cfg.block_size
    num_blocks = cfg.max_seq_len // B_blk

    for bs in [1, 4, 8]:
        torch.manual_seed(100 + bs)
        x_test = torch.randint(10, 200, (bs, cfg.max_seq_len))
        t_test = torch.full((bs, cfg.max_seq_len), 0.50)

        with torch.no_grad():
            logits_par, _, _, _, _ = model(x_test, t_test, num_recursions=2)

            logits_kv_list = []
            kv_caches = None
            committed_len = 0
            for b in range(num_blocks):
                blk_tokens = x_test[:, b * B_blk : (b + 1) * B_blk]
                t_b = t_test[:, b * B_blk : (b + 1) * B_blk]
                l_b, _, _, kv_caches, _ = model(
                    blk_tokens,
                    t_b,
                    num_recursions=2,
                    kv_caches_per_r=kv_caches,
                    start_pos=committed_len,
                )
                logits_kv_list.append(l_b)
                committed_len += B_blk

            logits_kv = torch.cat(logits_kv_list, dim=1)
            diff = float((logits_par - logits_kv).abs().max().item())

        passed = bool(diff <= 1e-4)
        results[f"batch_{bs}"] = {
            "max_abs_diff": diff,
            "passed": passed,
        }
        status_str = "PASS" if passed else "FAIL"
        print(f"  - Batch size {bs}: max diff = {diff:.2e} -> [{status_str}]")

    return results


def run_e0_3_batch_independence(model: nn.Module, cfg: HybridConfig) -> Dict[str, Any]:
    """E0.3 Independencia estricta entre muestras de lote."""
    print("\n[E0.3] Verificando Independencia Estricta de Muestras en Lote...")
    model.eval()

    torch.manual_seed(777)
    x_pool = torch.randint(10, 200, (8, cfg.max_seq_len))
    t_pool = torch.full((8, cfg.max_seq_len), 0.50)

    # Evaluación de S_0 en Lote A: [S_0, S_1, S_2, S_3]
    batch_a = x_pool[[0, 1, 2, 3]]
    t_a = t_pool[[0, 1, 2, 3]]
    with torch.no_grad():
        torch.manual_seed(42)
        out_a, _, _, _, _ = model(batch_a, t_a, num_recursions=2)
        logits_s0_a = out_a[0].clone()

    # Evaluación de S_0 en Lote B: [S_0, S_4, S_5, S_6]
    batch_b = x_pool[[0, 4, 5, 6]]
    t_b = t_pool[[0, 4, 5, 6]]
    with torch.no_grad():
        torch.manual_seed(42)
        out_b, _, _, _, _ = model(batch_b, t_b, num_recursions=2)
        logits_s0_b = out_b[0].clone()

    # Evaluación de S_0 aislado (Batch 1)
    batch_solo = x_pool[[0]]
    t_solo = t_pool[[0]]
    with torch.no_grad():
        torch.manual_seed(42)
        out_solo, _, _, _, _ = model(batch_solo, t_solo, num_recursions=2)
        logits_s0_solo = out_solo[0].clone()

    diff_ab = float((logits_s0_a - logits_s0_b).abs().max().item())
    diff_solo = float((logits_s0_a - logits_s0_solo).abs().max().item())

    passed = bool(diff_ab <= 1e-5 and diff_solo <= 1e-5)
    status_str = "PASS" if passed else "FAIL"
    print(f"  - Max diff Lote A vs Lote B: {diff_ab:.2e}")
    print(f"  - Max diff Lote A vs Solo:   {diff_solo:.2e} -> [{status_str}]")

    return {
        "max_diff_batch_ab": diff_ab,
        "max_diff_batch_solo": diff_solo,
        "passed": passed,
    }


def run_e0_4_parsers(cfg: HybridConfig) -> Dict[str, Any]:
    """E0.4 Integridad determinista de codificación UTF-8 y analizadores de ancho fijo."""
    print("\n[E0.4] Verificando Integridad de Analizadores y Formato de Ancho Fijo...")
    ds = ProceduralCompositionalDatasetV2(cfg, split="all")
    total_records = len(ds.unique_records)
    parsed_ok_finals = 0
    parsed_ok_scratchpad = 0

    range_watts = []
    range_ms = []

    for prompt_b, target_b, watt_true, ms_true in ds.unique_records:
        range_watts.append(watt_true)
        range_ms.append(ms_true)

        target_str = target_b.decode("utf-8")
        ok_f, w_p, ms_p = parse_fixed_prediction(target_str)
        if ok_f and w_p == watt_true and ms_p == ms_true:
            parsed_ok_finals += 1

        if getattr(cfg, "use_scratchpad", False):
            sub_w, sub_ms, _, _ = ds.record_meta[(prompt_b, target_b)]
            ok_s, pw_p, pms_p, w_p, ms_p = parse_fixed_scratchpad(target_str)
            if ok_s and pw_p == sub_w and pms_p == sub_ms and w_p == watt_true and ms_p == ms_true:
                parsed_ok_scratchpad += 1

    pct_final = 100.0 * parsed_ok_finals / total_records
    passed = (parsed_ok_finals == total_records)
    if getattr(cfg, "use_scratchpad", False):
        passed = passed and (parsed_ok_scratchpad == total_records)

    status_str = "PASS" if passed else "FAIL"
    print(f"  - Registros evaluados: {total_records}")
    print(f"  - Parser de respuestas finales: {parsed_ok_finals}/{total_records} ({pct_final:.1f}%)")
    if getattr(cfg, "use_scratchpad", False):
        pct_sc = 100.0 * parsed_ok_scratchpad / total_records
        print(f"  - Parser de scratchpad:        {parsed_ok_scratchpad}/{total_records} ({pct_sc:.1f}%)")
    print(f"  - Rango de Watt: [{min(range_watts)}, {max(range_watts)}] (debe estar en [1, 999])")
    print(f"  - Rango de MS:   [{min(range_ms)}, {max(range_ms)}] (debe estar en [1, 999]) -> [{status_str}]")

    return {
        "total_evaluated": total_records,
        "parsed_finals_ok": parsed_ok_finals,
        "parsed_scratchpad_ok": parsed_ok_scratchpad if getattr(cfg, "use_scratchpad", False) else None,
        "min_watt": min(range_watts),
        "max_watt": max(range_watts),
        "min_ms": min(range_ms),
        "max_ms": max(range_ms),
        "in_3_digits_range": bool(min(range_watts) >= 1 and max(range_watts) <= 999 and min(range_ms) >= 1 and max(range_ms) <= 999),
        "passed": passed,
    }


def run_e0_5_data_disjunction_and_baselines(cfg: HybridConfig) -> Dict[str, Any]:
    """E0.5 Validación criptográfica de particiones y cálculo de techos triviales sobre train."""
    print("\n[E0.5] Verificando Disyunción Criptográfica SHA-256 y Techos Triviales...")

    ds_train = ProceduralCompositionalDatasetV2(cfg, split="train")
    ds_dev = ProceduralCompositionalDatasetV2(cfg, split="dev")
    ds_azar = ProceduralCompositionalDatasetV2(cfg, split="test-azar")
    ds_comp = ProceduralCompositionalDatasetV2(cfg, split="test-composicional")

    def get_hashes(unique_records: List[Tuple[bytes, bytes, int, int]]) -> Set[str]:
        h_set = set()
        for p_b, t_b, _, _ in unique_records:
            h = hashlib.sha256(p_b + b"||" + t_b).hexdigest()
            h_set.add(h)
        return h_set

    h_train = get_hashes(ds_train.unique_records)
    h_dev = get_hashes(ds_dev.unique_records)
    h_azar = get_hashes(ds_azar.unique_records)
    h_comp = get_hashes(ds_comp.unique_records)

    n_train = len(h_train)
    n_dev = len(h_dev)
    n_azar = len(h_azar)
    n_comp = len(h_comp)
    total_unique = n_train + n_dev + n_azar + n_comp

    print(f"  - Conteos únicos de partición: train={n_train}, dev={n_dev}, test-azar={n_azar}, test-comp={n_comp}")
    print(f"  - Suma total de combinaciones únicas: {total_unique} (requerido: 2880)")

    # Comprobación de intersección nula
    inter_train_comp = h_train.intersection(h_comp)
    inter_train_azar = h_train.intersection(h_azar)
    inter_train_dev = h_train.intersection(h_dev)
    inter_comp_azar = h_comp.intersection(h_azar)
    inter_comp_dev = h_comp.intersection(h_dev)
    inter_azar_dev = h_azar.intersection(h_dev)

    disjunction_ok = (
        len(inter_train_comp) == 0 and
        len(inter_train_azar) == 0 and
        len(inter_train_dev) == 0 and
        len(inter_comp_azar) == 0 and
        len(inter_comp_dev) == 0 and
        len(inter_azar_dev) == 0 and
        total_unique == 2880
    )

    # Cálculo formal de techos triviales estrictamente sobre train
    # 1. Moda por posición de dígito
    # Analizamos la distribución de cada posición en target_b
    digit_positions_counts = collections.defaultdict(collections.Counter)
    watt_values = []
    ms_values = []

    for _, t_b, w, ms in ds_train.unique_records:
        t_str = t_b.decode("utf-8")
        watt_values.append(w)
        ms_values.append(ms)
        for pos, ch in enumerate(t_str):
            if ch.isdigit():
                digit_positions_counts[pos][ch] += 1

    modal_digits = {pos: ctr.most_common(1)[0][0] for pos, ctr in digit_positions_counts.items()}
    modal_watt = collections.Counter(watt_values).most_common(1)[0][0]
    modal_ms = collections.Counter(ms_values).most_common(1)[0][0]

    print(f"  - Valores modales calculados sobre train: watt_modal={modal_watt:03d}, ms_modal={modal_ms:03d}")

    # Evaluar los techos triviales sobre test-composicional
    comp_records = ds_comp.unique_records
    n_comp_total = len(comp_records)

    exact_match_modal_comp = sum(1 for _, _, w, ms in comp_records if (w == modal_watt and ms == modal_ms))
    pct_em_modal_comp = 100.0 * exact_match_modal_comp / n_comp_total

    # Exactitud de dígitos por moda posicional en test-comp
    correct_pos_digits = 0
    total_pos_digits = 0
    for _, t_b, _, _ in comp_records:
        t_str = t_b.decode("utf-8")
        for pos, ch in enumerate(t_str):
            if ch.isdigit():
                total_pos_digits += 1
                if modal_digits.get(pos, None) == ch:
                    correct_pos_digits += 1

    pct_pos_digit_acc_comp = 100.0 * correct_pos_digits / max(1, total_pos_digits)

    # Evaluar los techos triviales sobre test-azar
    azar_records = ds_azar.unique_records
    n_azar_total = len(azar_records)
    exact_match_modal_azar = sum(1 for _, _, w, ms in azar_records if (w == modal_watt and ms == modal_ms))
    pct_em_modal_azar = 100.0 * exact_match_modal_azar / n_azar_total

    print(f"  - Techo Trivial (Exact Match por Moda) en test-comp: {pct_em_modal_comp:.2f}% ({exact_match_modal_comp}/{n_comp_total})")
    print(f"  - Techo Trivial (Exactitud de Dígitos)  en test-comp: {pct_pos_digit_acc_comp:.2f}%")
    print(f"  - Techo Trivial (Exact Match por Moda) en test-azar: {pct_em_modal_azar:.2f}% ({exact_match_modal_azar}/{n_azar_total})")

    status_str = "PASS" if disjunction_ok else "FAIL"
    print(f"  - Disyunción Criptográfica SHA-256: [{status_str}]")

    return {
        "partitions_counts": {
            "train": n_train,
            "dev": n_dev,
            "test_azar": n_azar,
            "test_composicional": n_comp,
            "total_unique": total_unique,
        },
        "disjunction_passed": disjunction_ok,
        "trivial_baselines": {
            "modal_watt": modal_watt,
            "modal_ms": modal_ms,
            "test_comp": {
                "exact_match_pct": pct_em_modal_comp,
                "digit_accuracy_pct": pct_pos_digit_acc_comp,
            },
            "test_azar": {
                "exact_match_pct": pct_em_modal_azar,
            },
        },
        "passed": disjunction_ok,
    }


def main():
    print_banner("EJECUCIÓN DE LA SUITE DE HIGIENE DE MEDICIÓN (E0) — BASIC-LITE")
    torch.set_num_threads(2)
    torch.manual_seed(42)

    cfg = HybridConfig(
        max_seq_len=160,
        use_scratchpad=True,
        use_explicit_operands=False,
        cpu_threads=2,
        dev_mode=False,
    )

    print(f"Inicializando modelo de prueba ({cfg.d_model}d, {cfg.n_heads}h, max_seq_len={cfg.max_seq_len})...")
    model = TernaryRecursiveDiffusionLM(cfg)

    t0 = time.time()
    res_e0_1 = run_e0_1_causality(model, cfg)
    res_e0_2 = run_e0_2_cache_equivalence(model, cfg)
    res_e0_3 = run_e0_3_batch_independence(model, cfg)
    res_e0_4 = run_e0_4_parsers(cfg)
    res_e0_5 = run_e0_5_data_disjunction_and_baselines(cfg)
    elapsed = time.time() - t0

    all_passed = (
        res_e0_1["train"]["passed"] and
        res_e0_1["eval"]["passed"] and
        all(v["passed"] for v in res_e0_2.values()) and
        res_e0_3["passed"] and
        res_e0_4["passed"] and
        res_e0_5["passed"]
    )

    final_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": elapsed,
        "all_passed": all_passed,
        "E0_1_causality": res_e0_1,
        "E0_2_cache_equivalence": res_e0_2,
        "E0_3_batch_independence": res_e0_3,
        "E0_4_parsers": res_e0_4,
        "E0_5_data_and_baselines": res_e0_5,
    }

    out_path = "e0_higiene_results.json"
    with open(out_path, "w") as f:
        json.dump(final_report, f, indent=2)

    print_banner("VEREDICTO FINAL DE LA PUERTA E0")
    if all_passed:
        print(f"  >>> PUERTA E0 SUPERADA AL 100% (Tiempo: {elapsed:.2f}s) <<<")
        print(f"  Informe guardado en: {out_path}")
        sys.exit(0)
    else:
        print(f"  >>> ALERTA CRÍTICA: FALLÓ LA PUERTA E0 (Tiempo: {elapsed:.2f}s) <<<")
        print(f"  Revisar detalles en: {out_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
