import time
from typing import Dict, Any, List, Tuple
import torch
from config import HybridConfig
from data_v2 import ProceduralCompositionalDatasetV2, parse_fixed_scratchpad
from trdm.models.trdm_v6_causal import TRDMv6CausalModel


def evaluate_causal_v6(
    model: TRDMv6CausalModel,
    dataset: ProceduralCompositionalDatasetV2,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluación rigurosa del modelo TRDM-v6 bajo inferencia causal de 2 pasos.
    Calcula Exact Match Total, Scratchpad EM, Final EM, Exactitud de Dígitos,
    Validez de Formato y Latencia.
    """
    model.eval()
    total_cases = len(dataset)
    em_total_count = 0
    sp_em_count = 0
    final_em_count = 0
    fmt_valid_count = 0

    total_digits = 0
    matching_digits = 0
    total_time_ms = 0.0

    samples = dataset.samples

    for idx in range(total_cases):
        prompt_b, target_b, expected_watt, expected_ms = samples[idx]
        sub_w, sub_ms, _, _ = dataset.record_meta.get(
            (prompt_b, target_b), (None, None, expected_watt, expected_ms)
        )

        t0 = time.perf_counter()
        gen_str, meta = model.generate_causal_two_step(prompt_b)
        t1 = time.perf_counter()
        total_time_ms += (t1 - t0) * 1000.0

        valid, pred_pw, pred_pms, pred_watt, pred_ms = parse_fixed_scratchpad(gen_str)

        if valid:
            fmt_valid_count += 1

            # Comparación de Subtotales (Etapa 1)
            sp_match = (pred_pw == sub_w) and (pred_pms == sub_ms)
            if sp_match:
                sp_em_count += 1

            # Comparación de Totales Finales (Etapa 2)
            fn_match = (pred_watt == expected_watt) and (pred_ms == expected_ms)
            if fn_match:
                final_em_count += 1

            # Exact Match Total
            if sp_match and fn_match:
                em_total_count += 1

        # Comparación granular de dígitos
        expected_target_str = target_b.decode("utf-8")
        # Ambos deben tener los 12 dígitos en sus posiciones
        # Target: pw: XXX\npms: YYY\nwatt: ZZZ\nms: WWW\n
        digit_indices = [
            (4, 7),   # pw
            (13, 16), # pms
            (23, 26), # watt
            (31, 34), # ms
        ]

        for s_start, s_end in digit_indices:
            exp_sub = expected_target_str[s_start:s_end]
            pred_sub = gen_str[s_start:s_end] if len(gen_str) >= s_end else "???"
            for e_char, p_char in zip(exp_sub, pred_sub):
                total_digits += 1
                if e_char == p_char:
                    matching_digits += 1

    em_pct = (em_total_count / total_cases) * 100.0
    sp_em_pct = (sp_em_count / total_cases) * 100.0
    final_em_pct = (final_em_count / total_cases) * 100.0
    fmt_valid_pct = (fmt_valid_count / total_cases) * 100.0
    digit_acc_pct = (matching_digits / total_digits) * 100.0 if total_digits > 0 else 0.0
    ms_per_case = total_time_ms / total_cases if total_cases > 0 else 0.0

    return {
        "exact_match_pct": em_pct,
        "scratchpad_exact_pct": sp_em_pct,
        "final_exact_pct": final_em_pct,
        "fmt_valid_pct": fmt_valid_pct,
        "digit_acc_pct": digit_acc_pct,
        "ms_per_case": ms_per_case,
        "em_count": em_total_count,
        "total_cases": total_cases,
    }
