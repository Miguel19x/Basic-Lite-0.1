import math
import random
import re
import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from config import HybridConfig
from model import TernaryRecursiveDiffusionLM
from data import ProceduralCompositionalDataset
from trdm.diffusion.sampler import generate_block_causal_with_kv_cache
from trdm.data.parsers import (
    parse_watt_ms_prediction,
    parse_scratchpad_prediction,
)

_PW_PMS_RE = re.compile(r"pw:\s*(\d+)\s*\npms:\s*(\d+)")
_WATT_MS_RE = re.compile(r"watt:\s*(\d+)\s*\nms:\s*(\d+)")


def parse_only_subtotals(text: str) -> Tuple[bool, Optional[int], Optional[int]]:
    m = _PW_PMS_RE.search(text)
    if m:
        try:
            return True, int(m.group(1)), int(m.group(2))
        except ValueError:
            pass
    return False, None, None


def parse_only_finals(text: str) -> Tuple[bool, Optional[int], Optional[int]]:
    m = _WATT_MS_RE.search(text)
    if m:
        try:
            return True, int(m.group(1)), int(m.group(2))
        except ValueError:
            pass
    return False, None, None


def compute_isolated_digit_acc(pred_str: str, target_str: str) -> Tuple[int, int]:
    """Calcula aciertos y total de dígitos exclusivamente en posiciones donde target_str tiene un dígito."""
    correct, total = 0, 0
    for idx, ch in enumerate(target_str):
        if ch.isdigit():
            total += 1
            if idx < len(pred_str) and pred_str[idx] == ch:
                correct += 1
    return correct, total


def run_matriz_ruptura():
    cfg = HybridConfig()
    cfg.seed = 42
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)

    print("=== Cargando Checkpoint para la Matriz de Diagnóstico ===")
    model = TernaryRecursiveDiffusionLM(cfg).to(cfg.device)
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.load_bitcos_archive(cfg.bitcos_export_path)
    model.prepare_for_eval()

    train_ds = ProceduralCompositionalDataset(cfg, split="train", num_samples=100)
    val_ds = ProceduralCompositionalDataset(cfg, split="val", num_samples=100)

    # 96 tuplas de train y 96 tuplas de held-out
    rng = random.Random(cfg.seed)
    train_records = rng.sample(train_ds.unique_records, min(96, len(train_ds.unique_records)))
    val_records = list(val_ds.unique_records)

    splits = [("Train (96 casos)", train_records), ("Held-Out OOD (96 casos)", val_records)]

    for split_name, records in splits:
        print(f"\n=======================================================")
        print(f"  EVALUACIÓN DE MATRIZ: {split_name}")
        print(f"=======================================================")

        n = len(records)
        c1_pw_em, c1_pms_em, c1_both_em = 0, 0, 0
        c1_pw_dig_c, c1_pw_dig_t = 0, 0
        c1_pms_dig_c, c1_pms_dig_t = 0, 0
        c1_fmt = 0

        c2_w_em, c2_ms_em, c2_both_em = 0, 0, 0
        c2_w_dig_c, c2_w_dig_t = 0, 0
        c2_ms_dig_c, c2_ms_dig_t = 0, 0
        c2_fmt = 0

        c3_w_em, c3_ms_em, c3_both_em = 0, 0, 0
        c3_w_dig_c, c3_w_dig_t = 0, 0
        c3_ms_dig_c, c3_ms_dig_t = 0, 0
        c3_fmt = 0
        c3_contingency = {"sub_ok_final_ok": 0, "sub_ok_final_err": 0, "sub_err_final_ok": 0, "sub_err_final_err": 0}

        c4_pw_em, c4_pms_em, c4_sub_both = 0, 0, 0
        c4_w_em, c4_ms_em, c4_both_em = 0, 0, 0
        c4_all_em = 0
        c4_trace_fmt, c4_final_fmt, c4_full_fmt = 0, 0, 0
        c4_pw_dig_c, c4_pw_dig_t = 0, 0
        c4_pms_dig_c, c4_pms_dig_t = 0, 0
        c4_w_dig_c, c4_w_dig_t = 0, 0
        c4_ms_dig_c, c4_ms_dig_t = 0, 0

        for prompt_b, target_b, true_watt, true_ms in records:
            p_str = prompt_b.decode("utf-8")
            t_str = target_b.decode("utf-8")
            # Extraer subtotales oráculo del target
            _, true_pw, true_pms, _, _ = parse_scratchpad_prediction(t_str, "")
            assert true_pw is not None and true_pms is not None

            # Sub-strings objetivo aislados
            target_sub_str = f"pw: {true_pw}\npms: {true_pms}\n"
            target_resp_str = f"watt: {true_watt}\nms: {true_ms}\n"

            # -------------------------------------------------------------
            # CONDICIÓN 1: Factores -> Generar solo subtotales (pw, pms)
            # -------------------------------------------------------------
            gen1, _, _ = generate_block_causal_with_kv_cache(model, p_str, cfg, gen_length=24, num_recursions_override=2)
            cont1 = gen1[len(p_str):] if gen1.startswith(p_str) else gen1
            fmt1, p1_pw, p1_pms = parse_only_subtotals(cont1)
            if fmt1:
                c1_fmt += 1
            if p1_pw == true_pw:
                c1_pw_em += 1
            if p1_pms == true_pms:
                c1_pms_em += 1
            if p1_pw == true_pw and p1_pms == true_pms:
                c1_both_em += 1

            # DigAcc aislada de pw y pms
            m_pw = re.search(r"pw:\s*(\d+)", cont1)
            if m_pw:
                c, t = compute_isolated_digit_acc(m_pw.group(1), str(true_pw))
                c1_pw_dig_c += c; c1_pw_dig_t += t
            else:
                c1_pw_dig_t += len(str(true_pw))
            m_pms = re.search(r"pms:\s*(\d+)", cont1)
            if m_pms:
                c, t = compute_isolated_digit_acc(m_pms.group(1), str(true_pms))
                c1_pms_dig_c += c; c1_pms_dig_t += t
            else:
                c1_pms_dig_t += len(str(true_pms))

            # -------------------------------------------------------------
            # CONDICIÓN 2: Factores + Subtotales Oráculo -> Generar respuesta final (watt, ms)
            # -------------------------------------------------------------
            prompt_c2 = p_str + target_sub_str
            gen2, _, _ = generate_block_causal_with_kv_cache(model, prompt_c2, cfg, gen_length=24, num_recursions_override=2)
            cont2 = gen2[len(prompt_c2):] if gen2.startswith(prompt_c2) else gen2
            fmt2, p2_w, p2_ms = parse_only_finals(cont2)
            if fmt2:
                c2_fmt += 1
            if p2_w == true_watt:
                c2_w_em += 1
            if p2_ms == true_ms:
                c2_ms_em += 1
            if p2_w == true_watt and p2_ms == true_ms:
                c2_both_em += 1

            m_w2 = re.search(r"watt:\s*(\d+)", cont2)
            if m_w2:
                c, t = compute_isolated_digit_acc(m_w2.group(1), str(true_watt))
                c2_w_dig_c += c; c2_w_dig_t += t
            else:
                c2_w_dig_t += len(str(true_watt))
            m_ms2 = re.search(r"ms:\s*(\d+)", cont2)
            if m_ms2:
                c, t = compute_isolated_digit_acc(m_ms2.group(1), str(true_ms))
                c2_ms_dig_c += c; c2_ms_dig_t += t
            else:
                c2_ms_dig_t += len(str(true_ms))

            # -------------------------------------------------------------
            # CONDICIÓN 3: Factores + Subtotales Predichos (en C1) -> Generar respuesta final
            # -------------------------------------------------------------
            sub_pred_str = f"pw: {p1_pw if p1_pw is not None else 0}\npms: {p1_pms if p1_pms is not None else 0}\n"
            prompt_c3 = p_str + sub_pred_str
            gen3, _, _ = generate_block_causal_with_kv_cache(model, prompt_c3, cfg, gen_length=24, num_recursions_override=2)
            cont3 = gen3[len(prompt_c3):] if gen3.startswith(prompt_c3) else gen3
            fmt3, p3_w, p3_ms = parse_only_finals(cont3)
            if fmt3:
                c3_fmt += 1
            w3_ok = (p3_w == true_watt)
            ms3_ok = (p3_ms == true_ms)
            if w3_ok:
                c3_w_em += 1
            if ms3_ok:
                c3_ms_em += 1
            if w3_ok and ms3_ok:
                c3_both_em += 1

            sub1_ok = (p1_pw == true_pw and p1_pms == true_pms)
            final3_ok = (w3_ok and ms3_ok)
            if sub1_ok and final3_ok:
                c3_contingency["sub_ok_final_ok"] += 1
            elif sub1_ok and not final3_ok:
                c3_contingency["sub_ok_final_err"] += 1
            elif not sub1_ok and final3_ok:
                c3_contingency["sub_err_final_ok"] += 1
            else:
                c3_contingency["sub_err_final_err"] += 1

            m_w3 = re.search(r"watt:\s*(\d+)", cont3)
            if m_w3:
                c, t = compute_isolated_digit_acc(m_w3.group(1), str(true_watt))
                c3_w_dig_c += c; c3_w_dig_t += t
            else:
                c3_w_dig_t += len(str(true_watt))
            m_ms3 = re.search(r"ms:\s*(\d+)", cont3)
            if m_ms3:
                c, t = compute_isolated_digit_acc(m_ms3.group(1), str(true_ms))
                c3_ms_dig_c += c; c3_ms_dig_t += t
            else:
                c3_ms_dig_t += len(str(true_ms))

            # -------------------------------------------------------------
            # CONDICIÓN 4: Generación Integrada Completa desde cero
            # -------------------------------------------------------------
            gen4, _, _ = generate_block_causal_with_kv_cache(model, p_str, cfg, gen_length=48, num_recursions_override=2)
            cont4 = gen4[len(p_str):] if gen4.startswith(p_str) else gen4
            full_ok4, p4_pw, p4_pms, p4_w, p4_ms = parse_scratchpad_prediction(gen4, p_str)
            fmt_tr4, _, _ = parse_only_subtotals(cont4)
            fmt_fn4, _, _ = parse_only_finals(cont4)

            if full_ok4:
                c4_full_fmt += 1
            if fmt_tr4:
                c4_trace_fmt += 1
            if fmt_fn4:
                c4_final_fmt += 1

            if p4_pw == true_pw:
                c4_pw_em += 1
            if p4_pms == true_pms:
                c4_pms_em += 1
            if p4_pw == true_pw and p4_pms == true_pms:
                c4_sub_both += 1
            if p4_w == true_watt:
                c4_w_em += 1
            if p4_ms == true_ms:
                c4_ms_em += 1
            if p4_w == true_watt and p4_ms == true_ms:
                c4_both_em += 1
            if (p4_pw == true_pw and p4_pms == true_pms and p4_w == true_watt and p4_ms == true_ms):
                c4_all_em += 1

            # Dígitos aislados en C4
            m_pw4 = re.search(r"pw:\s*(\d+)", cont4)
            if m_pw4:
                c, t = compute_isolated_digit_acc(m_pw4.group(1), str(true_pw))
                c4_pw_dig_c += c; c4_pw_dig_t += t
            else:
                c4_pw_dig_t += len(str(true_pw))
            m_pms4 = re.search(r"pms:\s*(\d+)", cont4)
            if m_pms4:
                c, t = compute_isolated_digit_acc(m_pms4.group(1), str(true_pms))
                c4_pms_dig_c += c; c4_pms_dig_t += t
            else:
                c4_pms_dig_t += len(str(true_pms))
            m_w4 = re.search(r"watt:\s*(\d+)", cont4)
            if m_w4:
                c, t = compute_isolated_digit_acc(m_w4.group(1), str(true_watt))
                c4_w_dig_c += c; c4_w_dig_t += t
            else:
                c4_w_dig_t += len(str(true_watt))
            m_ms4 = re.search(r"ms:\s*(\d+)", cont4)
            if m_ms4:
                c, t = compute_isolated_digit_acc(m_ms4.group(1), str(true_ms))
                c4_ms_dig_c += c; c4_ms_dig_t += t
            else:
                c4_ms_dig_t += len(str(true_ms))

        # Imprimir Resultados Desglosados
        print(f"\n--- 1. Condición 1: Solo Subtotales (pw, pms) desde Factores ---")
        print(f"  Formato Traza: {100.0 * c1_fmt / n:.1f}%")
        print(f"  DigAcc Pw: {100.0 * c1_pw_dig_c / max(1, c1_pw_dig_t):.1f}% | EM Pw: {100.0 * c1_pw_em / n:.1f}%")
        print(f"  DigAcc Pms: {100.0 * c1_pms_dig_c / max(1, c1_pms_dig_t):.1f}% | EM Pms: {100.0 * c1_pms_em / n:.1f}%")
        print(f"  Subtotal EM Conjunto (Pw y Pms): {100.0 * c1_both_em / n:.1f}%")

        print(f"\n--- 2. Condición 2: Respuesta (watt, ms) con Subtotales ORÁCULO ---")
        print(f"  Formato Respuesta: {100.0 * c2_fmt / n:.1f}%")
        print(f"  DigAcc Watt: {100.0 * c2_w_dig_c / max(1, c2_w_dig_t):.1f}% | EM Watt: {100.0 * c2_w_em / n:.1f}%")
        print(f"  DigAcc MS: {100.0 * c2_ms_dig_c / max(1, c2_ms_dig_t):.1f}% | EM MS: {100.0 * c2_ms_em / n:.1f}%")
        print(f"  Respuesta Final EM Conjunto: {100.0 * c2_both_em / n:.1f}%")

        print(f"\n--- 3. Condición 3: Respuesta con Subtotales PREDICHOS (2-Step) ---")
        print(f"  Formato Respuesta: {100.0 * c3_fmt / n:.1f}%")
        print(f"  DigAcc Watt: {100.0 * c3_w_dig_c / max(1, c3_w_dig_t):.1f}% | EM Watt: {100.0 * c3_w_em / n:.1f}%")
        print(f"  DigAcc MS: {100.0 * c3_ms_dig_c / max(1, c3_ms_dig_t):.1f}% | EM MS: {100.0 * c3_ms_em / n:.1f}%")
        print(f"  Respuesta Final EM Conjunto: {100.0 * c3_both_em / n:.1f}%")
        print(f"  Contingencia: Sub_OK->Final_OK: {c3_contingency['sub_ok_final_ok']} | Sub_OK->Final_ERR: {c3_contingency['sub_ok_final_err']}")
        print(f"                Sub_ERR->Final_OK: {c3_contingency['sub_err_final_ok']} | Sub_ERR->Final_ERR: {c3_contingency['sub_err_final_err']}")

        print(f"\n--- 4. Condición 4: Generación Integrada Completa ---")
        print(f"  Formato Traza: {100.0 * c4_trace_fmt / n:.1f}% | Formato Respuesta: {100.0 * c4_final_fmt / n:.1f}% | Formato Conjunto: {100.0 * c4_full_fmt / n:.1f}%")
        print(f"  DigAcc: Pw={100.0*c4_pw_dig_c/max(1,c4_pw_dig_t):.1f}%, Pms={100.0*c4_pms_dig_c/max(1,c4_pms_dig_t):.1f}%, Watt={100.0*c4_w_dig_c/max(1,c4_w_dig_t):.1f}%, MS={100.0*c4_ms_dig_c/max(1,c4_ms_dig_t):.1f}%")
        print(f"  EMs: Pw={100.0*c4_pw_em/n:.1f}%, Pms={100.0*c4_pms_em/n:.1f}%, Sub-EM={100.0*c4_sub_both/n:.1f}%")
        print(f"  EMs: Watt={100.0*c4_w_em/n:.1f}%, MS={100.0*c4_ms_em/n:.1f}%, Final-EM={100.0*c4_both_em/n:.1f}%")
        print(f"  All-EM (Cadena Completa): {100.0 * c4_all_em / n:.1f}%")


if __name__ == "__main__":
    run_matriz_ruptura()
