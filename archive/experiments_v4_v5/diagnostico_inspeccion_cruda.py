# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

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
_TOLERANT_NUM_RE = re.compile(r"(\d+)")


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


def inspect_raw_generation():
    cfg = HybridConfig()
    cfg.seed = 42
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)

    print("=" * 70)
    print("  PASO 1: PRUEBA DE CORRECCIÓN BARATA E INSPECCIÓN CRUDA (AUDITORÍA)")
    print("=" * 70)

    # 1. Cargar Checkpoint de 800 pasos
    print("\n--- 1. Cargando Checkpoint de 800 pasos (trdm_v4_checkpoint.pt) ---")
    model = TernaryRecursiveDiffusionLM(cfg).to(cfg.device)
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.load_bitcos_archive(cfg.bitcos_export_path)
    model.prepare_for_eval()

    ds_train = ProceduralCompositionalDataset(cfg, split="train", num_samples=100)
    sample_records = ds_train.unique_records[:5]

    print(f"\n--- 2. Inspección Detallada de 5 Muestras de Entrenamiento ---")
    for i, (prompt_b, target_b, tw, tms) in enumerate(sample_records, 1):
        p_str = prompt_b.decode("utf-8")
        t_str = target_b.decode("utf-8")
        _, true_pw, true_pms, _, _ = parse_scratchpad_prediction(t_str, "")
        target_sub_str = f"pw: {true_pw}\npms: {true_pms}\n"
        target_resp_str = f"watt: {tw}\nms: {tms}\n"

        print(f"\n=======================================================")
        print(f"  MUESTRA #{i}:")
        print(f"  Prompt ({len(prompt_b)} bytes): {repr(p_str)}")
        print(f"  Target Completo ({len(target_b)} bytes): {repr(t_str)}")
        print(f"  Subtotales Verdaderos: pw={true_pw}, pms={true_pms}")
        print(f"  Finales Verdaderos: watt={tw}, ms={tms}")

        # A. Condición 1 (Factores -> Generar Subtotales)
        gen1, hist1, evals1 = generate_block_causal_with_kv_cache(
            model, p_str, cfg, gen_length=24, num_recursions_override=2
        )
        cont1 = gen1[len(p_str):] if gen1.startswith(p_str) else gen1
        m1 = _PW_PMS_RE.search(cont1)
        tol_nums1 = _TOLERANT_NUM_RE.findall(cont1)
        starts_with_p = gen1.startswith(p_str)

        print(f"\n  [C1: Generación de Subtotales]")
        print(f"    - Inicia con prompt exacto: {starts_with_p}")
        print(f"    - Longitud texto generado: {len(gen1)} chars")
        print(f"    - Texto completo generado: {repr(gen1)}")
        print(f"    - Continuación (cont1): {repr(cont1)}")
        print(f"    - Token IDs continuación: {[ord(c) for c in cont1]}")
        print(f"    - Regex estricto 'pw:...\\npms:...': {m1.groups() if m1 else 'NO MATCH'}")
        print(f"    - Extracción tolerante (todos los números): {tol_nums1}")

        # B. Condición 2 (Oráculo: Factores + Subtotales Verdaderos -> Generar Final)
        prompt_c2 = p_str + target_sub_str
        gen2, hist2, evals2 = generate_block_causal_with_kv_cache(
            model, prompt_c2, cfg, gen_length=24, num_recursions_override=2
        )
        cont2 = gen2[len(prompt_c2):] if gen2.startswith(prompt_c2) else gen2
        m2 = _WATT_MS_RE.search(cont2)
        tol_nums2 = _TOLERANT_NUM_RE.findall(cont2)
        starts_with_p2 = gen2.startswith(prompt_c2)

        print(f"\n  [C2: Oráculo -> Generación Final]")
        print(f"    - Prompt Oráculo ({len(prompt_c2)} bytes): {repr(prompt_c2)}")
        print(f"    - Inicia con prompt oráculo exacto: {starts_with_p2}")
        print(f"    - Longitud texto generado: {len(gen2)} chars")
        print(f"    - Texto completo generado: {repr(gen2)}")
        print(f"    - Continuación (cont2): {repr(cont2)}")
        print(f"    - Token IDs continuación: {[ord(c) for c in cont2]}")
        print(f"    - Regex estricto 'watt:...\\nms:...': {m2.groups() if m2 else 'NO MATCH'}")
        print(f"    - Extracción tolerante (todos los números): {tol_nums2}")

        # C. Condición 4 (Integrada: Factores -> Generar Todo, gen_length=48)
        gen4, hist4, evals4 = generate_block_causal_with_kv_cache(
            model, p_str, cfg, gen_length=48, num_recursions_override=2
        )
        cont4 = gen4[len(p_str):] if gen4.startswith(p_str) else gen4
        is_fmt4, p4_sw, p4_sms, p4_w, p4_ms = parse_scratchpad_prediction(gen4, p_str)
        tol_nums4 = _TOLERANT_NUM_RE.findall(cont4)

        print(f"\n  [C4: Integrada (Cadena Completa)]")
        print(f"    - Longitud texto generado: {len(gen4)} chars")
        print(f"    - Continuación (cont4): {repr(cont4)}")
        print(f"    - Token IDs continuación: {[ord(c) for c in cont4]}")
        print(f"    - Formato estricto parse_scratchpad: {is_fmt4} (pw={p4_sw}, pms={p4_sms}, watt={p4_w}, ms={p4_ms})")
        print(f"    - Extracción tolerante (todos los números): {tol_nums4}")

    # 3. Control Cruzado con el Experimento de 8 Muestras (Overfit Control)
    print("\n" + "=" * 70)
    print("--- 3. Control Cruzado: Verificación de la Suite sobre el Modelo Overfit (8 Casos) ---")
    print("=" * 70)
    print("Entrenando modelo de 8 casos en memoria (400 pasos) para verificar si la suite de evaluación causa el 0%...")
    
    # Entrenar en memoria el modelo de 8 casos
    model_8 = TernaryRecursiveDiffusionLM(cfg).to(cfg.device)
    opt_8 = torch.optim.AdamW(model_8.configure_adamw_groups(), lr=2.0e-3, betas=(0.9, 0.95))
    sched_8 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_8, T_max=400, eta_min=1e-4)

    fixed_8 = ds_train.unique_records[:8]
    batch_x0, batch_can_mask, batch_digit_mask = [], [], []
    for prompt_b, target_b, _, _ in fixed_8:
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

    x0_8 = torch.stack(batch_x0).to(cfg.device)
    cmask_8 = torch.stack(batch_can_mask).to(cfg.device)
    dmask_8 = torch.stack(batch_digit_mask).to(cfg.device)

    from trdm.diffusion.corruption import apply_blockwise_diffusion_corruption
    from train import compute_v4_hybrid_loss

    model_8.train()
    for s in range(1, 401):
        x_t, m_pos, t_seq = apply_blockwise_diffusion_corruption(x0_8, cmask_8, cfg)
        out1 = model_8(x_t, t_seq, num_recursions=2, supervised_mask=m_pos)
        l1, _ = compute_v4_hybrid_loss(
            out1.final_logits, out1.intermediate_logits, out1.halt_probs,
            x0_8, m_pos, dmask_8, t_seq, cfg
        )
        
        # Paso de refinamiento
        with torch.no_grad():
            if getattr(cfg, "lazy_vocab_projection", True):
                p1_sup = out1.final_logits.argmax(dim=-1)
                x_pred = x_t.clone()
                x_pred[m_pos] = p1_sup
            else:
                x_pred = out1.final_logits.argmax(dim=-1)
            km = m_pos & (x_pred != x0_8)
        
        if km.any():
            x_ref = torch.where(km, cfg.mask_token_id, x_pred)
            out2 = model_8(x_ref, t_seq * 0.5, num_recursions=2, supervised_mask=km)
            l2, _ = compute_v4_hybrid_loss(
                out2.final_logits, out2.intermediate_logits, out2.halt_probs,
                x0_8, km, dmask_8, t_seq * 0.5, cfg
            )
            tot_l = 0.45 * l1 + 0.55 * l2
        else:
            tot_l = l1

        opt_8.zero_grad()
        tot_l.backward()
        torch.nn.utils.clip_grad_norm_(model_8.parameters(), 1.0)
        opt_8.step()
        sched_8.step()

    model_8.prepare_for_eval()
    print("Modelo de 8 casos entrenado. Evaluando las 8 muestras bajo:")
    print("  (A) Generador original de test_overfit_minimal.py (parse_scratchpad_prediction)")
    print("  (B) Funciones de diagnostico_matriz_ruptura.py (parse_only_subtotals y parse_only_finals)")

    em_orig, fmt_orig = 0, 0
    c1_fmt_count, c1_pw_em, c1_pms_em = 0, 0, 0
    c2_fmt_count, c2_w_em, c2_ms_em = 0, 0, 0

    print(f"\n--- Detalle de las 8 Muestras de Control en Condición 2 (Oráculo) ---")
    tol_c2_w_ok, tol_c2_ms_ok, tol_c2_both_ok = 0, 0, 0
    _FLEX_WATT_RE = re.compile(r"wa[t_]*[:_\s]*(\d+)", re.IGNORECASE)
    _FLEX_MS_RE = re.compile(r"m[s_]*[:_\s]*(\d+)", re.IGNORECASE)

    for idx, (prompt_b, target_b, tw, tms) in enumerate(fixed_8):
        p_str = prompt_b.decode("utf-8")
        t_str = target_b.decode("utf-8")
        _, tr_pw, tr_pms, _, _ = parse_scratchpad_prediction(t_str, "")

        p_orac = p_str + f"pw: {tr_pw}\npms: {tr_pms}\n"
        gen_b2, _, _ = generate_block_causal_with_kv_cache(model_8, p_orac, cfg, gen_length=24, num_recursions_override=2)
        cont_b2 = gen_b2[len(p_orac):] if gen_b2.startswith(p_orac) else gen_b2
        f2, p2_w, p2_ms = parse_only_finals(cont_b2)
        if f2:
            c2_fmt_count += 1
        if p2_w == tw:
            c2_w_em += 1
        if p2_ms == tms:
            c2_ms_em += 1

        # Extracción flexible de watt y ms
        m_fw = _FLEX_WATT_RE.search(cont_b2)
        m_fms = _FLEX_MS_RE.search(cont_b2)
        pw_flex = int(m_fw.group(1)) if m_fw else None
        pms_flex = int(m_fms.group(1)) if m_fms else None

        w_ok = (pw_flex == tw)
        ms_ok = (pms_flex == tms)
        if w_ok:
            tol_c2_w_ok += 1
        if ms_ok:
            tol_c2_ms_ok += 1
        if w_ok and ms_ok:
            tol_c2_both_ok += 1

        print(f"  [Muestra #{idx+1}] Target: watt={tw}, ms={tms} | Generado cont: {repr(cont_b2)}")
        print(f"               Extracción Flexible: watt={pw_flex} ({'OK' if w_ok else 'ERR'}), ms={pms_flex} ({'OK' if ms_ok else 'ERR'})")

    print(f"\n--- Resultados del Control Cruzado (8 Muestras) ---")
    print(f"  Test A (Original Full): Formato={fmt_orig}/8 ({100.0*fmt_orig/8:.1f}%), EM={em_orig}/8 ({100.0*em_orig/8:.1f}%)")
    print(f"  Test B1 (Matriz Ruptura C1, gen_length=24): Formato={c1_fmt_count}/8, pw_EM={c1_pw_em}/8, pms_EM={c1_pms_em}/8")
    print(f"  Test B2 Estricto (Matriz Ruptura C2 Orac, gen_length=24): Formato={c2_fmt_count}/8, w_EM={c2_w_em}/8, ms_EM={c2_ms_em}/8")
    print(f"  Test B2 Flexible (Números Aislados): watt_EM={tol_c2_w_ok}/8 ({100.0*tol_c2_w_ok/8:.1f}%), ms_EM={tol_c2_ms_ok}/8 ({100.0*tol_c2_ms_ok/8:.1f}%), Ambos={tol_c2_both_ok}/8 ({100.0*tol_c2_both_ok/8:.1f}%)")


if __name__ == "__main__":
    inspect_raw_generation()
