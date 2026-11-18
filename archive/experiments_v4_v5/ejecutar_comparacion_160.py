# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

import math
import os
import random
import re
import resource
import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import HybridConfig
from model import TernaryRecursiveDiffusionLM
from data import ProceduralCompositionalDataset
from train import compute_v4_hybrid_loss, update_bitnet_schedule
from trdm.diffusion.corruption import apply_blockwise_diffusion_corruption
from evaluar_muestreador_corregido import (
    generate_block_causal_corrected,
    parse_subtotals_dual,
    parse_finals_dual,
)
from trdm.data.parsers import parse_scratchpad_prediction


def get_peak_ram_mib() -> float:
    """Retorna el pico máximo de RSS del proceso registrado por el kernel Linux en MiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def compute_isolated_digit_acc(pred_str: str, target_str: str) -> Tuple[int, int]:
    """Calcula aciertos y total de dígitos exclusivamente en posiciones donde target_str tiene un dígito."""
    correct, total = 0, 0
    for idx, ch in enumerate(target_str):
        if ch.isdigit():
            total += 1
            if idx < len(pred_str) and pred_str[idx] == ch:
                correct += 1
    return correct, total


def train_variant(cfg: HybridConfig, variant_name: str) -> Dict[str, float]:
    print("\n" + "=" * 78)
    print(f"  INICIANDO ENTRENAMIENTO: {variant_name.upper()}")
    print(f"  max_seq_len={cfg.max_seq_len} | use_explicit_operands={cfg.use_explicit_operands} | max_steps={cfg.max_steps}")
    print("=" * 78)

    # Inicialización limpia desde semilla definida
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)

    train_ds = ProceduralCompositionalDataset(cfg, split="train", num_samples=3200)
    val_ds = ProceduralCompositionalDataset(cfg, split="val", num_samples=cfg.val_samples_full)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = TernaryRecursiveDiffusionLM(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.configure_adamw_groups(), lr=cfg.learning_rate, betas=cfg.betas)

    step = 0
    train_time_accum = 0.0
    step_times: List[float] = []

    model.train()
    t_start_total = time.perf_counter()

    while step < cfg.max_steps:
        for x_0, can_mask, digit_mask in train_loader:
            t_step0 = time.perf_counter()
            step += 1
            lr_now = update_bitnet_schedule(optimizer, step, cfg)
            r_train = cfg.num_recursions

            # Paso encadenado 1 (ruido alto t1)
            x_t1, mask_pos1, t1 = apply_blockwise_diffusion_corruption(x_0, can_mask, cfg)
            sup_mask1 = mask_pos1 if cfg.lazy_vocab_projection else None
            logits1, inter1, halt1, _, carried1 = model(
                x_t1, t1, num_recursions=r_train, record_telemetry=False, supervised_mask=sup_mask1
            )
            loss1, _ = compute_v4_hybrid_loss(logits1, inter1, halt1, x_0, mask_pos1, digit_mask, t1, cfg)

            # Paso encadenado 2 con predicción desacoplada (.detach())
            if logits1.ndim == 2:
                preds_sup1 = logits1.detach().argmax(dim=-1)
                preds1 = x_t1.clone()
                preds1[mask_pos1] = preds_sup1
            else:
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

            sup_mask2 = keep_mask if cfg.lazy_vocab_projection else None
            logits2, inter2, halt2, _, _ = model(
                x_t2, t2, delta_t=dt, prev_states=carried1, num_recursions=r_train,
                record_telemetry=False, supervised_mask=sup_mask2
            )
            loss2, metrics2 = compute_v4_hybrid_loss(logits2, inter2, halt2, x_0, keep_mask, digit_mask, t2, cfg)

            total_loss = 0.45 * loss1 + 0.55 * loss2
            assert torch.isfinite(total_loss), f"Pérdida no finita en paso {step}: {total_loss.item()}"

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if torch.isfinite(grad_norm):
                optimizer.step()

            elapsed_step = time.perf_counter() - t_step0
            train_time_accum += elapsed_step
            if step > 1:
                step_times.append(elapsed_step * 1000.0)

            if step % 100 == 0 or step == 1:
                ram_now = get_peak_ram_mib()
                ms_step = (train_time_accum / step) * 1000.0
                print(
                    f"  [{variant_name}] Paso {step:03d}/{cfg.max_steps} | LR: {lr_now:.2e} | "
                    f"Loss: {total_loss.item():.3f} | GradNorm: {grad_norm.item():.2f} | "
                    f"TrainDig (Reconst): {metrics2['train_digit_acc']:.2f} | "
                    f"Tiempo: {ms_step:.1f} ms/p | RAM Pico: {ram_now:.1f} MiB",
                    flush=True
                )

            if step >= cfg.max_steps:
                break

    total_wall_s = time.perf_counter() - t_start_total

    # Guardar estado completo con configuración efectiva
    checkpoint_dict = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "config": cfg,
        "variant_name": variant_name,
    }
    torch.save(checkpoint_dict, cfg.checkpoint_path)
    packed_bytes = model.export_bitcos_archive(cfg.bitcos_export_path)

    steady_step_ms = sum(step_times) / max(1, len(step_times))
    print(
        f"  [Guardado {variant_name}] Checkpoint: {cfg.checkpoint_path} | BITCOSv2: {packed_bytes/1024:.1f} KiB | "
        f"Tiempo Total: {total_wall_s:.1f} s (~{total_wall_s/60.0:.1f} min) | Régimen: {steady_step_ms:.1f} ms/paso"
    )

    stats = {
        "total_wall_s": total_wall_s,
        "steady_step_ms": steady_step_ms,
        "peak_ram_mib": get_peak_ram_mib(),
        "final_loss": float(total_loss.item()),
    }

    del model
    del optimizer
    del checkpoint_dict
    return stats


def evaluate_variant_principal(
    cfg: HybridConfig,
    variant_name: str,
    train_records_96: List[Tuple[bytes, bytes, int, int]],
    heldout_records_96: List[Tuple[bytes, bytes, int, int]],
) -> Dict[str, Dict[str, float]]:
    print(f"\n" + "=" * 78)
    print(f"  EVALUACIÓN PRINCIPAL (R=2): {variant_name.upper()}")
    print(f"  Checkpoint: {cfg.checkpoint_path} | Dataset: {'Opción A (Operandos)' if cfg.use_explicit_operands else 'Referencia (Categorías)'}")
    print("=" * 78)

    model = TernaryRecursiveDiffusionLM(cfg).to(cfg.device)
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.load_bitcos_archive(cfg.bitcos_export_path)
    model.prepare_for_eval()

    splits = [
        ("Train (Muestra fija de 96 casos)", train_records_96),
        ("Held-Out OOD (96 casos reservados)", heldout_records_96),
    ]

    results_by_split = {}

    for split_label, records in splits:
        n = len(records)
        print(f"\n--- Evaluando {split_label} ({n} casos) ---")

        # 1. Medir Reconstrucción Estática (Masked Argmax)
        ds = ProceduralCompositionalDataset(cfg, split="train" if "Train" in split_label else "val", num_samples=96)
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        rec_dig_ok, rec_dig_tot = 0, 0
        with torch.no_grad():
            for x_0, can_mask, digit_mask in loader:
                x_t, m_pos, t_vec = apply_blockwise_diffusion_corruption(x_0, can_mask, cfg, fixed_t=0.50)
                out = model(x_t, t_vec, num_recursions=2, supervised_mask=m_pos if cfg.lazy_vocab_projection else None)
                if out.final_logits.ndim == 2:
                    p_sup = out.final_logits.argmax(dim=-1)
                    c_sup = (p_sup == x_0[m_pos])
                    d_sup = digit_mask[m_pos]
                    rec_dig_ok += int((c_sup & d_sup).sum().item())
                    rec_dig_tot += int(d_sup.sum().item())
                else:
                    p = out.final_logits.argmax(dim=-1)
                    m_d = m_pos & digit_mask
                    rec_dig_ok += int(((p == x_0) & m_d).sum().item())
                    rec_dig_tot += int(m_d.sum().item())
        reconst_dig_acc = 100.0 * rec_dig_ok / max(1, rec_dig_tot)

        # 2. Generación Libre con Muestreador Corregido
        # C1: Subtotales
        c1_strict_valid, c1_strict_em = 0, 0
        c1_diag_exclusive, c1_diag_em = 0, 0
        c1_pw_em, c1_pms_em = 0, 0
        c1_pw_dig_c, c1_pw_dig_t = 0, 0
        c1_pms_dig_c, c1_pms_dig_t = 0, 0

        # C2: Oráculo -> Respuesta Final
        c2_strict_valid, c2_strict_em = 0, 0
        c2_diag_exclusive, c2_diag_em = 0, 0
        c2_w_em, c2_ms_em = 0, 0
        c2_w_dig_c, c2_w_dig_t = 0, 0
        c2_ms_dig_c, c2_ms_dig_t = 0, 0

        # C4: Cadena Completa Integrada
        c4_full_strict_em = 0
        c4_full_diag_em = 0

        for prompt_b, target_b, tw, tms in records:
            p_str = prompt_b.decode("utf-8")
            t_str = target_b.decode("utf-8")
            _, tr_pw, tr_pms, _, _ = parse_scratchpad_prediction(t_str, "")

            # C1: Factores -> Generar Subtotales
            gen1, _, _ = generate_block_causal_corrected(
                model, p_str, cfg, gen_length=24, num_recursions_override=2, restrict_vocab=True
            )
            cont1 = gen1[len(p_str):] if gen1.startswith(p_str) else gen1
            st1, p1_pw, p1_pms = parse_subtotals_dual(cont1)

            pw_ok = (p1_pw == tr_pw)
            pms_ok = (p1_pms == tr_pms)
            sub_both = (pw_ok and pms_ok)

            if pw_ok: c1_pw_em += 1
            if pms_ok: c1_pms_em += 1

            if st1 == "STRICT_VALID":
                c1_strict_valid += 1
                if sub_both: c1_strict_em += 1
            elif st1 == "UNAMBIGUOUS_RECOVERABLE":
                c1_diag_exclusive += 1
                if sub_both: c1_diag_em += 1

            # DigAcc aislada de subtotales
            m_pw = re.search(r"pw[:_\s]*(\d+)", cont1, re.I)
            if m_pw:
                c, t = compute_isolated_digit_acc(m_pw.group(1), str(tr_pw))
                c1_pw_dig_c += c; c1_pw_dig_t += t
            else:
                c1_pw_dig_t += len(str(tr_pw))
            m_pms = re.search(r"pms[:_\s]*(\d+)", cont1, re.I)
            if m_pms:
                c, t = compute_isolated_digit_acc(m_pms.group(1), str(tr_pms))
                c1_pms_dig_c += c; c1_pms_dig_t += t
            else:
                c1_pms_dig_t += len(str(tr_pms))

            # C2: Factores + Subtotales Oráculo -> Generar Final
            prompt_c2 = p_str + f"pw: {tr_pw}\npms: {tr_pms}\n"
            gen2, _, _ = generate_block_causal_corrected(
                model, prompt_c2, cfg, gen_length=24, num_recursions_override=2, restrict_vocab=True
            )
            cont2 = gen2[len(prompt_c2):] if gen2.startswith(prompt_c2) else gen2
            st2, p2_w, p2_ms = parse_finals_dual(cont2)

            w_ok = (p2_w == tw)
            ms_ok = (p2_ms == tms)
            resp_both = (w_ok and ms_ok)

            if w_ok: c2_w_em += 1
            if ms_ok: c2_ms_em += 1

            if st2 == "STRICT_VALID":
                c2_strict_valid += 1
                if resp_both: c2_strict_em += 1
            elif st2 == "UNAMBIGUOUS_RECOVERABLE":
                c2_diag_exclusive += 1
                if resp_both: c2_diag_em += 1

            # DigAcc aislada de finales
            m_w = re.search(r"watt[:_\s]*(\d+)", cont2, re.I)
            if m_w:
                c, t = compute_isolated_digit_acc(m_w.group(1), str(tw))
                c2_w_dig_c += c; c2_w_dig_t += t
            else:
                c2_w_dig_t += len(str(tw))
            m_ms = re.search(r"ms[:_\s]*(\d+)", cont2, re.I)
            if m_ms:
                c, t = compute_isolated_digit_acc(m_ms.group(1), str(tms))
                c2_ms_dig_c += c; c2_ms_dig_t += t
            else:
                c2_ms_dig_t += len(str(tms))

            # C4: Cadena Completa Integrada (gen_length=48)
            gen4, _, _ = generate_block_causal_corrected(
                model, p_str, cfg, gen_length=48, num_recursions_override=2, restrict_vocab=True
            )
            is_fmt4, p4_sw, p4_sms, p4_w, p4_ms = parse_scratchpad_prediction(gen4, p_str)
            all_correct = (p4_sw == tr_pw and p4_sms == tr_pms and p4_w == tw and p4_ms == tms)
            if is_fmt4 and all_correct:
                c4_full_strict_em += 1
            elif all_correct:
                c4_full_diag_em += 1

        # Resumen cuantitativo para el split
        c1_cobertura = c1_strict_valid + c1_diag_exclusive
        c1_global_em = c1_strict_em + c1_diag_em
        c2_cobertura = c2_strict_valid + c2_diag_exclusive
        c2_global_em = c2_strict_em + c2_diag_em
        c4_global_em = c4_full_strict_em + c4_full_diag_em

        pw_dig_acc = 100.0 * c1_pw_dig_c / max(1, c1_pw_dig_t)
        pms_dig_acc = 100.0 * c1_pms_dig_c / max(1, c1_pms_dig_t)
        w_dig_acc = 100.0 * c2_w_dig_c / max(1, c2_w_dig_t)
        ms_dig_acc = 100.0 * c2_ms_dig_c / max(1, c2_ms_dig_t)

        print(f"  Reconstrucción Estática DigAcc: {reconst_dig_acc:.1f}%")
        print(f"  C1 (Subtotales):")
        print(f"    - Cobertura Total: {c1_cobertura}/{n} ({100.0*c1_cobertura/n:.1f}%) [Estricto: {c1_strict_valid}, Recup. Diagnóstico Exclusivo: {c1_diag_exclusive}]")
        print(f"    - Subtotal EM Global (Pw y Pms): {c1_global_em}/{n} ({100.0*c1_global_em/n:.1f}%) [Estricto: {c1_strict_em}, Diagnóstico: {c1_diag_em}]")
        print(f"    - Desglose EM: Pw EM = {c1_pw_em}/{n} ({100.0*c1_pw_em/n:.1f}%), Pms EM = {c1_pms_em}/{n} ({100.0*c1_pms_em/n:.1f}%)")
        print(f"    - DigAcc Libre: Pw DigAcc = {pw_dig_acc:.1f}%, Pms DigAcc = {pms_dig_acc:.1f}%")
        print(f"  C2 (Oráculo -> Respuesta Final):")
        print(f"    - Cobertura Total: {c2_cobertura}/{n} ({100.0*c2_cobertura/n:.1f}%) [Estricto: {c2_strict_valid}, Recup. Diagnóstico Exclusivo: {c2_diag_exclusive}]")
        print(f"    - Respuesta Final EM Global (Watt y MS): {c2_global_em}/{n} ({100.0*c2_global_em/n:.1f}%) [Estricto: {c2_strict_em}, Diagnóstico: {c2_diag_em}]")
        print(f"    - Desglose EM: Watt EM = {c2_w_em}/{n} ({100.0*c2_w_em/n:.1f}%), MS EM = {c2_ms_em}/{n} ({100.0*c2_ms_em/n:.1f}%)")
        print(f"    - DigAcc Libre: Watt DigAcc = {w_dig_acc:.1f}%, MS DigAcc = {ms_dig_acc:.1f}%")
        print(f"  C4 (Cadena Completa Integrada):")
        print(f"    - Full Chain EM Global: {c4_global_em}/{n} ({100.0*c4_global_em/n:.1f}%) [Estricto: {c4_full_strict_em}, Diagnóstico: {c4_full_diag_em}]")

        split_key = "train" if "Train" in split_label else "heldout"
        results_by_split[split_key] = {
            "reconst_dig_acc": reconst_dig_acc,
            "c1_cobertura": c1_cobertura,
            "c1_global_em": c1_global_em,
            "c1_pw_em": c1_pw_em,
            "c1_pms_em": c1_pms_em,
            "c1_pw_dig_acc": pw_dig_acc,
            "c1_pms_dig_acc": pms_dig_acc,
            "c2_cobertura": c2_cobertura,
            "c2_global_em": c2_global_em,
            "c2_w_em": c2_w_em,
            "c2_ms_em": c2_ms_em,
            "c2_w_dig_acc": w_dig_acc,
            "c2_ms_dig_acc": ms_dig_acc,
            "c4_global_em": c4_global_em,
        }

    del model
    return results_by_split


def run_full_comparison_pipeline():
    print("=" * 80)
    print("  EXPERIMENTO COMPARATIVO CONTROLADO A LONGITUD 160 (800 PASOS)")
    print("  Variante 1: Referencia Categórica (L=160, Categorías)")
    print("  Variante 2: Opción A (L=160, Categorías + Operandos Explícitos)")
    print("=" * 80)

    # 1. Fijar exactamente los mismos 96 ejemplos de Train para ambas variantes
    cfg_setup = HybridConfig(max_seq_len=160)
    train_ds_ref = ProceduralCompositionalDataset(cfg_setup, split="train", num_samples=384)
    val_ds_ref = ProceduralCompositionalDataset(cfg_setup, split="val", num_samples=96)

    rng = random.Random(42)
    # Lista de tuplas base (prompt_b, target_b, watt, ms)
    train_indices = rng.sample(range(len(train_ds_ref.unique_records)), 96)

    # 2. Configuración para Variante 1: Referencia (160)
    cfg_ref160 = HybridConfig(
        max_seq_len=160,
        use_explicit_operands=False,
        max_steps=800,
        checkpoint_path="trdm_v4_ref160_checkpoint.pt",
        bitcos_export_path="trdm_v4_ref160_bitcos.bin",
    )

    # 3. Configuración para Variante 2: Opción A (160)
    cfg_optA160 = HybridConfig(
        max_seq_len=160,
        use_explicit_operands=True,
        max_steps=800,
        checkpoint_path="trdm_v4_optA_checkpoint.pt",
        bitcos_export_path="trdm_v4_optA_bitcos.bin",
    )

    # Preparar registros para Referencia
    ds_ref_full = ProceduralCompositionalDataset(cfg_ref160, split="train", num_samples=384)
    ref_train_96 = [ds_ref_full.unique_records[i] for i in train_indices]
    ref_val_96 = list(ProceduralCompositionalDataset(cfg_ref160, split="val", num_samples=96).unique_records)

    # Preparar registros para Opción A
    ds_optA_full = ProceduralCompositionalDataset(cfg_optA160, split="train", num_samples=384)
    optA_train_96 = [ds_optA_full.unique_records[i] for i in train_indices]
    optA_val_96 = list(ProceduralCompositionalDataset(cfg_optA160, split="val", num_samples=96).unique_records)

    # =====================================================================
    # PASO 1: ENTRENAR VARIANTE REFERENCIA (160)
    # =====================================================================
    stats_ref = train_variant(cfg_ref160, "Referencia_160")

    # =====================================================================
    # PASO 2: ENTRENAR VARIANTE OPCIÓN A (160)
    # =====================================================================
    stats_optA = train_variant(cfg_optA160, "OpcionA_160")

    # =====================================================================
    # PASO 3: EVALUACIÓN PRINCIPAL COMPARATIVA (R=2)
    # =====================================================================
    eval_ref = evaluate_variant_principal(cfg_ref160, "Referencia_160", ref_train_96, ref_val_96)
    eval_optA = evaluate_variant_principal(cfg_optA160, "OpcionA_160", optA_train_96, optA_val_96)

    # =====================================================================
    # PASO 4: REPORTE FINAL COMPARATIVO
    # =====================================================================
    print("\n" + "=" * 80)
    print("  CUADRO FINAL COMPARATIVO: REFERENCIA (160) VS OPCIÓN A (160)")
    print("=" * 80)

    print(f"\n[A] Recursos y Tiempos de Entrenamiento:")
    print(f"  - Referencia_160: Tiempo Total = {stats_ref['total_wall_s']:.1f} s (~{stats_ref['total_wall_s']/60.0:.1f} min) | Régimen = {stats_ref['steady_step_ms']:.1f} ms/p | RAM Pico = {stats_ref['peak_ram_mib']:.1f} MiB | Pérdida Final = {stats_ref['final_loss']:.3f}")
    print(f"  - OpcionA_160:    Tiempo Total = {stats_optA['total_wall_s']:.1f} s (~{stats_optA['total_wall_s']/60.0:.1f} min) | Régimen = {stats_optA['steady_step_ms']:.1f} ms/p | RAM Pico = {stats_optA['peak_ram_mib']:.1f} MiB | Pérdida Final = {stats_optA['final_loss']:.3f}")

    print(f"\n[B] Métricas Principales en Train (Muestra fija de 96 casos):")
    print(f"  Métrica                                  | Referencia_160 | Opción A_160   | Delta")
    print(f"  -----------------------------------------|----------------|----------------|-------")
    r_tr = eval_ref["train"]; o_tr = eval_optA["train"]
    print(f"  Reconstrucción Estática DigAcc           | {r_tr['reconst_dig_acc']:>12.1f}% | {o_tr['reconst_dig_acc']:>12.1f}% | {o_tr['reconst_dig_acc']-r_tr['reconst_dig_acc']:>+5.1f}%")
    print(f"  C1 Cobertura Diagnóstica Total           | {r_tr['c1_cobertura']:>11d}/96 | {o_tr['c1_cobertura']:>11d}/96 | {o_tr['c1_cobertura']-r_tr['c1_cobertura']:>+5d}")
    print(f"  C1 Subtotales EM Global (Pw y Pms)       | {r_tr['c1_global_em']:>11d}/96 | {o_tr['c1_global_em']:>11d}/96 | {o_tr['c1_global_em']-r_tr['c1_global_em']:>+5d}")
    print(f"  C1 Pw EM Numérico                        | {r_tr['c1_pw_em']:>11d}/96 | {o_tr['c1_pw_em']:>11d}/96 | {o_tr['c1_pw_em']-r_tr['c1_pw_em']:>+5d}")
    print(f"  C1 Pms EM Numérico                       | {r_tr['c1_pms_em']:>11d}/96 | {o_tr['c1_pms_em']:>11d}/96 | {o_tr['c1_pms_em']-r_tr['c1_pms_em']:>+5d}")
    print(f"  C1 Pw DigAcc Libre                       | {r_tr['c1_pw_dig_acc']:>12.1f}% | {o_tr['c1_pw_dig_acc']:>12.1f}% | {o_tr['c1_pw_dig_acc']-r_tr['c1_pw_dig_acc']:>+5.1f}%")
    print(f"  C1 Pms DigAcc Libre                      | {r_tr['c1_pms_dig_acc']:>12.1f}% | {o_tr['c1_pms_dig_acc']:>12.1f}% | {o_tr['c1_pms_dig_acc']-r_tr['c1_pms_dig_acc']:>+5.1f}%")
    print(f"  C2 Cobertura Diagnóstica Total           | {r_tr['c2_cobertura']:>11d}/96 | {o_tr['c2_cobertura']:>11d}/96 | {o_tr['c2_cobertura']-r_tr['c2_cobertura']:>+5d}")
    print(f"  C2 Respuesta Oráculo EM Global           | {r_tr['c2_global_em']:>11d}/96 | {o_tr['c2_global_em']:>11d}/96 | {o_tr['c2_global_em']-r_tr['c2_global_em']:>+5d}")
    print(f"  C2 Watt EM Numérico                      | {r_tr['c2_w_em']:>11d}/96 | {o_tr['c2_w_em']:>11d}/96 | {o_tr['c2_w_em']-r_tr['c2_w_em']:>+5d}")
    print(f"  C2 MS EM Numérico                        | {r_tr['c2_ms_em']:>11d}/96 | {o_tr['c2_ms_em']:>11d}/96 | {o_tr['c2_ms_em']-r_tr['c2_ms_em']:>+5d}")
    print(f"  C2 Watt DigAcc Libre                     | {r_tr['c2_w_dig_acc']:>12.1f}% | {o_tr['c2_w_dig_acc']:>12.1f}% | {o_tr['c2_w_dig_acc']-r_tr['c2_w_dig_acc']:>+5.1f}%")
    print(f"  C2 MS DigAcc Libre                       | {r_tr['c2_ms_dig_acc']:>12.1f}% | {o_tr['c2_ms_dig_acc']:>12.1f}% | {o_tr['c2_ms_dig_acc']-r_tr['c2_ms_dig_acc']:>+5.1f}%")
    print(f"  C4 Cadena Completa Integrada EM Global   | {r_tr['c4_global_em']:>11d}/96 | {o_tr['c4_global_em']:>11d}/96 | {o_tr['c4_global_em']-r_tr['c4_global_em']:>+5d}")

    print(f"\n[C] Métricas Principales en Held-Out OOD (96 casos reservados):")
    print(f"  Métrica                                  | Referencia_160 | Opción A_160   | Delta")
    print(f"  -----------------------------------------|----------------|----------------|-------")
    r_ho = eval_ref["heldout"]; o_ho = eval_optA["heldout"]
    print(f"  Reconstrucción Estática DigAcc           | {r_ho['reconst_dig_acc']:>12.1f}% | {o_ho['reconst_dig_acc']:>12.1f}% | {o_ho['reconst_dig_acc']-r_ho['reconst_dig_acc']:>+5.1f}%")
    print(f"  C1 Cobertura Diagnóstica Total           | {r_ho['c1_cobertura']:>11d}/96 | {o_ho['c1_cobertura']:>11d}/96 | {o_ho['c1_cobertura']-r_ho['c1_cobertura']:>+5d}")
    print(f"  C1 Subtotales EM Global (Pw y Pms)       | {r_ho['c1_global_em']:>11d}/96 | {o_ho['c1_global_em']:>11d}/96 | {o_ho['c1_global_em']-r_ho['c1_global_em']:>+5d}")
    print(f"  C1 Pw EM Numérico                        | {r_ho['c1_pw_em']:>11d}/96 | {o_ho['c1_pw_em']:>11d}/96 | {o_ho['c1_pw_em']-r_ho['c1_pw_em']:>+5d}")
    print(f"  C1 Pms EM Numérico                       | {r_ho['c1_pms_em']:>11d}/96 | {o_ho['c1_pms_em']:>11d}/96 | {o_ho['c1_pms_em']-r_ho['c1_pms_em']:>+5d}")
    print(f"  C1 Pw DigAcc Libre                       | {r_ho['c1_pw_dig_acc']:>12.1f}% | {o_ho['c1_pw_dig_acc']:>12.1f}% | {o_ho['c1_pw_dig_acc']-r_ho['c1_pw_dig_acc']:>+5.1f}%")
    print(f"  C1 Pms DigAcc Libre                      | {r_ho['c1_pms_dig_acc']:>12.1f}% | {o_ho['c1_pms_dig_acc']:>12.1f}% | {o_ho['c1_pms_dig_acc']-r_ho['c1_pms_dig_acc']:>+5.1f}%")
    print(f"  C2 Cobertura Diagnóstica Total           | {r_ho['c2_cobertura']:>11d}/96 | {o_ho['c2_cobertura']:>11d}/96 | {o_ho['c2_cobertura']-r_ho['c2_cobertura']:>+5d}")
    print(f"  C2 Respuesta Oráculo EM Global           | {r_ho['c2_global_em']:>11d}/96 | {o_ho['c2_global_em']:>11d}/96 | {o_ho['c2_global_em']-r_ho['c2_global_em']:>+5d}")
    print(f"  C2 Watt EM Numérico                      | {r_ho['c2_w_em']:>11d}/96 | {o_ho['c2_w_em']:>11d}/96 | {o_ho['c2_w_em']-r_ho['c2_w_em']:>+5d}")
    print(f"  C2 MS EM Numérico                        | {r_ho['c2_ms_em']:>11d}/96 | {o_ho['c2_ms_em']:>11d}/96 | {o_ho['c2_ms_em']-r_ho['c2_ms_em']:>+5d}")
    print(f"  C2 Watt DigAcc Libre                     | {r_ho['c2_w_dig_acc']:>12.1f}% | {o_ho['c2_w_dig_acc']:>12.1f}% | {o_ho['c2_w_dig_acc']-r_ho['c2_w_dig_acc']:>+5.1f}%")
    print(f"  C2 MS DigAcc Libre                       | {r_ho['c2_ms_dig_acc']:>12.1f}% | {o_ho['c2_ms_dig_acc']:>12.1f}% | {o_ho['c2_ms_dig_acc']-r_ho['c2_ms_dig_acc']:>+5.1f}%")
    print(f"  C4 Cadena Completa Integrada EM Global   | {r_ho['c4_global_em']:>11d}/96 | {o_ho['c4_global_em']:>11d}/96 | {o_ho['c4_global_em']-r_ho['c4_global_em']:>+5d}")


if __name__ == "__main__":
    run_full_comparison_pipeline()
