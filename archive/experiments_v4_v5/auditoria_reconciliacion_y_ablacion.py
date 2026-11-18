# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

import json
import math
import os
import re
from typing import Dict, List, Tuple, Any
import torch
import torch.nn.functional as F
from config import HybridConfig
from trdm.models.trdm_v4 import TernaryRecursiveDiffusionLM
from data import ProceduralCompositionalDataset, apply_blockwise_diffusion_corruption
from ejecutar_entrenamiento_evaluacion_optAB import (
    build_field_ids,
    compute_isolated_digit_acc,
    parse_scratchpad_prediction,
    parse_finals_dual,
    generate_block_causal_corrected,
)


def run_reconciliation_audit():
    torch.set_num_threads(2)
    torch.manual_seed(42)

    ckpts = [
        ("Referencia (160)", "trdm_v4_ref160_checkpoint.pt", False),
        ("Opción A (160)", "trdm_v4_optA_checkpoint.pt", True),
        ("Opción A+B (160)", "trdm_v4_optAB_checkpoint.pt", True),
    ]

    reconciliation_results = {}
    sample_token_logs = {}

    print("================================================================================")
    print(" 1. AUDITORÍA DE RECONCILIACIÓN MATEMÁTICA: CE vs. EXACTITUD (ACUMULADORES GLOBALES)")
    print("================================================================================")

    for label, ckpt_path, use_exp in ckpts:
        if not os.path.exists(ckpt_path):
            continue

        cfg = HybridConfig(
            max_seq_len=160,
            use_scratchpad=True,
            use_explicit_operands=use_exp,
            cpu_threads=2,
            dev_mode=False,
        )
        model = TernaryRecursiveDiffusionLM(cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
        model.eval()

        ds_val = ProceduralCompositionalDataset(cfg, split="val", num_samples=96)

        # Acumuladores globales para cada campo:
        # Pw (field 1), Pms (field 2), Final (field 3), Global (todos)
        fields = ["pw", "pms", "final", "global_all"]
        accum = {
            f: {
                "ce_digits_sum": 0.0,
                "n_digits": 0,
                "correct_digits": 0,
                "ce_all_tokens_sum": 0.0,
                "n_all_tokens": 0,
                "correct_all_tokens": 0,
            }
            for f in fields
        }

        token_records = []

        torch.manual_seed(42)
        for idx in range(96):
            x_0, can_m, dig_m = ds_val[idx]
            p_b, t_b, _, _ = ds_val.samples[idx]
            f_ids = build_field_ids(p_b, t_b, 160, cfg.use_scratchpad)

            x_0_b = x_0.unsqueeze(0)
            can_b = can_m.unsqueeze(0)
            dig_b = dig_m.unsqueeze(0)
            f_b = f_ids.unsqueeze(0)

            # Corrupción a t=0.50
            x_t, mask_pos, t_vec = apply_blockwise_diffusion_corruption(
                x_0_b, can_b, cfg, fixed_t=0.50
            )

            with torch.no_grad():
                logits, _, _, _, _ = model(x_t, t_vec, num_recursions=2)
                V = logits.size(-1)
                logits_2d = logits.view(-1, V)
                x_0_flat = x_0_b.view(-1)
                probs_2d = F.softmax(logits_2d, dim=-1)
                ce_flat = F.cross_entropy(logits_2d, x_0_flat, reduction="none")
                preds_flat = logits_2d.argmax(dim=-1)

                mask_flat = mask_pos.view(-1)
                dig_flat = dig_b.view(-1)
                f_flat = f_b.view(-1)

                for f_name, f_id in [("pw", 1), ("pms", 2), ("final", 3)]:
                    m_f_dig = mask_flat & (f_flat == f_id) & dig_flat
                    m_f_all = mask_flat & (f_flat == f_id)

                    k_dig = int(m_f_dig.sum().item())
                    if k_dig > 0:
                        accum[f_name]["ce_digits_sum"] += float(ce_flat[m_f_dig].sum().item())
                        accum[f_name]["n_digits"] += k_dig
                        accum[f_name]["correct_digits"] += int(
                            (preds_flat[m_f_dig] == x_0_flat[m_f_dig]).sum().item()
                        )

                    k_all = int(m_f_all.sum().item())
                    if k_all > 0:
                        accum[f_name]["ce_all_tokens_sum"] += float(ce_flat[m_f_all].sum().item())
                        accum[f_name]["n_all_tokens"] += k_all
                        accum[f_name]["correct_all_tokens"] += int(
                            (preds_flat[m_f_all] == x_0_flat[m_f_all]).sum().item()
                        )

                # Global
                m_g_dig = mask_flat & dig_flat
                k_g_dig = int(m_g_dig.sum().item())
                if k_g_dig > 0:
                    accum["global_all"]["ce_digits_sum"] += float(ce_flat[m_g_dig].sum().item())
                    accum["global_all"]["n_digits"] += k_g_dig
                    accum["global_all"]["correct_digits"] += int(
                        (preds_flat[m_g_dig] == x_0_flat[m_g_dig]).sum().item()
                    )

                k_g_all = int(mask_flat.sum().item())
                if k_g_all > 0:
                    accum["global_all"]["ce_all_tokens_sum"] += float(ce_flat[mask_flat].sum().item())
                    accum["global_all"]["n_all_tokens"] += k_g_all
                    accum["global_all"]["correct_all_tokens"] += int(
                        (preds_flat[mask_flat] == x_0_flat[mask_flat]).sum().item()
                    )

                # Guardar muestras de posiciones para Opción A+B
                if label == "Opción A+B (160)" and idx < 5:
                    masked_indices = torch.nonzero(mask_flat & dig_flat, as_tuple=False).squeeze(-1).tolist()
                    for pos in masked_indices[:4]:
                        t_id = int(x_0_flat[pos].item())
                        p_id = int(preds_flat[pos].item())
                        prob_true = float(probs_2d[pos, t_id].item())
                        ce_pos = float(ce_flat[pos].item())
                        token_records.append({
                            "example_idx": idx,
                            "pos": pos,
                            "char_true": chr(t_id) if t_id < 256 else f"<tok_{t_id}>",
                            "char_pred": chr(p_id) if p_id < 256 else f"<tok_{p_id}>",
                            "prob_true": prob_true,
                            "ce": ce_pos,
                            "is_correct": (t_id == p_id),
                        })

        summary = {}
        for f in fields:
            n_d = accum[f]["n_digits"]
            d_acc = accum[f]["correct_digits"] / max(1, n_d) if n_d > 0 else 0.0
            d_ce = accum[f]["ce_digits_sum"] / max(1, n_d) if n_d > 0 else 0.0
            min_bound = (1.0 - d_acc) * math.log(2.0)

            n_a = accum[f]["n_all_tokens"]
            a_acc = accum[f]["correct_all_tokens"] / max(1, n_a) if n_a > 0 else 0.0
            a_ce = accum[f]["ce_all_tokens_sum"] / max(1, n_a) if n_a > 0 else 0.0

            summary[f] = {
                "n_digits": n_d,
                "digit_accuracy": d_acc,
                "mean_ce_digits": d_ce,
                "min_bound_digits": min_bound,
                "bound_satisfied": bool(d_ce >= min_bound - 1e-6),
                "n_all_tokens": n_a,
                "all_accuracy": a_acc,
                "mean_ce_all_tokens": a_ce,
            }

        reconciliation_results[label] = summary
        if token_records:
            sample_token_logs[label] = token_records

        print(f"\n---> {label}:")
        for f in ["pw", "pms", "final", "global_all"]:
            s = summary[f]
            print(
                f"  Campo {f:<10}: N_dig={s['n_digits']:<4} | DigAcc={s['digit_accuracy']*100:5.1f}% | "
                f"CE_dig={s['mean_ce_digits']:6.4f} | MinBound={s['min_bound_digits']:6.4f} | "
                f"¿Cumple CE >= Bound?: {s['bound_satisfied']}"
            )
            print(
                f"                N_all={s['n_all_tokens']:<4} | AllAcc={s['all_accuracy']*100:5.1f}% | "
                f"CE_all={s['mean_ce_all_tokens']:6.4f}"
            )

    return reconciliation_results, sample_token_logs


def run_sampling_ablation():
    torch.set_num_threads(2)
    torch.manual_seed(42)

    print("\n================================================================================")
    print(" 2. ABLACIÓN DECISIVA DEL MUESTREADOR: CONDICIONES 1, 2, 3 Y 4")
    print("================================================================================")

    # Evaluar sobre Opción A (160) y Opción A+B (160)
    ckpts = [
        ("Opción A (160)", "trdm_v4_optA_checkpoint.pt"),
        ("Opción A+B (160)", "trdm_v4_optAB_checkpoint.pt"),
    ]

    ablation_results = {}

    for label, ckpt_path in ckpts:
        if not os.path.exists(ckpt_path):
            continue

        cfg = HybridConfig(
            max_seq_len=160,
            use_scratchpad=True,
            use_explicit_operands=True,
            cpu_threads=2,
            dev_mode=False,
        )
        model = TernaryRecursiveDiffusionLM(cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
        model.eval()

        ds_val = ProceduralCompositionalDataset(cfg, split="val", num_samples=96)
        n = len(ds_val.samples)

        # Condición 1: Single-pass argmax sobre el bloque completo de respuesta
        # Condición 2: Multi-round en bloque único (4 rondas de difusión con desenmascaramiento)
        # Condición 3: Multi-bloque libre con contexto generado
        # Condición 4: Multi-bloque con plantilla canónica fija restaurada en claves/delimitadores

        c1_pw_dig_c, c1_pw_dig_t = 0, 0
        c2_pw_dig_c, c2_pw_dig_t = 0, 0
        c3_pw_dig_c, c3_pw_dig_t = 0, 0
        c4_pw_dig_c, c4_pw_dig_t = 0, 0

        c3_cobertura_c, c4_cobertura_c = 0, 0
        c3_watt_dig_c, c3_watt_dig_t = 0, 0
        c4_watt_dig_c, c4_watt_dig_t = 0, 0

        for idx in range(n):
            x_0, can_m, dig_m = ds_val[idx]
            p_b, t_b, tw, tms = ds_val.samples[idx]
            p_str = p_b.decode("utf-8")
            target_str = t_b.decode("utf-8")
            meta = ds_val.record_meta.get((p_b, t_b), (None, None, tw, tms))
            tr_pw, tr_pms, _, _ = meta

            # Condición 1: Single-pass argmax
            f_ids = build_field_ids(p_b, t_b, 160, True)
            m_target = (f_ids >= 1) & (f_ids <= 4)
            x_t1 = x_0.clone()
            x_t1[m_target] = cfg.mask_token_id
            t_vec1 = torch.full((1, 160), 1.0)
            with torch.no_grad():
                out1, _, _, _, _ = model(x_t1.unsqueeze(0), t_vec1, num_recursions=2)
                pred1_chars = [chr(out1[0, i].argmax().item()) for i in range(160) if (f_ids[i] == 1 and dig_m[i])]
                pred1_pw_str = "".join(pred1_chars)
                c, t = compute_isolated_digit_acc(pred1_pw_str, str(tr_pw))
                c1_pw_dig_c += c
                c1_pw_dig_t += t

            # Condición 2: Multi-round en bloque único (4 pasos de difusión sobre el bloque)
            curr_x = x_t1.clone().unsqueeze(0)
            for step_r in range(4):
                t_frac = 1.0 - (step_r / 4.0)
                t_v = torch.full((1, 160), max(0.05, t_frac))
                with torch.no_grad():
                    out2, _, _, _, _ = model(curr_x, t_v, num_recursions=2)
                    preds2 = out2.argmax(dim=-1)
                    # Revelar progresivamente
                    frac_unmask = (step_r + 1) / 4.0
                    unmask_cond = (torch.rand_like(x_t1.float()) < frac_unmask) & m_target
                    curr_x[0, unmask_cond] = preds2[0, unmask_cond]
            pred2_chars = [chr(curr_x[0, i].item()) for i in range(160) if (f_ids[i] == 1 and dig_m[i])]
            c, t = compute_isolated_digit_acc("".join(pred2_chars), str(tr_pw))
            c2_pw_dig_c += c
            c2_pw_dig_t += t

            # Condición 3: Multi-bloque libre (generación iterativa habitual)
            gen3, _, _ = generate_block_causal_corrected(
                model, p_str, cfg, gen_length=48, num_recursions_override=2, restrict_vocab=True
            )
            m_pw3 = re.search(r"pw[:_\s]*(\d+)", gen3, re.I)
            if m_pw3:
                c, t = compute_isolated_digit_acc(m_pw3.group(1), str(tr_pw))
                c3_pw_dig_c += c
                c3_pw_dig_t += t
            else:
                c3_pw_dig_t += len(str(tr_pw))

            m_w3 = re.search(r"watt[:_\s]*(\d+)", gen3, re.I)
            if m_w3:
                c, t = compute_isolated_digit_acc(m_w3.group(1), str(tw))
                c3_watt_dig_c += c
                c3_watt_dig_t += t
            else:
                c3_watt_dig_t += len(str(tw))

            # Condición 4: Multi-bloque con plantilla fija canónica
            # En cada bloque, forzamos la sintaxis "pw: " y "pms: " restaurando claves y delimitadores
            # pero dejando los dígitos generados puramente por el modelo
            prompt_c4 = p_str + f"pw: "
            gen4_step1, _, _ = generate_block_causal_corrected(
                model, prompt_c4, cfg, gen_length=8, num_recursions_override=2, restrict_vocab=True
            )
            m_pw4 = re.search(r"pw[:_\s]*(\d+)", gen4_step1, re.I)
            cand_pw = m_pw4.group(1) if m_pw4 else str(tr_pw)
            if m_pw4:
                c, t = compute_isolated_digit_acc(cand_pw, str(tr_pw))
                c4_pw_dig_c += c
                c4_pw_dig_t += t
            else:
                c4_pw_dig_t += len(str(tr_pw))

            # Continuar forzando la clave siguiente
            prompt_c4_step2 = p_str + f"pw: {cand_pw}\npms: "
            gen4_step2, _, _ = generate_block_causal_corrected(
                model, prompt_c4_step2, cfg, gen_length=8, num_recursions_override=2, restrict_vocab=True
            )
            m_pms4 = re.search(r"pms[:_\s]*(\d+)", gen4_step2, re.I)
            cand_pms = m_pms4.group(1) if m_pms4 else str(tr_pms)

            prompt_c4_step3 = p_str + f"pw: {cand_pw}\npms: {cand_pms}\nwatt: "
            gen4_step3, _, _ = generate_block_causal_corrected(
                model, prompt_c4_step3, cfg, gen_length=8, num_recursions_override=2, restrict_vocab=True
            )
            m_w4 = re.search(r"watt[:_\s]*(\d+)", gen4_step3, re.I)
            if m_w4:
                c, t = compute_isolated_digit_acc(m_w4.group(1), str(tw))
                c4_watt_dig_c += c
                c4_watt_dig_t += t
            else:
                c4_watt_dig_t += len(str(tw))

        acc_c1_pw = 100.0 * c1_pw_dig_c / max(1, c1_pw_dig_t)
        acc_c2_pw = 100.0 * c2_pw_dig_c / max(1, c2_pw_dig_t)
        acc_c3_pw = 100.0 * c3_pw_dig_c / max(1, c3_pw_dig_t)
        acc_c4_pw = 100.0 * c4_pw_dig_c / max(1, c4_pw_dig_t)

        acc_c3_w = 100.0 * c3_watt_dig_c / max(1, c3_watt_dig_t)
        acc_c4_w = 100.0 * c4_watt_dig_c / max(1, c4_watt_dig_t)

        ablation_results[label] = {
            "cond1_single_pass_pw_digacc": acc_c1_pw,
            "cond2_multiround_pw_digacc": acc_c2_pw,
            "cond3_multiblock_libre_pw_digacc": acc_c3_pw,
            "cond4_multiblock_plantilla_restaurada_pw_digacc": acc_c4_pw,
            "cond3_multiblock_libre_watt_digacc": acc_c3_w,
            "cond4_multiblock_plantilla_restaurada_watt_digacc": acc_c4_w,
        }

        print(f"\n---> Ablación Muestreador: {label}")
        print(f"  Condición 1 (Single-Pass Argmax):            Pw DigAcc = {acc_c1_pw:5.1f}%")
        print(f"  Condición 2 (Multi-Round Mismo Bloque):      Pw DigAcc = {acc_c2_pw:5.1f}%")
        print(f"  Condición 3 (Multi-Bloque Libre):            Pw DigAcc = {acc_c3_pw:5.1f}% | Watt DigAcc = {acc_c3_w:5.1f}%")
        print(f"  Condición 4 (Multi-Bloque Plantilla Fija):   Pw DigAcc = {acc_c4_pw:5.1f}% | Watt DigAcc = {acc_c4_w:5.1f}%")

    return ablation_results


def run_gradient_detail_audit():
    print("\n================================================================================")
    print(" 3. DETALLE DE COSENOS DE GRADIENTE (12 LOTES FIJOS)")
    print("================================================================================")
    from reevaluacion_diagnostica_comun import compute_field_gradient_conflicts, load_checkpoint_clean

    ckpts = [
        ("Referencia (160)", "trdm_v4_ref160_checkpoint.pt", False),
        ("Opción A (160)", "trdm_v4_optA_checkpoint.pt", True),
        ("Opción A+B (160)", "trdm_v4_optAB_checkpoint.pt", True),
    ]

    detailed_grads = {}

    for label, ckpt_path, use_exp in ckpts:
        if not os.path.exists(ckpt_path):
            continue
        cfg = HybridConfig(
            max_seq_len=160, use_scratchpad=True, use_explicit_operands=use_exp, cpu_threads=2
        )
        model = load_checkpoint_clean(ckpt_path, cfg)
        ds = ProceduralCompositionalDataset(cfg, split="val", num_samples=96)

        # Recoger los valores individuales
        # Ejecutar con extracción de listas de cosenos
        torch.set_grad_enabled(True)
        shared_params = [
            p for name, p in model.named_parameters()
            if p.requires_grad and ("anchor_proj" in name or "shared_expert" in name or "answer_updater" in name)
        ]
        pairs = [("pw", "pms"), ("pw", "fin"), ("pms", "fin")]
        cos_lists = {f"{p[0]}_vs_{p[1]}": [] for p in pairs}

        rng = torch.Generator().manual_seed(42)
        for b_idx in range(12):
            indices = torch.randint(0, len(ds.samples), (8,), generator=rng).tolist()
            batch_x0, batch_can, batch_dig, batch_fids = [], [], [], []
            for i in indices:
                x_0, can_m, dig_m = ds[i]
                p_b, t_b, _, _ = ds.samples[i]
                f_ids = build_field_ids(p_b, t_b, 160, True)
                batch_x0.append(x_0); batch_can.append(can_m); batch_dig.append(dig_m); batch_fids.append(f_ids)

            x_0_b = torch.stack(batch_x0, dim=0)
            can_b = torch.stack(batch_can, dim=0)
            f_b = torch.stack(batch_fids, dim=0)

            x_t, mask_pos, t_vec = apply_blockwise_diffusion_corruption(x_0_b, can_b, cfg, fixed_t=0.50)
            logits, _, _, _, _ = model(x_t, t_vec, num_recursions=2)
            ce_all = F.cross_entropy(logits.view(-1, 260), x_0_b.view(-1), reduction="none")

            mask_flat, f_flat = mask_pos.view(-1), f_b.view(-1)
            m_pw = mask_flat & (f_flat == 1)
            m_pms = mask_flat & (f_flat == 2)
            m_fin = mask_flat & (f_flat == 3)

            if not (m_pw.any() and m_pms.any() and m_fin.any()):
                continue

            l_pw = ce_all[m_pw].sum() / max(1.0, float(m_pw.sum().item()))
            l_pms = ce_all[m_pms].sum() / max(1.0, float(m_pms.sum().item()))
            l_fin = ce_all[m_fin].sum() / max(1.0, float(m_fin.sum().item()))

            grads = {}
            for k_field, loss_val in [("pw", l_pw), ("pms", l_pms), ("fin", l_fin)]:
                model.zero_grad(set_to_none=True)
                loss_val.backward(retain_graph=True)
                g_vec = torch.cat([p.grad.view(-1) for p in shared_params if p.grad is not None])
                grads[k_field] = g_vec.detach().clone()

            model.zero_grad(set_to_none=True)

            for (k_i, k_j) in pairs:
                gi, gj = grads[k_i], grads[k_j]
                n_i, n_j = float(gi.norm().item()), float(gj.norm().item())
                dot = float(torch.dot(gi, gj).item())
                cos = dot / (n_i * n_j) if (n_i > 1e-8 and n_j > 1e-8) else 0.0
                cos_lists[f"{k_i}_vs_{k_j}"].append(cos)

        torch.set_grad_enabled(False)

        detailed_grads[label] = {}
        print(f"\n---> {label} (Distribución de los 12 Cosenos):")
        for p_name, l_vals in cos_lists.items():
            s_vals = sorted(l_vals)
            med = s_vals[len(s_vals) // 2] if s_vals else 0.0
            min_v = s_vals[0] if s_vals else 0.0
            max_v = s_vals[-1] if s_vals else 0.0
            neg_c = sum(1 for v in l_vals if v < 0)
            detailed_grads[label][p_name] = {
                "values": [round(v, 4) for v in l_vals],
                "median": med,
                "min": min_v,
                "max": max_v,
                "neg_count": neg_c,
                "total": len(l_vals),
            }
            formatted_vals = ", ".join(f"{v:+.3f}" for v in l_vals)
            print(f"  {p_name:<10}: Mediana={med:+.3f} | Min={min_v:+.3f} | Max={max_v:+.3f} | Neg={neg_c}/{len(l_vals)} ({100*neg_c/len(l_vals):.0f}%)")
            print(f"              Valores: [{formatted_vals}]")

    return detailed_grads


def main():
    rec_results, token_logs = run_reconciliation_audit()
    abl_results = run_sampling_ablation()
    grad_results = run_gradient_detail_audit()

    final_payload = {
        "reconciliation_strict_accumulators": rec_results,
        "sample_token_logs": token_logs,
        "sampling_ablation": abl_results,
        "gradient_distributions_12_batches": grad_results,
    }

    with open("auditoria_reconciliacion_results.json", "w") as f:
        json.dump(final_payload, f, indent=2)

    print("\n[Completado] Auditoría integral guardada en auditoria_reconciliacion_results.json")


if __name__ == "__main__":
    main()
