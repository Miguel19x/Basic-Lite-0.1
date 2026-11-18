# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

import json
import math
import re
import os
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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


def load_checkpoint_clean(path: str, cfg: HybridConfig) -> TernaryRecursiveDiffusionLM:
    model = TernaryRecursiveDiffusionLM(cfg)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_strict_common_loss_and_acc(
    logits: torch.Tensor,
    x_0: torch.Tensor,
    mask_positions: torch.Tensor,
    digit_mask: torch.Tensor,
    field_ids: torch.Tensor,
    cfg: HybridConfig,
) -> Dict[str, float]:
    """
    Calcula pérdidas y exactitudes diagnósticas comunes desacopladas estrictamente
    por sus propios denominadores de posiciones enmascaradas.
    """
    B, L, V = logits.shape
    logits_flat = logits.view(-1, V)
    x_0_flat = x_0.view(-1)
    mask_flat = mask_positions.view(-1)
    dig_flat = digit_mask.view(-1)
    field_flat = field_ids.view(-1)

    ce_all = F.cross_entropy(logits_flat, x_0_flat, reduction="none")

    # Mascaras por campo
    m_all = mask_flat
    m_pw = mask_flat & (field_flat == 1)
    m_pms = mask_flat & (field_flat == 2)
    m_fin = mask_flat & (field_flat == 3)

    preds_flat = logits_flat.argmax(dim=-1)
    correct_flat = (preds_flat == x_0_flat)

    def _stats(m_sub, m_sub_dig):
        count = float(m_sub.sum().item())
        loss_val = float((ce_all[m_sub].sum() / max(1.0, count)).item()) if count > 0 else 0.0
        dig_count = float(m_sub_dig.sum().item())
        dig_acc = float((correct_flat[m_sub_dig].sum().float() / max(1.0, dig_count)).item()) if dig_count > 0 else 0.0
        all_acc = float((correct_flat[m_sub].sum().float() / max(1.0, count)).item()) if count > 0 else 0.0
        return loss_val, dig_acc, all_acc, count, dig_count

    l_all, dig_all, all_acc_all, _, _ = _stats(m_all, m_all & dig_flat)
    l_pw, dig_pw, _, _, _ = _stats(m_pw, m_pw & dig_flat)
    l_pms, dig_pms, _, _, _ = _stats(m_pms, m_pms & dig_flat)
    l_fin, dig_fin, _, _, _ = _stats(m_fin, m_fin & dig_flat)

    return {
        "loss_common_global": l_all,
        "loss_pw": l_pw,
        "loss_pms": l_pms,
        "loss_final": l_fin,
        "dig_acc_global": dig_all,
        "dig_acc_pw": dig_pw,
        "dig_acc_pms": dig_pms,
        "dig_acc_final": dig_fin,
        "all_acc_global": all_acc_all,
    }


def evaluate_diagnostic_staircase(
    model: TernaryRecursiveDiffusionLM,
    dataset: ProceduralCompositionalDataset,
    cfg: HybridConfig,
) -> Dict[str, Any]:
    """
    Escalera de Ocultación Progresiva:
    1. Enmascaramiento parcial habitual (t=0.50 fijo)
    2. Campo numérico Pw completamente enmascarado
    3. Bloque de respuesta completo enmascarado (single-pass)
    4. Generación iterativa habitual (efecto conjunto modelo + muestreador)
    """
    model.eval()
    n_samples = len(dataset.samples)

    # Nivel 1: Enmascaramiento Parcial Habitual (t=0.50)
    l1_stats = []
    torch.manual_seed(42)
    for idx in range(n_samples):
        x_0, can_mask, digit_mask = dataset[idx]
        p_b, t_b, _, _ = dataset.samples[idx]
        field_ids = build_field_ids(p_b, t_b, cfg.max_seq_len, cfg.use_scratchpad)

        x_0_b = x_0.unsqueeze(0)
        can_b = can_mask.unsqueeze(0)
        dig_b = digit_mask.unsqueeze(0)
        f_b = field_ids.unsqueeze(0)

        x_t, mask_pos, t_vec = apply_blockwise_diffusion_corruption(x_0_b, can_b, cfg, fixed_t=0.50)
        with torch.no_grad():
            out_logits, _, _, _, _ = model(x_t, t_vec, num_recursions=2)
            st = compute_strict_common_loss_and_acc(out_logits, x_0_b, mask_pos, dig_b, f_b, cfg)
            l1_stats.append(st)

    n1_avg = {k: sum(s[k] for s in l1_stats) / len(l1_stats) for k in l1_stats[0]}

    # Nivel 2: Campo Pw completamente enmascarado (ocultar todo Pw, resto visible)
    l2_stats = []
    for idx in range(n_samples):
        x_0, can_mask, digit_mask = dataset[idx]
        p_b, t_b, _, _ = dataset.samples[idx]
        field_ids = build_field_ids(p_b, t_b, cfg.max_seq_len, cfg.use_scratchpad)

        x_0_b = x_0.unsqueeze(0)
        dig_b = digit_mask.unsqueeze(0)
        f_b = field_ids.unsqueeze(0)

        # Enmascarar estrictamente el campo 1 (Pw)
        m_pw_full = (f_b == 1)
        x_t2 = x_0_b.clone()
        x_t2[m_pw_full] = cfg.mask_token_id
        t_vec2 = torch.full((1, cfg.max_seq_len), 0.50)

        with torch.no_grad():
            out_logits2, _, _, _, _ = model(x_t2, t_vec2, num_recursions=2)
            st2 = compute_strict_common_loss_and_acc(out_logits2, x_0_b, m_pw_full, dig_b, f_b, cfg)
            l2_stats.append(st2)

    n2_avg = {k: sum(s[k] for s in l2_stats) / len(l2_stats) for k in l2_stats[0]}

    # Nivel 3: Bloque de respuesta completo enmascarado (single pass, todo el target oculto)
    l3_stats = []
    for idx in range(n_samples):
        x_0, can_mask, digit_mask = dataset[idx]
        p_b, t_b, _, _ = dataset.samples[idx]
        field_ids = build_field_ids(p_b, t_b, cfg.max_seq_len, cfg.use_scratchpad)

        x_0_b = x_0.unsqueeze(0)
        dig_b = digit_mask.unsqueeze(0)
        f_b = field_ids.unsqueeze(0)

        # Enmascarar todos los campos del target (1, 2, 3, 4)
        m_target_full = (f_b >= 1) & (f_b <= 4)
        x_t3 = x_0_b.clone()
        x_t3[m_target_full] = cfg.mask_token_id
        t_vec3 = torch.full((1, cfg.max_seq_len), 1.0)

        with torch.no_grad():
            out_logits3, _, _, _, _ = model(x_t3, t_vec3, num_recursions=2)
            st3 = compute_strict_common_loss_and_acc(out_logits3, x_0_b, m_target_full, dig_b, f_b, cfg)
            l3_stats.append(st3)

    n3_avg = {k: sum(s[k] for s in l3_stats) / len(l3_stats) for k in l3_stats[0]}

    return {
        "nivel_1_enmascaramiento_parcial": n1_avg,
        "nivel_2_campo_pw_completamente_oculto": n2_avg,
        "nivel_3_target_completamente_oculto_singlepass": n3_avg,
    }


def compute_field_gradient_conflicts(
    model: TernaryRecursiveDiffusionLM,
    dataset: ProceduralCompositionalDataset,
    cfg: HybridConfig,
    num_batches: int = 12,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Diagnóstico de Conflicto de Gradientes:
    Calcula normas, productos escalares y similitud coseno entre gradientes
    de los campos L_pw, L_pms y L_fin sobre parámetros compartidos del núcleo.
    NO modifica ni actualiza ningún peso.
    """
    model.eval()
    torch.set_grad_enabled(True)

    # Identificar parámetros compartidos del núcleo recurrente
    shared_params = []
    for name, p in model.named_parameters():
        if p.requires_grad and ("anchor_proj" in name or "shared_expert" in name or "answer_updater" in name):
            shared_params.append(p)

    pairs = [("pw", "pms"), ("pw", "fin"), ("pms", "fin")]
    stats = {p: {"cosines": [], "dots": [], "neg_cos_count": 0, "norm_i": [], "norm_j": []} for p in pairs}

    rng = torch.Generator().manual_seed(42)

    for b_idx in range(num_batches):
        indices = torch.randint(0, len(dataset.samples), (batch_size,), generator=rng).tolist()
        batch_x0, batch_can, batch_dig, batch_fids = [], [], [], []
        for i in indices:
            x_0, can_m, dig_m = dataset[i]
            p_b, t_b, _, _ = dataset.samples[i]
            f_ids = build_field_ids(p_b, t_b, cfg.max_seq_len, cfg.use_scratchpad)
            batch_x0.append(x_0)
            batch_can.append(can_m)
            batch_dig.append(dig_m)
            batch_fids.append(f_ids)

        x_0_b = torch.stack(batch_x0, dim=0)
        can_b = torch.stack(batch_can, dim=0)
        dig_b = torch.stack(batch_dig, dim=0)
        f_b = torch.stack(batch_fids, dim=0)

        # Corrupción habitual a t=0.50
        x_t, mask_pos, t_vec = apply_blockwise_diffusion_corruption(x_0_b, can_b, cfg, fixed_t=0.50)

        # Forward
        logits, _, _, _, _ = model(x_t, t_vec, num_recursions=2)
        V = logits.size(-1)
        ce_all = F.cross_entropy(logits.view(-1, V), x_0_b.view(-1), reduction="none")

        mask_flat = mask_pos.view(-1)
        f_flat = f_b.view(-1)

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
            n_i = float(gi.norm().item())
            n_j = float(gj.norm().item())
            dot = float(torch.dot(gi, gj).item())
            if n_i > 1e-8 and n_j > 1e-8:
                cos = dot / (n_i * n_j)
            else:
                cos = 0.0

            stats[(k_i, k_j)]["cosines"].append(cos)
            stats[(k_i, k_j)]["dots"].append(dot)
            stats[(k_i, k_j)]["norm_i"].append(n_i)
            stats[(k_i, k_j)]["norm_j"].append(n_j)
            if cos < 0:
                stats[(k_i, k_j)]["neg_cos_count"] += 1

    torch.set_grad_enabled(False)

    summary = {}
    for pair, d in stats.items():
        n = len(d["cosines"])
        pair_name = f"{pair[0]}_vs_{pair[1]}"
        if n > 0:
            avg_cos = sum(d["cosines"]) / n
            avg_dot = sum(d["dots"]) / n
            neg_frac = d["neg_cos_count"] / n
            avg_ni = sum(d["norm_i"]) / n
            avg_nj = sum(d["norm_j"]) / n
        else:
            avg_cos, avg_dot, neg_frac, avg_ni, avg_nj = 0.0, 0.0, 0.0, 0.0, 0.0

        summary[pair_name] = {
            "mean_cosine": avg_cos,
            "mean_dot_product": avg_dot,
            "fraction_negative_cosine": neg_frac,
            "mean_norm_i": avg_ni,
            "mean_norm_j": avg_nj,
            "batches_evaluated": n,
        }

    return summary


def run_full_diagnostic_suite():
    torch.set_num_threads(2)
    torch.manual_seed(42)

    ckpts = [
        ("Referencia (160)", "trdm_v4_ref160_checkpoint.pt", False),
        ("Opción A (160)", "trdm_v4_optA_checkpoint.pt", True),
        ("Opción A+B (160)", "trdm_v4_optAB_checkpoint.pt", True),
    ]

    results = {}

    print("================================================================================")
    print(" SUITE DIAGNÓSTICA COMÚN: REEVALUACIÓN, ESCALERA DE OCULTACIÓN Y GRADIENTES")
    print("================================================================================")

    for label, ckpt_path, use_exp in ckpts:
        if not os.path.exists(ckpt_path):
            print(f"Error: {ckpt_path} no encontrado.")
            continue

        print(f"\n---> Evaluando Variante: {label} (use_explicit_operands={use_exp})")
        cfg = HybridConfig(
            max_seq_len=160,
            use_scratchpad=True,
            use_explicit_operands=use_exp,
            cpu_threads=2,
            dev_mode=False,
        )

        model = load_checkpoint_clean(ckpt_path, cfg)
        ds_val = ProceduralCompositionalDataset(cfg, split="val", num_samples=96)

        print("  1. Evaluando Escalera de Ocultación Progresiva...")
        staircase = evaluate_diagnostic_staircase(model, ds_val, cfg)

        print("  2. Diagnosticando Conflicto de Gradientes entre Campos...")
        grad_conflicts = compute_field_gradient_conflicts(model, ds_val, cfg, num_batches=12, batch_size=8)

        results[label] = {
            "staircase": staircase,
            "gradient_conflicts": grad_conflicts,
        }

        # Resumen impreso inmediato
        n1 = staircase["nivel_1_enmascaramiento_parcial"]
        n2 = staircase["nivel_2_campo_pw_completamente_oculto"]
        n3 = staircase["nivel_3_target_completamente_oculto_singlepass"]

        print(f"  [Nivel 1 Parcial] Pérdida Común: {n1['loss_common_global']:.3f} | DigAcc Pw: {n1['dig_acc_pw']*100:.1f}% | DigAcc Fin: {n1['dig_acc_final']*100:.1f}%")
        print(f"  [Nivel 2 Pw Oculto] Pérdida Pw: {n2['loss_pw']:.3f} | DigAcc Pw: {n2['dig_acc_pw']*100:.1f}%")
        print(f"  [Nivel 3 Target Oculto] Pérdida Global: {n3['loss_common_global']:.3f} | DigAcc Global: {n3['dig_acc_global']*100:.1f}%")
        for pair_name, g_info in grad_conflicts.items():
            print(f"  [Gradiente {pair_name}] Cos: {g_info['mean_cosine']:.3f} | Frac Negativo: {g_info['fraction_negative_cosine']*100:.1f}% | ||gi||: {g_info['mean_norm_i']:.2e} | ||gj||: {g_info['mean_norm_j']:.2e}")

    with open("reevaluacion_diagnostica_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n[Completado] Resultados guardados en reevaluacion_diagnostica_results.json")
    return results


if __name__ == "__main__":
    run_full_diagnostic_suite()
