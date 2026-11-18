# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

import random
import torch

from config import HybridConfig
from model import TernaryRecursiveDiffusionLM
from data import ProceduralCompositionalDataset
from trdm.diffusion.sampler import (
    generate_block_causal_with_kv_cache,
    evaluate_exact_match_split,
)

__all__ = [
    "generate_block_causal_with_kv_cache",
    "evaluate_exact_match_split",
]


if __name__ == "__main__":
    cfg = HybridConfig()
    model = TernaryRecursiveDiffusionLM(cfg)
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.load_bitcos_archive(cfg.bitcos_export_path)
    model.prepare_for_eval()

    train_ds = ProceduralCompositionalDataset(cfg, split="train", num_samples=100)
    val_ds = ProceduralCompositionalDataset(cfg, split="val", num_samples=100)

    # 1. Prueba de memorización sobre 96 combinaciones de entrenamiento (muestreo representativo)
    rng_eval = random.Random(cfg.seed)
    train_subset = rng_eval.sample(train_ds.unique_records, min(96, len(train_ds.unique_records)))
    m_train = evaluate_exact_match_split(model, train_subset, cfg, r_infer=cfg.num_recursions)
    print(f"=== 1. Control de Memorización en Entrenamiento (96 tuplas Train, R={cfg.num_recursions}) ===")
    sub_em_str = f" | Sub EM: {m_train['sub_both_em_pct']:.1f}%" if "sub_both_em_pct" in m_train else ""
    print(
        f"Formato: {m_train['fmt_pct']:.1f}% | DigAcc: {m_train['digit_acc_pct']:.1f}% | "
        f"Watt EM: {m_train['watt_em_pct']:.1f}% | MS EM: {m_train['ms_em_pct']:.1f}% | "
        f"Full Tuple EM: {m_train['both_em_pct']:.1f}% (±{m_train['both_se_pct']:.1f}%){sub_em_str}"
    )

    # 2. Evaluación rigurosa de la Hipótesis H1 sobre las 96 combinaciones Held-Out completas
    print("\n=== 2. Evaluación Hipótesis H1 en las 96 Tuplas Reservadas (Held-Out OOD) ===")
    print(f"{'R_infer':<8} | {'Fmt %':<7} | {'DigAcc %':<9} | {'Watt EM %':<10} | {'MS EM %':<8} | {'Full EM % (±SE)':<18} | {'Latencia':<9}")
    print("-" * 82)
    for r_val in (1, 2, 4, 8):
        m_val = evaluate_exact_match_split(model, val_ds.unique_records, cfg, r_infer=r_val)
        em_str = f"{m_val['both_em_pct']:.1f}% (±{m_val['both_se_pct']:.1f}%)"
        extra_str = (
            f" | SubEM: {m_val['sub_both_em_pct']:>5.1f}% | AllEM: {m_val['all_em_pct']:>5.1f}%"
            if "sub_both_em_pct" in m_val
            else ""
        )
        print(
            f"R = {r_val:<4} | {m_val['fmt_pct']:>5.1f}% | {m_val['digit_acc_pct']:>7.1f}% | "
            f"{m_val['watt_em_pct']:>8.1f}% | {m_val['ms_em_pct']:>6.1f}% | {em_str:<18} | {m_val['latency_ms']:>6.1f} ms"
            f"{extra_str}"
        )
