#!/usr/bin/env python3
# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Basic-Lite 0.2 (Fase 2) — Script Maestro de Entrenamiento y Evaluación Confirmatoria TRDM-v6 para Google Colab
Ejecuta el pipeline de 2 etapas con Abacus Embeddings, Staged Warmup y evaluación completa (DEV, Test-Comp, Test-Azar).
Compatible con GPU (CUDA) y CPU. Persiste automáticamente en /content/drive/MyDrive/BasicLite/ si está disponible.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import HybridConfig
from data_v2 import ProceduralCompositionalDatasetV2
from trdm.models.trdm_v6_causal import TRDMv6CausalModel
from eval_v6 import evaluate_causal_v6


OUTPUT_DIR = "/content/drive/MyDrive/BasicLite"
if not os.path.exists("/content/drive/MyDrive"):
    print("[*] Google Drive no montado en /content/drive/MyDrive. Guardando salidas en './output'.")
    OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_SEEDS = [42, 123, 456, 789, 999]


def collate_fn_v6(batch, max_seq_len: int, pad_token_id: int, bos_token_id: int, eos_token_id: int):
    x_0_list = []
    plen_list = []
    for prompt_b, target_b, _, _ in batch:
        plen = 1 + len(prompt_b)
        plen_list.append(plen)
        full_seq = [bos_token_id] + list(prompt_b) + list(target_b) + [eos_token_id]
        full_seq = full_seq[:max_seq_len]
        pad_len = max_seq_len - len(full_seq)
        full_seq = full_seq + [pad_token_id] * pad_len
        x_0_list.append(full_seq)
    return torch.tensor(x_0_list, dtype=torch.long), plen_list


def run_seed_training_and_eval(
    seed: int,
    steps: int = 3000,
    lr: float = 3e-4,
    batch_size: int = 32,
    warmup_ratio: float = 0.40,
    eval_interval: int = 500,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("\n" + "=" * 80)
    print(f"[SEMILLA {seed}] Iniciando TRDM-v6 Causal de 2 Etapas ({steps} pasos en {device})")
    print(f"Hiperparámetros: LR={lr:.2e} | Lote={batch_size} | WarmupRatio={warmup_ratio:.2f}")
    print("=" * 80)

    cfg = HybridConfig(
        seed=seed,
        device=str(device),
        use_scratchpad=True,
        use_explicit_operands=False,
    )

    train_dataset = ProceduralCompositionalDatasetV2(cfg, split="train")
    dev_dataset = ProceduralCompositionalDatasetV2(cfg, split="dev")
    test_comp_dataset = ProceduralCompositionalDatasetV2(cfg, split="test-composicional")
    test_azar_dataset = ProceduralCompositionalDatasetV2(cfg, split="test-azar")

    model = TRDMv6CausalModel(cfg).to(device)
    param_groups = model.configure_adamw_groups()
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.98), eps=1e-8)

    def get_batch_iterator():
        while True:
            indices = torch.randperm(len(train_dataset.samples)).tolist()
            for i in range(0, len(indices) - batch_size + 1, batch_size):
                batch_samples = [train_dataset.samples[idx] for idx in indices[i : i + batch_size]]
                yield collate_fn_v6(
                    batch_samples,
                    cfg.max_seq_len,
                    cfg.pad_token_id,
                    cfg.bos_token_id,
                    cfg.eos_token_id,
                )

    batch_iter = get_batch_iterator()
    t0_start = time.time()
    best_dev_score = -1.0
    best_dev_em = -1.0

    ckpt_path = os.path.join(OUTPUT_DIR, f"trdm_v6_seed{seed}_best.pt")

    warmup_steps = int(steps * warmup_ratio)
    model.train()

    for step in range(1, steps + 1):
        x_0, plen_list = next(batch_iter)
        x_0 = x_0.to(device)
        t_noise = torch.ones(x_0.shape[0], device=device)

        if step <= warmup_steps:
            sched_p = 1.0
        else:
            progress = (step - warmup_steps) / max(1, steps - warmup_steps)
            sched_p = max(0.70, 1.0 - 0.30 * progress)

        optimizer.zero_grad()
        out = model.forward_two_stage(x_0, plen_list, t_noise, scheduled_sampling_p=sched_p)
        loss = out["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 100 == 0 or step == 1:
            print(
                f"[S{seed}|{step:04d}/{steps}] Loss: {loss.item():.4f} "
                f"(S1 Pw/Pms: {out['loss_stage1'].item():.4f}, S2 Tot: {out['loss_stage2'].item():.4f}) | "
                f"SchedP: {sched_p:.2f}"
            )

        if step % eval_interval == 0 or step == steps:
            dev_metrics = evaluate_causal_v6(model, dev_dataset, device)
            print(
                f"  -> DEV EVAL [Paso {step:04d}]: ExactMatch={dev_metrics['exact_match_pct']:.2f}% "
                f"({dev_metrics['em_count']}/{dev_metrics['total_cases']}) | "
                f"ScratchpadEM={dev_metrics['scratchpad_exact_pct']:.2f}% | "
                f"FinalEM={dev_metrics['final_exact_pct']:.2f}% | "
                f"DigAcc={dev_metrics['digit_acc_pct']:.2f}%"
            )

            dev_score = (
                dev_metrics["exact_match_pct"] * 10000.0
                + dev_metrics["final_exact_pct"] * 1000.0
                + dev_metrics["scratchpad_exact_pct"] * 100.0
                + dev_metrics["digit_acc_pct"]
            )

            if dev_score > best_dev_score:
                best_dev_score = dev_score
                best_dev_em = dev_metrics["exact_match_pct"]
                print(f"     [*] Nuevo mejor checkpoint (Score={dev_score:.2f}) -> {ckpt_path}")
                torch.save(
                    {
                        "step": step,
                        "seed": seed,
                        "model_state": model.state_dict(),
                        "cfg": cfg,
                        "dev_metrics": dev_metrics,
                        "best_dev_score": best_dev_score,
                    },
                    ckpt_path,
                )
            model.train()

    train_duration = time.time() - t0_start
    print(f"[SEMILLA {seed}] Entrenamiento completado en {train_duration:.2f} s ({train_duration / steps * 1000:.1f} ms/paso)")

    # Cargar mejor checkpoint para evaluaciones oficiales
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"[SEMILLA {seed}] Checkpoint óptimo restaurado desde: {ckpt_path} (Paso {ckpt['step']})")

    print(f"\n[SEMILLA {seed}] Evaluando DEV final...")
    final_dev = evaluate_causal_v6(model, dev_dataset, device)

    print(f"[SEMILLA {seed}] Evaluando TEST-COMPOSICIONAL (480 casos)...")
    final_comp = evaluate_causal_v6(model, test_comp_dataset, device)

    print(f"[SEMILLA {seed}] Evaluando TEST-AZAR (240 casos)...")
    final_azar = evaluate_causal_v6(model, test_azar_dataset, device)

    print(
        f"\n[RESUMEN SEMILLA {seed}]\n"
        f"  DEV:       EM={final_dev['exact_match_pct']:.2f}% | SP_EM={final_dev['scratchpad_exact_pct']:.2f}% | FinalEM={final_dev['final_exact_pct']:.2f}%\n"
        f"  TEST-COMP: EM={final_comp['exact_match_pct']:.2f}% | SP_EM={final_comp['scratchpad_exact_pct']:.2f}% | FinalEM={final_comp['final_exact_pct']:.2f}%\n"
        f"  TEST-AZAR: EM={final_azar['exact_match_pct']:.2f}% | SP_EM={final_azar['scratchpad_exact_pct']:.2f}% | FinalEM={final_azar['final_exact_pct']:.2f}%"
    )

    return {
        "seed": seed,
        "train_duration_sec": train_duration,
        "dev": final_dev,
        "test_comp": final_comp,
        "test_azar": final_azar,
    }


def aggregate_stats(values: List[float]) -> Dict[str, Any]:
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1)) if n > 1 else 0.0
    t_crit = 2.776 if n == 5 else (1.96 if n > 30 else 2.0)
    ci95 = t_crit * (std / math.sqrt(n)) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def main():
    parser = argparse.ArgumentParser(description="Master Runner TRDM-v6 para Colab")
    parser.add_argument("--steps", type=int, default=3000, help="Pasos de entrenamiento por semilla")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de lote")
    parser.add_argument("--lr", type=float, default=3e-4, help="Tasa de aprendizaje")
    parser.add_argument("--warmup_ratio", type=float, default=0.40, help="Fracción de pasos con sched_p = 1.0")
    parser.add_argument("--eval_interval", type=int, default=500, help="Intervalo de evaluación en DEV")
    parser.add_argument("--pilot_only", action="store_true", help="Ejecutar solo semilla 42")
    parser.add_argument("--seeds", type=str, default="", help="Lista de semillas personalizadas separadas por coma (ej. '789,999')")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda o cpu")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.seeds.strip():
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    elif args.pilot_only:
        seeds = [42]
    else:
        seeds = DEFAULT_SEEDS

    print("=" * 80)
    print("BASIC-LITE 0.2 (FASE 2) — EXPERIMENTO CONFIRMATORIO TRDM-v6")
    print(f"Dispositivo activo: {device} | Semillas a ejecutar: {seeds} | Pasos/semilla: {args.steps}")
    print(f"Directorio de guardado: {OUTPUT_DIR}")
    print("=" * 80)

    out_json = os.path.join(OUTPUT_DIR, "v6_confirmatorio_results.json")
    per_seed_results = {}

    # Si ya existe un reporte previo en Drive, recuperar las semillas ya completadas
    if os.path.exists(out_json):
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                per_seed_results = prev_data.get("per_seed_results", {})
                print(f"[*] Recuperadas {len(per_seed_results)} semillas previas desde: {out_json}")
        except Exception as e:
            print(f"[!] Aviso al leer reporte previo: {e}")

    for seed in seeds:
        res = run_seed_training_and_eval(
            seed=seed,
            steps=args.steps,
            lr=args.lr,
            batch_size=args.batch_size,
            warmup_ratio=args.warmup_ratio,
            eval_interval=args.eval_interval,
            device=device,
        )
        per_seed_results[str(seed)] = res

    # Recalcular estadísticas agregadas sobre todas las semillas disponibles
    all_available_seeds = [int(s) for s in per_seed_results.keys()]
    summary = {}
    if len(all_available_seeds) > 1:
        for split_key in ["dev", "test_comp", "test_azar"]:
            for metric in ["exact_match_pct", "scratchpad_exact_pct", "final_exact_pct", "digit_acc_pct", "fmt_valid_pct"]:
                vals = [per_seed_results[str(s)][split_key][metric] for s in all_available_seeds if split_key in per_seed_results[str(s)]]
                if vals:
                    summary[f"{split_key}_{metric}"] = aggregate_stats(vals)

    final_payload = {
        "experiment": "TRDM_v6_Causal_2Stage_Confirmatorio",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "steps_per_seed": args.steps,
        "warmup_ratio": args.warmup_ratio,
        "seeds": sorted(all_available_seeds),
        "per_seed_results": per_seed_results,
        "aggregate_summary": summary,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"[*] Experimento completado. Resultados consolidados guardados en: {out_json}")
    if len(all_available_seeds) > 1:
        print(f"\n--- RESUMEN GLOBAL ({len(all_available_seeds)} SEMILLAS: {sorted(all_available_seeds)}) ---")
        for key, stats in summary.items():
            print(f"  {key:30s}: {stats['mean']:.2f}% ± {stats['std']:.2f}% (IC95: {stats['ci95']:.2f}%) [min={stats['min']:.2f}%, max={stats['max']:.2f}%]")
    print("=" * 80)


if __name__ == "__main__":
    main()
