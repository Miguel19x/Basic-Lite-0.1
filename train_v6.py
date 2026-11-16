#!/usr/bin/env python3
"""
Basic-Lite 0.2 — Entrenamiento de Fase 2: TRDM-v6 Causal de 2 Etapas
Incorpora Abacus Positional Embeddings y Desacoplamiento Causal (Subtotales -> Totales).
"""

import os
import sys
import time
import argparse
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader

from config import HybridConfig
from data_v2 import ProceduralCompositionalDatasetV2
from trdm.models.trdm_v6_causal import TRDMv6CausalModel
from eval_v6 import evaluate_causal_v6


def collate_fn_v6(batch, max_seq_len: int, pad_token_id: int, bos_token_id: int, eos_token_id: int):
    """Collate function que devuelve x_0 y las longitudes de los prompts."""
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


def train_v6(
    steps: int = 1000,
    lr: float = 3e-4,
    batch_size: int = 32,
    seed: int = 42,
    device_name: str = "auto",
    eval_interval: int = 200,
    checkpoint_out: str = "trdm_v6_checkpoint.pt",
):
    torch.manual_seed(seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print("=" * 80)
    print("Basic-Lite 0.2 (Fase 2) — Entrenamiento TRDM-v6 Causal de 2 Etapas")
    print(f"Dispositivo: {device} | Semilla: {seed} | Pasos: {steps} | LR: {lr:.2e} | Lote: {batch_size}")
    print("=" * 80)

    cfg = HybridConfig(
        seed=seed,
        device=str(device),
        use_scratchpad=True,
        use_explicit_operands=False,
    )

    # Conjuntos de datos pre-registrados
    train_dataset = ProceduralCompositionalDatasetV2(cfg, split="train")
    dev_dataset = ProceduralCompositionalDatasetV2(cfg, split="dev")
    test_comp_dataset = ProceduralCompositionalDatasetV2(cfg, split="test-composicional")

    model = TRDMv6CausalModel(cfg).to(device)

    # Optimizador con grupos desacoplados de decaimiento
    param_groups = model.configure_adamw_groups()
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.98), eps=1e-8)

    # Creador de lotes
    def get_batch_iterator():
        while True:
            # Mezclar muestras
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

    print(f"[*] Inicio del entrenamiento. Muestras Train: {len(train_dataset)} | Dev: {len(dev_dataset)}")
    t0_start = time.time()
    best_dev_em = -1.0

    model.train()
    for step in range(1, steps + 1):
        x_0, plen_list = next(batch_iter)
        x_0 = x_0.to(device)
        t_noise = torch.ones(x_0.shape[0], device=device)

        # Scheduled sampling: decae suavemente de 0.95 a 0.70 a lo largo de los pasos
        sched_p = max(0.70, 0.95 - 0.25 * (step / steps))

        optimizer.zero_grad()
        out = model.forward_two_stage(x_0, plen_list, t_noise, scheduled_sampling_p=sched_p)
        loss = out["loss"]
        loss.backward()

        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            print(
                f"Paso {step:04d}/{steps} | Loss: {loss.item():.4f} "
                f"(S1 Pw/Pms: {out['loss_stage1'].item():.4f}, S2 Tot: {out['loss_stage2'].item():.4f}) | "
                f"SchedP: {sched_p:.2f}"
            )

        if step % eval_interval == 0 or step == steps:
            print(f"\n--- EVALUACIÓN EN DEV (Paso {step}) ---")
            dev_metrics = evaluate_causal_v6(model, dev_dataset, device)
            print(
                f"  DEV: ExactMatch={dev_metrics['exact_match_pct']:.2f}% "
                f"({dev_metrics['em_count']}/{dev_metrics['total_cases']}) | "
                f"ScratchpadEM={dev_metrics['scratchpad_exact_pct']:.2f}% | "
                f"FinalEM={dev_metrics['final_exact_pct']:.2f}% | "
                f"DigAcc={dev_metrics['digit_acc_pct']:.2f}% | "
                f"FmtValid={dev_metrics['fmt_valid_pct']:.2f}%"
            )

            if dev_metrics["exact_match_pct"] > best_dev_em:
                best_dev_em = dev_metrics["exact_match_pct"]
                print(f"  [*] Guardando mejor checkpoint en: {checkpoint_out}")
                torch.save(
                    {
                        "step": step,
                        "model_state": model.state_dict(),
                        "cfg": cfg,
                        "best_dev_em": best_dev_em,
                        "dev_metrics": dev_metrics,
                    },
                    checkpoint_out,
                )
            model.train()

    total_time = time.time() - t0_start
    print("\n" + "=" * 80)
    print(f"Entrenamiento completado en {total_time:.2f} s ({total_time / steps * 1000:.1f} ms/paso)")
    print(f"Mejor Exact Match en DEV: {best_dev_em:.2f}%")
    print("=" * 80)

    # Evaluación final en Test-Composicional
    print("\n--- EVALUACIÓN FINAL EN TEST-COMPOSICIONAL (480 casos) ---")
    if os.path.exists(checkpoint_out):
        ckpt = torch.load(checkpoint_out, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    comp_metrics = evaluate_causal_v6(model, test_comp_dataset, device)
    print(
        f"  TEST-COMP: ExactMatch={comp_metrics['exact_match_pct']:.2f}% "
        f"({comp_metrics['em_count']}/{comp_metrics['total_cases']}) | "
        f"ScratchpadEM={comp_metrics['scratchpad_exact_pct']:.2f}% | "
        f"FinalEM={comp_metrics['final_exact_pct']:.2f}% | "
        f"DigAcc={comp_metrics['digit_acc_pct']:.2f}% | "
        f"FmtValid={comp_metrics['fmt_valid_pct']:.2f}%"
    )

    return comp_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento TRDM-v6 Causal de 2 Etapas")
    parser.add_argument("--steps", type=int, default=1000, help="Pasos de entrenamiento")
    parser.add_argument("--lr", type=float, default=3e-4, help="Tasa de aprendizaje")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de lote")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--device", type=str, default="auto", help="cpu o cuda")
    parser.add_argument("--eval_interval", type=int, default=200, help="Intervalo de evaluación")
    parser.add_argument("--out", type=str, default="trdm_v6_checkpoint.pt", help="Ruta de checkpoint")
    args = parser.parse_args()

    train_v6(
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device_name=args.device,
        eval_interval=args.eval_interval,
        checkpoint_out=args.out,
    )
