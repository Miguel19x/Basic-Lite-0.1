# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.benchmark as benchmark


def quantize_ternary_ste(x: torch.Tensor, scale: float) -> torch.Tensor:
    scaled = x / (scale + 1e-6)
    rounded = torch.clamp(torch.round(scaled), -1.0, 1.0)
    # Straight-Through Estimator
    return scaled + (rounded - scaled).detach()


def train_ternary_low_rank(
    w_target: torch.Tensor,
    rank: int,
    num_steps: int = 150,
    lr: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Optimiza factores ternarios A [out_f, rank] y B [rank, in_f] con STE
    para minimizar ||W - scale * (A @ B)||_F^2.
    """
    out_f, in_f = w_target.shape

    # Inicialización mediante SVD truncado
    U, S, Vh = torch.linalg.svd(w_target.float(), full_matrices=False)
    Ur = U[:, :rank]
    Sr = S[:rank]
    Vhr = Vh[:rank, :]

    sqrt_s = torch.diag(torch.sqrt(Sr))
    A_init = Ur @ sqrt_s
    B_init = sqrt_s @ Vhr

    A_param = nn.Parameter(A_init.clone())
    B_param = nn.Parameter(B_init.clone())
    scale_param = nn.Parameter(torch.tensor(1.0))

    opt = optim.Adam([A_param, B_param, scale_param], lr=lr)

    for _ in range(num_steps):
        opt.zero_grad()
        # Escala por fila/columna o escalar
        s_a = A_param.abs().mean().clamp(min=1e-4).detach()
        s_b = B_param.abs().mean().clamp(min=1e-4).detach()

        A_tern = quantize_ternary_ste(A_param, s_a)
        B_tern = quantize_ternary_ste(B_param, s_b)

        eff_scale = scale_param * s_a * s_b
        W_pred = eff_scale * (A_tern @ B_tern)

        loss = nn.functional.mse_loss(W_pred, w_target.float())
        loss.backward()
        opt.step()

    with torch.no_grad():
        s_a = A_param.abs().mean().clamp(min=1e-4)
        s_b = B_param.abs().mean().clamp(min=1e-4)
        A_final = torch.clamp(torch.round(A_param / s_a), -1.0, 1.0).to(torch.int8)
        B_final = torch.clamp(torch.round(B_param / s_b), -1.0, 1.0).to(torch.int8)
        final_scale = float((scale_param * s_a * s_b).item())

    return A_final, B_final, final_scale


def evaluate_ternary_factorization():
    torch.set_num_threads(2)
    torch.manual_seed(42)

    in_f = 192
    out_f = 384
    # Matriz original ternaria {-1, 0, 1}
    W_orig = torch.randint(-1, 2, (out_f, in_f), dtype=torch.int8)
    W_float = W_orig.float()

    # Memoria base de la matriz ternaria (2 bits por peso)
    orig_weights = out_f * in_f
    orig_bytes_2bit = (orig_weights * 2) // 8

    # Activaciones de prueba
    batch_size = 1
    x_test = torch.randn(batch_size, in_f)
    y_ref = x_test @ W_float.t()

    ranks = [8, 16, 24, 32, 48, 64, 96, 128]
    records = []

    print(f"=== Evaluación Factorización de Bajo Rango Ternaria ({out_f}x{in_f}) ===")
    print(f"Original: {orig_weights} pesos ternarios = {orig_bytes_2bit / 1024:.2f} KiB (2 bits)\n")
    print(f"{'Rank':<6}{'Parámetros':<12}{'Memoria (KiB)':<15}{'Compresión':<12}{'Error Rel Frob':<16}{'Sim Coseno':<12}{'Latencia (us)':<12}")
    print("-" * 85)

    for r in ranks:
        A_tern, B_tern, scale = train_ternary_low_rank(W_orig, rank=r, num_steps=120)

        # Reconstrucción
        W_recon = scale * (A_tern.float() @ B_tern.float())
        fro_err = float((torch.norm(W_float - W_recon) / torch.norm(W_float)).item())

        y_approx = x_test @ W_recon.t()
        cos_sim = float(nn.functional.cosine_similarity(y_ref, y_approx, dim=-1).mean().item())

        # Parámetros y bytes a 2 bits
        fact_params = out_f * r + r * in_f
        fact_bytes_2bit = ((fact_params * 2) // 8) + 4  # + 4 bytes para scale float
        compression = orig_bytes_2bit / fact_bytes_2bit

        # Medición con Timer
        def run_orig():
            return x_test @ W_float.t()

        def run_fact():
            return (x_test @ B_tern.float().t()) @ (scale * A_tern.float().t())

        t_fact = benchmark.Timer(
            stmt="run_fact()",
            globals={"run_fact": run_fact},
            num_threads=2,
        ).blocked_autorange(min_run_time=0.3)

        lat_us = t_fact.median * 1e6

        print(
            f"{r:<6}{fact_params:<12}{fact_bytes_2bit / 1024:<15.2f}{compression:<12.2f}x"
            f"{fro_err:<16.4f}{cos_sim:<12.4f}{lat_us:<12.2f}"
        )

        records.append({
            "rank": r,
            "params": fact_params,
            "bytes_2bit": fact_bytes_2bit,
            "compression_ratio": compression,
            "frobenius_relative_error": fro_err,
            "cosine_similarity": cos_sim,
            "latency_us": lat_us,
        })

    with open("experimento_factorizacion_ternaria_results.json", "w") as f:
        json.dump(records, f, indent=2)

    return records


if __name__ == "__main__":
    from typing import Tuple
    evaluate_ternary_factorization()
