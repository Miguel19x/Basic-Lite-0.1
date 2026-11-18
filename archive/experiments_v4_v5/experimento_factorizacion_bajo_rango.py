# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Experimento Aislado: Factorización de Bajo Rango en Proyección de Experto (TRDM-v5 / Basic-Lite 0.2)

Responde a la pregunta experimental planteada:
«¿Podemos sustituir una matriz ternaria por dos factores que, sumando precisión, escalas,
buffers y cálculo, ocupen menos y mantengan suficiente exactitud?»

Compara:
1. Matriz original ternaria (W1.58 / 2-bit) vs. Factores A y B (rango r in {16, 32, 48, 64, 96, 128}).
2. Presupuesto real de bytes (FP16, INT8, Ternario empaquetado).
3. Error de reconstrucción Frobenius y error relativo en activaciones reales.
4. Latencia de ejecución en CPU (W @ x vs A @ (B @ x)).
"""

import json
import time
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark

from trdm.layers.bitlinear import BitLinear


def print_title(title: str) -> None:
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def evaluar_factorizacion_rango(
    out_features: int = 384,
    in_features: int = 192,
    ranks: List[int] = [16, 32, 48, 64, 96, 128],
    num_samples: int = 64,
) -> Dict[str, Any]:
    print_title("EXPERIMENTO: Factorización de Bajo Rango W ≈ A @ B en Experto Ternario")

    torch.manual_seed(42)
    # 1. Simular matriz de pesos de experto con correlación de rango real (no puro ruido blanco)
    # Los pesos de redes neuronales reales tienen espectros de valores singulares con decaimiento rápido
    k_latent = 64
    w_latent = torch.randn(out_features, k_latent) @ torch.randn(k_latent, in_features)
    scale_w = w_latent.abs().mean()
    # Cuantizar a ternario W1.58
    w_ternary = torch.clamp(torch.round(w_latent / scale_w), -1.0, 1.0) * scale_w
    w_fp32 = w_ternary.clone()

    # Generar activaciones de prueba
    x_test = torch.randn(num_samples, in_features)
    y_ground_truth = x_test @ w_fp32.t()  # [N, out_features]
    norm_gt = y_ground_truth.norm()

    # Presupuesto de la matriz ternaria original
    total_weights_orig = out_features * in_features
    bytes_orig_ternary_2bit = (total_weights_orig * 2) // 8  # 18,432 bytes = 18.0 KiB
    bytes_orig_fp32 = total_weights_orig * 4                 # 294,912 bytes = 288.0 KiB

    # Descomposición SVD sobre la matriz de pesos
    U, S, Vh = torch.linalg.svd(w_fp32, full_matrices=False)

    resultados_rangos = []

    print(f"Dimensiones de Proyección   : {out_features} x {in_features} ({total_weights_orig} parámetros)")
    print(f"Tamaño Original Ternario (2b): {bytes_orig_ternary_2bit} bytes ({bytes_orig_ternary_2bit / 1024.0:.1f} KiB)")
    print(f"Tamaño Original FP32         : {bytes_orig_fp32} bytes ({bytes_orig_fp32 / 1024.0:.1f} KiB)")
    print("-" * 80)
    print(f"{'Rango r':<8} | {'Valores':<8} | {'Bytes FP16':<11} | {'Bytes INT8':<11} | {'Bytes 2-bit':<11} | {'Error Frob %':<12} | {'Error Act %':<12}")
    print("-" * 80)

    for r in ranks:
        # Factores A (out x r) y B (r x in)
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]

        # Distribuir la raíz cuadrada de los valores singulares entre ambos factores
        sqrt_S = torch.diag(torch.sqrt(S_r))
        A = U_r @ sqrt_S          # [out_features, r]
        B = sqrt_S @ Vh_r         # [r, in_features]

        # Reconstrucción matricial
        w_recon = A @ B
        error_frob = float((w_fp32 - w_recon).norm() / w_fp32.norm() * 100.0)

        # Salida aplicando la factorización consecutiva: y_approx = (x @ B.T) @ A.T
        y_approx = (x_test @ B.t()) @ A.t()
        error_act = float((y_ground_truth - y_approx).norm() / norm_gt * 100.0)

        # Presupuesto en bytes para diferentes precisiones de los factores
        total_vals_fact = r * (out_features + in_features)
        bytes_fp16 = total_vals_fact * 2
        bytes_int8 = total_vals_fact * 1
        bytes_2bit = (total_vals_fact * 2) // 8

        print(
            f"{r:<8} | {total_vals_fact:<8} | {bytes_fp16:<11} | {bytes_int8:<11} | {bytes_2bit:<11} | "
            f"{error_frob:<12.2f} | {error_act:<12.2f}"
        )

        resultados_rangos.append({
            "r": r,
            "values_count": total_vals_fact,
            "values_ratio_vs_orig": round(total_vals_fact / total_weights_orig, 3),
            "bytes_fp16": bytes_fp16,
            "bytes_int8": bytes_int8,
            "bytes_2bit": bytes_2bit,
            "ratio_bytes_fp16_vs_orig_ternary": round(bytes_fp16 / bytes_orig_ternary_2bit, 2),
            "ratio_bytes_int8_vs_orig_ternary": round(bytes_int8 / bytes_orig_ternary_2bit, 2),
            "ratio_bytes_2bit_vs_orig_ternary": round(bytes_2bit / bytes_orig_ternary_2bit, 2),
            "frobenius_error_pct": round(error_frob, 2),
            "activation_error_pct": round(error_act, 2),
        })

    print("-" * 80)

    # Benchmark de Latencia con torch.utils.benchmark.Timer para el rango óptimo r=32 y r=48
    print("\n--- Benchmark de Latencia en CPU (1 Hilo, Timer Oficial) ---")
    r_bench = 32
    U_r = U[:, :r_bench]
    S_r = S[:r_bench]
    Vh_r = Vh[:r_bench, :]
    sqrt_S = torch.diag(torch.sqrt(S_r))
    A_bench = (U_r @ sqrt_S).contiguous()
    B_bench = (sqrt_S @ Vh_r).contiguous()
    x_bench = torch.randn(1, in_features)

    def run_mat_directa():
        return x_bench @ w_fp32.t()

    def run_fact_dos_etapas():
        return (x_bench @ B_bench.t()) @ A_bench.t()

    timer_dir = benchmark.Timer(
        stmt="run_mat_directa()",
        globals={"run_mat_directa": run_mat_directa},
        num_threads=1,
    ).blocked_autorange(min_run_time=0.2)

    timer_fact = benchmark.Timer(
        stmt="run_fact_dos_etapas()",
        globals={"run_fact_dos_etapas": run_fact_dos_etapas},
        num_threads=1,
    ).blocked_autorange(min_run_time=0.2)

    lat_dir_us = timer_dir.median * 1e6
    lat_fact_us = timer_fact.median * 1e6

    print(f"Matriz Directa W @ x (384x192)      : Mediana = {lat_dir_us:.2f} µs (IQR: {timer_dir.iqr * 1e6:.2f} µs)")
    print(f"Factorizada A @ (B @ x) (r={r_bench})       : Mediana = {lat_fact_us:.2f} µs (IQR: {timer_fact.iqr * 1e6:.2f} µs)")
    print(f"Ratio de Latencia Directa / Fact    : {lat_dir_us / lat_fact_us:.2f}x")

    summary = {
        "dimensions": f"{out_features}x{in_features}",
        "orig_ternary_2bit_bytes": bytes_orig_ternary_2bit,
        "results_by_rank": resultados_rangos,
        "benchmark_latency": {
            "rank": r_bench,
            "direct_matmul_us": round(lat_dir_us, 2),
            "factorized_matmul_us": round(lat_fact_us, 2),
            "speedup_ratio": round(lat_dir_us / lat_fact_us, 2),
        },
    }

    with open("experimento_factorizacion_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    evaluar_factorizacion_rango()
