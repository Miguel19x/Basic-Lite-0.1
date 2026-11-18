# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

import json
import os
import sys
import numpy as np
import torch
import torch.utils.benchmark as benchmark

from trdm.serialization.paged_bitcos import PagedBitCOSSerializer
from trdm.ops.paged_kernel import paged_direct_block_matmul
from trdm.ops.paged_c_kernel import (
    paged_direct_block_matmul_fast_c,
    paged_direct_matmul_fast_c_multi,
)


def run_verification_and_benchmark():
    torch.set_num_threads(2)
    torch.manual_seed(42)

    in_features = 192
    out_features = 384
    page_size = 4096
    batch_size = 1

    # 1. Matriz de pesos ternarios sintéticos
    raw_weights = torch.randint(-1, 2, (out_features, in_features), dtype=torch.int8)
    scale_w = 0.05
    dense_w_fp32 = (raw_weights.to(torch.float32) * scale_w).t().contiguous()

    # 2. Empaquetar páginas en formato 'fast'
    weights_np = raw_weights.numpy()
    rows_per_page = max(1, (page_size - 64) // max(1, int(in_features * 0.26) + 4))
    pages_fast = []
    for r_start in range(0, out_features, rows_per_page):
        r_end = min(out_features, r_start + rows_per_page)
        chunk = weights_np[r_start:r_end, :]
        p_bytes = PagedBitCOSSerializer.pack_ternary_page(
            row_start=r_start,
            out_features=out_features,
            in_features=in_features,
            scale_w=scale_w,
            q_w_chunk=chunk,
            page_size=page_size,
            format_type="fast",
        )
        pages_fast.append(p_bytes)

    # 3. Activaciones INT8
    x_int8 = torch.randint(-128, 127, (batch_size, in_features), dtype=torch.int8)
    x_fp32 = x_int8.to(torch.float32) * 0.02

    # 4. Verificación de equivalencia entera exacta
    # A. Python vs C por página
    max_diff_single = 0
    py_parts = []
    for p_bytes in pages_fast:
        _, _, _, acc_py = paged_direct_block_matmul(x_int8, p_bytes)
        _, _, _, acc_c = paged_direct_block_matmul_fast_c(x_int8, p_bytes)
        diff = (acc_py - acc_c).abs().max().item()
        if diff > max_diff_single:
            max_diff_single = diff
        py_parts.append(acc_py)

    acc_full_py = torch.cat(py_parts, dim=-1)

    # B. Python vs C Multi-Página Fisionado
    acc_full_multi_c = paged_direct_matmul_fast_c_multi(x_int8, pages_fast, out_features)
    max_diff_multi = (acc_full_py - acc_full_multi_c).abs().max().item()

    print(f"[Equivalencia Numérica] Diferencia entera máxima (Python vs C Individual): {max_diff_single}")
    print(f"[Equivalencia Numérica] Diferencia entera máxima (Python vs C Multi-Página): {max_diff_multi}")
    assert max_diff_single == 0, f"Fallo de equivalencia individual: {max_diff_single}"
    assert max_diff_multi == 0, f"Fallo de equivalencia multi-página: {max_diff_multi}"

    # 5. Funciones para benchmark con Timer
    def run_pages_python():
        accs = []
        for p in pages_fast:
            _, _, _, acc = paged_direct_block_matmul(x_int8, p)
            accs.append(acc)
        return torch.cat(accs, dim=-1)

    def run_pages_c_single():
        accs = []
        for p in pages_fast:
            _, _, _, acc = paged_direct_block_matmul_fast_c(x_int8, p)
            accs.append(acc)
        return torch.cat(accs, dim=-1)

    def run_pages_c_fused_multi():
        return paged_direct_matmul_fast_c_multi(x_int8, pages_fast, out_features)

    def run_fp32_resident():
        return torch.matmul(x_fp32, dense_w_fp32)

    # 6. Medición rigurosa con Timer
    t_py = benchmark.Timer(
        stmt="run_pages_python()",
        globals={"run_pages_python": run_pages_python},
        num_threads=2,
    ).blocked_autorange(min_run_time=1.0)

    t_c_single = benchmark.Timer(
        stmt="run_pages_c_single()",
        globals={"run_pages_c_single": run_pages_c_single},
        num_threads=2,
    ).blocked_autorange(min_run_time=1.0)

    t_c_multi = benchmark.Timer(
        stmt="run_pages_c_fused_multi()",
        globals={"run_pages_c_fused_multi": run_pages_c_fused_multi},
        num_threads=2,
    ).blocked_autorange(min_run_time=1.0)

    t_fp32 = benchmark.Timer(
        stmt="run_fp32_resident()",
        globals={"run_fp32_resident": run_fp32_resident},
        num_threads=2,
    ).blocked_autorange(min_run_time=1.0)

    py_us = t_py.median * 1e6
    py_iqr = t_py.iqr * 1e6
    c_single_us = t_c_single.median * 1e6
    c_single_iqr = t_c_single.iqr * 1e6
    c_multi_us = t_c_multi.median * 1e6
    c_multi_iqr = t_c_multi.iqr * 1e6
    fp32_us = t_fp32.median * 1e6
    fp32_iqr = t_fp32.iqr * 1e6

    speedup_multi_vs_py = py_us / c_multi_us
    ratio_multi_vs_fp32 = c_multi_us / fp32_us

    print(f"\n{'Variante':<32}{'Mediana (us)':<16}{'IQR (us)':<14}{'vs Python':<14}{'vs FP32'}")
    print("-" * 85)
    print(f"{'1. Kernel Python (Fast)':<32}{py_us:<16.2f}{py_iqr:<14.2f}{'1.00x':<14}{f'{py_us/fp32_us:.2f}x'}")
    print(f"{'2. Kernel C (Por Página)':<32}{c_single_us:<16.2f}{c_single_iqr:<14.2f}{f'{py_us/c_single_us:.2f}x':<14}{f'{c_single_us/fp32_us:.2f}x'}")
    print(f"{'3. Kernel C Fisionado Multi-Página':<32}{c_multi_us:<16.2f}{c_multi_iqr:<14.2f}{f'{speedup_multi_vs_py:.2f}x':<14}{f'{ratio_multi_vs_fp32:.2f}x'}")
    print(f"{'4. PyTorch FP32 Residente (BLAS)':<32}{fp32_us:<16.2f}{fp32_iqr:<14.2f}{f'{py_us/fp32_us:.2f}x':<14}{'1.00x'}")

    results = {
        "max_diff_single": int(max_diff_single),
        "max_diff_multi": int(max_diff_multi),
        "status": "PASS",
        "python_fast_median_us": py_us,
        "python_fast_iqr_us": py_iqr,
        "c_single_median_us": c_single_us,
        "c_single_iqr_us": c_single_iqr,
        "c_multi_median_us": c_multi_us,
        "c_multi_iqr_us": c_multi_iqr,
        "fp32_resident_median_us": fp32_us,
        "fp32_resident_iqr_us": fp32_iqr,
        "speedup_multi_vs_python": speedup_multi_vs_py,
        "ratio_multi_vs_fp32": ratio_multi_vs_fp32,
    }

    with open("benchmark_kernel_c_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_verification_and_benchmark()
