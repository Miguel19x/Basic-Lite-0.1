import os
import sys
import ctypes
import subprocess
from pathlib import Path
import torch

_LIB_INSTANCE = None

C_SOURCE = """
#include <stdint.h>
#include <omp.h>

void ternary_addsub_cpu_kernel(
    const int8_t* x,
    const int8_t* w,
    int32_t* out,
    int64_t M,
    int64_t K,
    int64_t N
) {
    #pragma omp parallel for collapse(2)
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            int32_t acc = 0;
            const int8_t* x_row = x + m * K;
            const int8_t* w_row = w + n * K;
            #pragma omp simd reduction(+:acc)
            for (int64_t k = 0; k < K; ++k) {
                const int8_t code = w_row[k];
                if (code == 1) {
                    acc += (int32_t)x_row[k];
                } else if (code == -1) {
                    acc -= (int32_t)x_row[k];
                }
            }
            out[m * N + n] = acc;
        }
    }
}
"""


def compile_and_load_ternary_kernel() -> ctypes.CDLL:
    global _LIB_INSTANCE
    if _LIB_INSTANCE is not None:
        return _LIB_INSTANCE

    curr_dir = Path(__file__).parent.resolve()
    c_path = curr_dir / "ternary_kernel.c"
    so_path = curr_dir / "libternary.so"

    if not so_path.exists() or (c_path.exists() and c_path.stat().st_mtime > so_path.stat().st_mtime):
        with open(c_path, "w") as f:
            f.write(C_SOURCE)
        cmd = [
            "gcc", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC",
            str(c_path), "-o", str(so_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Error compilando kernel ternario C:\n{res.stderr}")

    lib = ctypes.CDLL(str(so_path))
    lib.ternary_addsub_cpu_kernel.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
    ]
    lib.ternary_addsub_cpu_kernel.restype = None
    _LIB_INSTANCE = lib
    return lib


def ternary_addsub_matmul(x_int8: torch.Tensor, w_codes: torch.Tensor) -> torch.Tensor:
    """
    Multiplicación de matrices ternarias sin multiplicaciones:
    x_int8: [..., K] tensor int8
    w_codes: [N, K] tensor int8 con valores {-1, 0, +1}
    Retorna: [..., N] tensor int32 calculado exclusivamente por sumas y restas con OpenMP/AVX2.
    """
    orig_shape = x_int8.shape
    x_2d = x_int8.view(-1, orig_shape[-1]).contiguous()
    w_2d = w_codes.contiguous()

    M, K = x_2d.shape
    N, K_w = w_2d.shape
    assert K == K_w, f"Dimensión interna K incompatible: {K} vs {K_w}"

    out_2d = torch.empty((M, N), dtype=torch.int32, device=x_int8.device)

    lib = compile_and_load_ternary_kernel()
    lib.ternary_addsub_cpu_kernel(
        ctypes.cast(x_2d.data_ptr(), ctypes.c_char_p),
        ctypes.cast(w_2d.data_ptr(), ctypes.c_char_p),
        ctypes.c_void_p(out_2d.data_ptr()),
        M, K, N
    )

    return out_2d.view(*orig_shape[:-1], N)


if __name__ == "__main__":
    M, K, N = 16, 192, 384
    x = torch.randint(-127, 127, (M, K), dtype=torch.int8)
    w = torch.randint(-1, 2, (N, K), dtype=torch.int8)
    out = ternary_addsub_matmul(x, w)
    ref = torch.matmul(x.to(torch.int32), w.t().to(torch.int32))
    assert (out - ref).abs().max().item() == 0, "Error en kernel C"
    print("Kernel C AVX2/OpenMP verificado y funcional al 100%.")
