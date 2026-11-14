import os
import sys
import torch
from torch.utils.cpp_extension import load_inline

cpp_source = """
#include <torch/extension.h>
#include <cstdint>

torch::Tensor ternary_addsub_matmul_cpu(
    torch::Tensor x_int8,   // [M, K] int8
    torch::Tensor w_codes   // [N, K] int8 con valores {-1, 0, +1}
) {
    auto x_c = x_int8.contiguous();
    auto w_c = w_codes.contiguous();
    const int64_t M = x_c.size(0);
    const int64_t K = x_c.size(1);
    const int64_t N = w_c.size(0);

    auto out = torch::zeros({M, N}, torch::dtype(torch::kInt32));
    const int8_t* x_ptr = x_c.data_ptr<int8_t>();
    const int8_t* w_ptr = w_c.data_ptr<int8_t>();
    int32_t* out_ptr = out.data_ptr<int32_t>();

    #pragma omp parallel for collapse(2)
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            int32_t acc = 0;
            const int8_t* x_row = x_ptr + m * K;
            const int8_t* w_row = w_ptr + n * K;
            for (int64_t k = 0; k < K; ++k) {
                const int8_t code = w_row[k];
                if (code == 1) {
                    acc += static_cast<int32_t>(x_row[k]);
                } else if (code == -1) {
                    acc -= static_cast<int32_t>(x_row[k]);
                }
            }
            out_ptr[m * N + n] = acc;
        }
    }
    return out;
}
"""

def load_ternary_cpu_kernel():
    """
    Compila al vuelo con load_inline un kernel en C++ con OpenMP/AVX2
    que ejecuta la capa ternaria únicamente sumando y restando activaciones enteras.
    """
    bin_dir = os.path.dirname(sys.executable)
    if bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"

    return load_inline(
        name="ternary_cpu_ext",
        cpp_sources=cpp_source,
        functions=["ternary_addsub_matmul_cpu"],
        extra_cflags=["-O3", "-march=native", "-fopenmp"],
        extra_ldflags=["-fopenmp"],
        verbose=False,
    )
