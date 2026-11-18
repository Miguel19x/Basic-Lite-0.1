
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
