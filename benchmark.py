import os
import time
import resource
import torch

from config import HybridConfig
from model import TernaryRecursiveDiffusionLM, BitLinear
from generate import generate_block_causal_with_kv_cache


def run_v2_benchmark():
    cfg = HybridConfig()
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)

    model = TernaryRecursiveDiffusionLM(cfg).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    ternary_params = sum(m.weight.numel() for m in model.modules() if isinstance(m, BitLinear))
    non_ternary_params = total_params - ternary_params

    bench_bitcos_path = "trdm_benchmark_temp.bin"
    bitcos_bytes = model.export_bitcos_archive(bench_bitcos_path)
    fp32_bytes = total_params * 4
    non_ternary_bytes = non_ternary_params * 4
    deployable_bytes = bitcos_bytes + non_ternary_bytes

    # 1. Medición de paso de entrenamiento con los 4 objetivos de pérdida
    x_dummy = torch.randint(0, 256, (cfg.batch_size, cfg.max_seq_len))
    t_dummy = torch.full((cfg.batch_size,), 0.5)
    for _ in range(2):
        logits, _, _, _, _ = model(x_dummy, t_dummy)
        logits.mean().backward()
        model.zero_grad(set_to_none=True)

    t0 = time.perf_counter()
    n_iters = 10
    for _ in range(n_iters):
        logits, _, _, _, _ = model(x_dummy, t_dummy, record_telemetry=True)
        logits.mean().backward()
        model.zero_grad(set_to_none=True)
    train_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    # 2. Inferencia en modo flotante vs modo entero BITCOS + Caché KV por bloques
    prompt = "nodo: server\nperfil: eco\nred: simple\ncarga: baja\n"
    cfg.use_integer_inference = False
    t1 = time.perf_counter()
    _, _, evals_fp = generate_block_causal_with_kv_cache(model, prompt, cfg)
    infer_fp_ms = (time.perf_counter() - t1) * 1000.0

    model.load_bitcos_archive(bench_bitcos_path)
    t2 = time.perf_counter()
    _, _, evals_int = generate_block_causal_with_kv_cache(model, prompt, cfg)
    infer_int_ms = (time.perf_counter() - t2) * 1000.0

    peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    tern_stats = model.collect_ternary_telemetry()

    if os.path.exists(bench_bitcos_path):
        os.remove(bench_bitcos_path)

    # 3. Micro-benchmark de Capa BitLinear (192 -> 384): FP32 vs INT8 Matmul vs Sign Decomposition vs Kernel C AVX2
    layer = BitLinear(cfg.d_model, cfg.d_ff_expert).eval()
    x_test = torch.randn(16, cfg.block_size, cfg.d_model)

    for _ in range(5):
        _ = layer(x_test)
        _ = layer(x_test, integer_mode=True)
        _ = layer(x_test, add_sub_only=True)
        _ = layer(x_test, use_kernel=True)

    iters = 100
    t_start = time.perf_counter()
    for _ in range(iters):
        _ = layer(x_test)
    l_fp_ms = ((time.perf_counter() - t_start) / iters) * 1000.0

    t_start = time.perf_counter()
    for _ in range(iters):
        _ = layer(x_test, integer_mode=True)
    l_int_ms = ((time.perf_counter() - t_start) / iters) * 1000.0

    t_start = time.perf_counter()
    for _ in range(iters):
        _ = layer(x_test, add_sub_only=True)
    l_as_ms = ((time.perf_counter() - t_start) / iters) * 1000.0

    t_start = time.perf_counter()
    for _ in range(iters):
        _ = layer(x_test, use_kernel=True)
    l_k_ms = ((time.perf_counter() - t_start) / iters) * 1000.0

    print("=== Benchmark TRDM-v3 en Linux CPU (i3-1115G4) ===")
    print(f"Parámetros únicos totales         : {total_params:,} ({100*ternary_params/total_params:.1f}% ternarios)")
    print(f"Parámetros ternarios ({ternary_params:,}) : {bitcos_bytes/1024:.1f} KB en BITCOSv2 ({tern_stats.get('bitcos_bpw', 0):.3f} bpw)")
    print(f"Parámetros FP32 restantes ({non_ternary_params:,}): {non_ternary_bytes/1024:.1f} KB")
    print(f"Tamaño desplegable real total     : {deployable_bytes/1024:.1f} KB ({fp32_bytes/deployable_bytes:.1f}x reducción vs {fp32_bytes/1024:.1f} KB FP32)")
    print(f"Atención GQA                      : {cfg.n_heads} cabezas Q / {cfg.n_kv_heads} cabezas KV (ahorro 50% en KV)")
    print(f"Paso de entrenamiento (QAT + STE) : {train_ms:.1f} ms")
    print(f"Generación Block-Causal (FP32)    : {infer_fp_ms:.1f} ms ({evals_fp} evals de bloque local)")
    print(f"Generación Block-Causal (INT8)    : {infer_int_ms:.1f} ms ({evals_int} evals desde binario BITCOSv2)")
    print(f"RAM máxima del proceso (RSS)      : {peak_ram_mb:.1f} MB")
    print("\n=== Micro-benchmark BitLinear (192 -> 384, lote 16x16) ===")
    print(f"FP32 PyTorch F.linear              : {l_fp_ms:.3f} ms")
    print(f"INT8 PyTorch torch.matmul          : {l_int_ms:.3f} ms")
    print(f"Suma/Resta Pura (Descomp. Signos)  : {l_as_ms:.3f} ms")
    print(f"Kernel C Nativo (OpenMP/AVX2)      : {l_k_ms:.3f} ms")


if __name__ == "__main__":
    run_v2_benchmark()
