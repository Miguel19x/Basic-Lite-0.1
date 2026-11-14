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

    bitcos_bytes = model.export_bitcos_archive(cfg.bitcos_export_path)
    fp32_bytes = total_params * 4

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
    prompt = "Pregunta: Cual es la capital de Alemania?\nRespuesta: "
    cfg.use_integer_inference = False
    t1 = time.perf_counter()
    _, _, evals_fp = generate_block_causal_with_kv_cache(model, prompt, cfg)
    infer_fp_ms = (time.perf_counter() - t1) * 1000.0

    model.load_bitcos_archive(cfg.bitcos_export_path)
    t2 = time.perf_counter()
    _, _, evals_int = generate_block_causal_with_kv_cache(model, prompt, cfg)
    infer_int_ms = (time.perf_counter() - t2) * 1000.0

    peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    tern_stats = model.collect_ternary_telemetry()

    if os.path.exists(cfg.bitcos_export_path):
        os.remove(cfg.bitcos_export_path)

    print("=== Benchmark TRDM-v3 en Linux CPU (i3-1115G4) ===")
    print(f"Parámetros únicos totales         : {total_params:,} ({100*ternary_params/total_params:.1f}% ternarios)")
    print(f"Atención GQA                      : {cfg.n_heads} cabezas Q / {cfg.n_kv_heads} cabezas KV (ahorro 50% en KV)")
    print(f"Peso en FP32 vs Archivo BITCOSv2  : {fp32_bytes/1024:.1f} KB -> {bitcos_bytes/1024:.1f} KB ({tern_stats.get('bitcos_bpw', 0):.3f} bpw)")
    print(f"Paso de entrenamiento (QAT + STE) : {train_ms:.1f} ms")
    print(f"Generación Block-Causal (FP32)    : {infer_fp_ms:.1f} ms ({evals_fp} evals de bloque local)")
    print(f"Generación Block-Causal (INT8)    : {infer_int_ms:.1f} ms ({evals_int} evals desde binario BITCOSv2)")
    print(f"RAM máxima del proceso (RSS)      : {peak_ram_mb:.1f} MB")


if __name__ == "__main__":
    run_v2_benchmark()
