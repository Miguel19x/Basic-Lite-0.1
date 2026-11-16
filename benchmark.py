import os
import tempfile
import time
import resource
import torch

from config import HybridConfig
from model import TernaryRecursiveDiffusionLM, BitLinear
from generate import generate_block_causal_with_kv_cache


def run_v4_benchmark():
    cfg = HybridConfig()
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)

    model = TernaryRecursiveDiffusionLM(cfg).to(cfg.device)
    if os.path.exists(cfg.checkpoint_path):
        ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        status_str = f"Checkpoint entrenado ({cfg.checkpoint_path})"
    else:
        status_str = "Pesos aleatorios (ejecuta train.py primero para medir sparsity real)"

    total_params = sum(p.numel() for p in model.parameters())
    ternary_params = sum(m.weight.numel() for m in model.modules() if isinstance(m, BitLinear))
    float_params = total_params - ternary_params

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp_path = tmp.name
    bitcos_bytes = model.export_bitcos_archive(tmp_path)
    model.load_bitcos_archive(tmp_path)
    os.remove(tmp_path)

    fp32_total_kib = (total_params * 4) / 1024.0
    bitcos_ternary_kib = bitcos_bytes / 1024.0
    fp32_remaining_kib = (float_params * 4) / 1024.0
    deployable_total_kib = bitcos_ternary_kib + fp32_remaining_kib

    prompt = "nodo: switch\ncarga: pico\nred: doble\nperfil: turbo\n"
    model.prepare_for_eval()

    def _bench_infer(int_mode: bool, add_sub: bool, n_runs: int = 10) -> float:
        cfg.use_integer_inference = int_mode
        cfg.use_add_sub_inference = add_sub
        # 3 pasadas de calentamiento (warmup)
        for _ in range(3):
            generate_block_causal_with_kv_cache(model, prompt, cfg)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            generate_block_causal_with_kv_cache(model, prompt, cfg)
        return ((time.perf_counter() - t0) / n_runs) * 1000.0

    fp32_cached_ms = _bench_infer(int_mode=False, add_sub=False)
    add_sub_ms = _bench_infer(int_mode=False, add_sub=True)
    int8_ms = _bench_infer(int_mode=True, add_sub=False)
    peak_ram_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    print(f"=== Benchmark Riguroso TRDM-v4 [{status_str}] ===")
    print(f"Parámetros totales únicos           : {total_params:,} ({100*ternary_params/total_params:.1f}% ternarios)")
    print(f"Tamaño modelo completo en FP32      : {fp32_total_kib:.1f} KiB")
    print(f"Tamaño desplegable real (Híbrido)   : {deployable_total_kib:.1f} KiB ({fp32_total_kib/deployable_total_kib:.2f}x reducción)")
    print(f"  - Matrices ternarias BITCOSv2     : {bitcos_ternary_kib:.1f} KiB")
    print(f"  - Componentes FP32 (Emb/Norm/MoE) : {fp32_remaining_kib:.1f} KiB")
    print(f"Inferencia FP32 (con W_eff cacheado): {fp32_cached_ms:.1f} ms")
    print(f"Inferencia Suma/Resta (máscaras +/-): {add_sub_ms:.1f} ms")
    print(f"Inferencia INT8 (torch.matmul int32): {int8_ms:.1f} ms")
    print(f"RAM máxima del proceso (RSS)        : {peak_ram_mib:.1f} MiB")


if __name__ == "__main__":
    run_v4_benchmark()
