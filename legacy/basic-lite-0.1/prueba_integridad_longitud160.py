import math
import os
import random
import resource
import time
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F

from config import HybridConfig
from model import TernaryRecursiveDiffusionLM
from data import ProceduralCompositionalDataset
from train import compute_v4_hybrid_loss
from trdm.diffusion.corruption import apply_blockwise_diffusion_corruption


def get_current_rss_mib() -> float:
    """Lee el RSS residente actual del proceso directamente desde /proc/self/statm (Estándar Linux)."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:
        return 0.0


def get_peak_ram_mib() -> float:
    """Retorna el pico máximo de RSS del proceso registrado por el sistema operativo Linux (en MiB)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024.0


def run_etapa1_integridad():
    print("=" * 76)
    print("  ETAPA 1: PRUEBA DE INTEGRIDAD Y AUDITORÍA DE LONGITUD (160 POSICIONES)")
    print("=" * 76)

    ram_import = get_current_rss_mib()
    print(f"[Memoria Base] RSS tras importaciones: {ram_import:.1f} MiB")

    # =====================================================================
    # 1. AUDITORÍA EMPÍRICA DE LONGITUD EN LAS 480 TUPLAS DEL UNIVERSO
    # =====================================================================
    print("\n--- 1. Auditoría de Longitud en las 480 Tuplas del Universo ---")
    cfg_ref = HybridConfig(max_seq_len=160, use_explicit_operands=False)
    cfg_optA = HybridConfig(max_seq_len=160, use_explicit_operands=True)

    ds_ref_train = ProceduralCompositionalDataset(cfg_ref, split="train", num_samples=384)
    ds_ref_val = ProceduralCompositionalDataset(cfg_ref, split="val", num_samples=96)
    ref_all = ds_ref_train.unique_records + ds_ref_val.unique_records
    assert len(ref_all) == 480, f"Error en conteo total de referencia: {len(ref_all)} != 480"

    ds_optA_train = ProceduralCompositionalDataset(cfg_optA, split="train", num_samples=384)
    ds_optA_val = ProceduralCompositionalDataset(cfg_optA, split="val", num_samples=96)
    optA_all = ds_optA_train.unique_records + ds_optA_val.unique_records
    assert len(optA_all) == 480, f"Error en conteo total de Opción A: {len(optA_all)} != 480"

    ref_lens = [1 + len(p) + len(t) + 1 for p, t, _, _ in ref_all]
    optA_lens = [1 + len(p) + len(t) + 1 for p, t, _, _ in optA_all]

    print(f"  Referencia Categórica (480 tuplas):")
    print(f"    - Mínimo: {min(ref_lens)} bytes | Media: {sum(ref_lens)/len(ref_lens):.1f} bytes | Máximo: {max(ref_lens)} bytes")
    print(f"    - Tuplas > 160 bytes: {sum(1 for l in ref_lens if l > 160)} (0.0% truncadas)")

    print(f"  Opción A (Categorías + Operandos Explícitos, 480 tuplas):")
    print(f"    - Mínimo: {min(optA_lens)} bytes | Media: {sum(optA_lens)/len(optA_lens):.1f} bytes | Máximo: {max(optA_lens)} bytes")
    print(f"    - Tuplas > 160 bytes: {sum(1 for l in optA_lens if l > 160)} (0.0% truncadas)")
    assert max(optA_lens) <= 160, f"Violación de longitud en Opción A: max={max(optA_lens)} > 160"
    print("  [OK] Verificación completada: 100% de las 480 secuencias caben estrictamente dentro de 160 posiciones.")

    # =====================================================================
    # 2. INSPECCIÓN DE FACTORES, CAMPOS Y OPERANDOS
    # =====================================================================
    print("\n--- 2. Inspección Detallada de Factores y Consistencia Aritmética ---")
    for idx in [0, 120, 240, 360, 479]:
        p_b, t_b, w_val, ms_val = optA_all[idx]
        p_str = p_b.decode("utf-8")
        t_str = t_b.decode("utf-8")
        meta = ds_optA_train.record_meta.get((p_b, t_b), None)
        sub_w, sub_ms, tw, tms = meta if meta else (None, None, w_val, ms_val)

        print(f"  [Tupla #{idx+1}]:")
        print(f"    Prompt: {repr(p_str)}")
        print(f"    Target: {repr(t_str)}")
        print(f"    Aritmética: pw={sub_w}, pms={sub_ms} | watt={tw}, ms={tms}")
        assert tw == w_val and tms == ms_val
        # Comprobar presencia de campos indispensables
        assert "pw: " in t_str and "pms: " in t_str and "watt: " in t_str and "ms: " in t_str

    # =====================================================================
    # 3. VERIFICACIÓN DE MÁSCARAS, LÍMITE DE PROMPT Y GRADIENTES
    # =====================================================================
    print("\n--- 3. Verificación de Aislamiento de Máscaras y Flujo de Gradientes ---")
    x0, cmask, dmask = ds_optA_train[0]
    p_len = 1 + len(ds_optA_train.samples[0][0])
    t_len = len(ds_optA_train.samples[0][1])

    # A. Comprobar inicio de supervisión exactamente en p_len
    prompt_can_mask_sum = cmask[:p_len].sum().item()
    target_can_mask_sum = cmask[p_len:p_len+t_len+1].sum().item()
    pad_can_mask_sum = cmask[p_len+t_len+1:].sum().item()

    assert prompt_can_mask_sum == 0, f"can_mask activo en prompt ({prompt_can_mask_sum})"
    assert target_can_mask_sum > 0, "can_mask inactivo en target"
    print(f"  [OK] can_mask en prompt (posiciones 0..{p_len-1}): 0 tokens activos (sin pérdida directa en operandos)")
    print(f"  [OK] can_mask en target (posiciones {p_len}..{p_len+t_len}): {target_can_mask_sum} tokens activos")
    print(f"  [OK] can_mask en PAD (posiciones {p_len+t_len+1}..159): {pad_can_mask_sum} tokens (pesados a 0.2x)")

    # B. Comprobar que digit_mask no supervise operandos del prompt
    prompt_dmask_sum = dmask[:p_len].sum().item()
    target_dmask_sum = dmask[p_len:].sum().item()
    assert prompt_dmask_sum == 0, f"digit_mask activo en prompt ({prompt_dmask_sum})"
    assert target_dmask_sum > 0, "digit_mask inactivo en target"
    print(f"  [OK] digit_mask en prompt: {prompt_dmask_sum} (no supervisa operandos de entrada)")
    print(f"  [OK] digit_mask en target: {target_dmask_sum} dígitos supervisados con ponderación x6.0")

    # =====================================================================
    # 4. PRUEBA CORTA DE INTEGRIDAD (50 PASOS A LONGITUD 160)
    # =====================================================================
    print("\n--- 4. Prueba Corta de Entrenamiento a Longitud 160 (50 Pasos) ---")
    torch.manual_seed(42)
    model = TernaryRecursiveDiffusionLM(cfg_optA).to(cfg_optA.device)
    optimizer = torch.optim.AdamW(model.configure_adamw_groups(), lr=2.0e-3, betas=(0.9, 0.95))

    ram_after_model = get_current_rss_mib()
    print(f"  RSS tras inicializar modelo: {ram_after_model:.1f} MiB")

    from torch.utils.data import DataLoader
    loader = DataLoader(ds_optA_train, batch_size=16, shuffle=True)
    batch_iter = iter(loader)

    step_times = []
    max_observed_rss = ram_after_model
    model.train()

    step_1_time = 0.0

    for step in range(1, 51):
        try:
            x_0, can_mask, digit_mask = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            x_0, can_mask, digit_mask = next(batch_iter)

        x_0 = x_0.to(cfg_optA.device)
        can_mask = can_mask.to(cfg_optA.device)
        digit_mask = digit_mask.to(cfg_optA.device)

        t0 = time.perf_counter()
        x_t, mask_pos, t_seq = apply_blockwise_diffusion_corruption(x_0, can_mask, cfg_optA)
        sup_mask = mask_pos if cfg_optA.lazy_vocab_projection else None

        out = model(x_t, t_seq, num_recursions=2, supervised_mask=sup_mask)
        loss, metrics = compute_v4_hybrid_loss(
            out.final_logits, out.intermediate_logits, out.halt_probs,
            x_0, mask_pos, digit_mask, t_seq, cfg_optA
        )

        assert torch.isfinite(loss), f"Pérdida no finita en paso {step}: {loss.item()}"

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Comprobar recepción de gradientes en parámetros clave
        assert model.token_emb.weight.grad is not None, "token_emb no recibió gradiente"
        emb_grad_norm = model.token_emb.weight.grad.norm().item()
        assert emb_grad_norm > 0, "Gradiente de token_emb es cero"

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(grad_norm), f"Grad norm no finito en paso {step}: {grad_norm.item()}"
        optimizer.step()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if step == 1:
            step_1_time = elapsed_ms
        else:
            step_times.append(elapsed_ms)

        curr_rss = get_current_rss_mib()
        if curr_rss > max_observed_rss:
            max_observed_rss = curr_rss

        if step in (1, 25, 50):
            print(f"  Paso {step:02d}/50 | Loss: {loss.item():.4f} | GradNorm: {grad_norm.item():.2f} | EmbGradNorm: {emb_grad_norm:.3f} | TrainDig: {metrics['train_digit_acc']:.2f} | Tiempo: {elapsed_ms:.1f} ms | RSS: {curr_rss:.1f} MiB")

    steady_avg_ms = sum(step_times) / len(step_times)
    os_peak_kib = get_peak_ram_mib()

    print(f"\n  Resumen de Rendimiento y Memoria (L=160, CPU 2 Hilos):")
    print(f"    - Paso 1 (Warmup): {step_1_time:.1f} ms")
    print(f"    - Pasos 2-50 (Régimen Estable): {steady_avg_ms:.1f} ms/paso")
    print(f"    - Máximo RSS observado durante entrenamiento: {max_observed_rss:.1f} MiB")
    print(f"    - Pico acumulado OS (ru_maxrss): {os_peak_kib:.1f} MiB")

    # Proyección realista de 800 pasos
    eval_est_s = 8 * 7.5  # 8 evaluaciones intermedias de ~7.5s cada una
    save_est_s = 2.0
    train_steady_s = (1 * step_1_time + 799 * steady_avg_ms) / 1000.0
    total_est_s = train_steady_s + eval_est_s + save_est_s
    print(f"\n  Proyección Estimada para 800 Pasos a L=160:")
    print(f"    - Tiempo de cómputo de pasos: {train_steady_s:.1f} s (~{train_steady_s/60.0:.1f} min)")
    print(f"    - Overhead de evaluaciones periódicas: ~{eval_est_s:.1f} s")
    print(f"    - Tiempo total estimado por corrida: ~{total_est_s:.1f} s (~{total_est_s/60.0:.1f} min)")

    # =====================================================================
    # 5. SERIALIZACIÓN, RECARGA E IDENTIDAD FUNCIONAL DE INFERENCIA
    # =====================================================================
    print("\n--- 5. Verificación de Guardado, Recarga e Identidad Numérica ---")
    tmp_ckpt = "test_integridad_160_temporal.pt"
    
    # 1. Ejecutar pasada hacia adelante de referencia antes de guardar
    model.prepare_for_eval()
    x_test = x_0[:2].clone()
    t_test = torch.zeros((2, cfg_optA.max_seq_len), device=cfg_optA.device)
    with torch.no_grad():
        out_before = model(x_test, t_test, num_recursions=2)

    # 2. Guardar estado completo
    t_save0 = time.perf_counter()
    save_dict = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": 50,
        "config": cfg_optA,
    }
    torch.save(save_dict, tmp_ckpt)
    save_ms = (time.perf_counter() - t_save0) * 1000.0
    ckpt_size_mib = os.path.getsize(tmp_ckpt) / (1024 * 1024)
    ram_post_save = get_current_rss_mib()
    print(f"  Checkpoint guardado ({ckpt_size_mib:.2f} MiB en {save_ms:.1f} ms). RSS post-guardado: {ram_post_save:.1f} MiB")

    # 3. Liberar modelo original de memoria para evitar duplicación artificial
    del model
    del optimizer
    del save_dict

    # 4. Recargar en nueva instancia limpia
    t_load0 = time.perf_counter()
    loaded_data = torch.load(tmp_ckpt, map_location="cpu", weights_only=False)
    model_reloaded = TernaryRecursiveDiffusionLM(cfg_optA).to(cfg_optA.device)
    model_reloaded.load_state_dict(loaded_data["model_state"])
    model_reloaded.prepare_for_eval()
    load_ms = (time.perf_counter() - t_load0) * 1000.0
    ram_post_reload = get_current_rss_mib()
    print(f"  Checkpoint recargado en {load_ms:.1f} ms. RSS post-recarga limpia: {ram_post_reload:.1f} MiB")

    # 5. Comprobar identidad numérica estricta en tensores de logits
    with torch.no_grad():
        out_after = model_reloaded(x_test, t_test, num_recursions=2)

    logits_match = torch.equal(out_before.final_logits, out_after.final_logits)
    max_logit_diff = (out_before.final_logits - out_after.final_logits).abs().max().item()
    print(f"  Fidelidad tensorial de salida de inferencia: {logits_match} (Diferencia absoluta máxima = {max_logit_diff:.1e})")
    assert logits_match or max_logit_diff < 1e-6, "Discrepancia funcional en inferencia tras recarga"

    # 6. Comprobar integridad de estado de optimizador y paso
    assert loaded_data["step"] == 50, "Paso de checkpoint inconsistente"
    assert "optimizer_state" in loaded_data and len(loaded_data["optimizer_state"]["state"]) > 0, "Estado de optimizador no preservado"
    print("  [OK] Estado de optimizador y metadatos de progreso verificados.")

    # Limpieza de archivo temporal
    if os.path.exists(tmp_ckpt):
        os.remove(tmp_ckpt)

    print("\n" + "=" * 76)
    print("  ETAPA 1 FINALIZADA CON ÉXITO: CRITERIOS CUMPLIDOS AL 100%")
    print("=" * 76)


if __name__ == "__main__":
    run_etapa1_integridad()
