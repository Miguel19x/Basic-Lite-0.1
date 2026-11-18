# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

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


def build_field_ids(prompt_b: bytes, target_b: bytes, max_seq_len: int, use_scratchpad: bool) -> torch.Tensor:
    """
    Construye de forma determinista y disjunta el tensor de IDs de campo:
      0: Prompt o PAD (no supervisado)
      1: Subtotal Pw ('pw: {sub_w}\\n')
      2: Subtotal Pms ('pms: {sub_ms}\\n')
      3: Respuesta Final ('watt: {watt}\\nms: {ms}\\n')
      4: Delimitador EOS ('<EOS>')
    """
    field_ids = torch.zeros(max_seq_len, dtype=torch.long)
    prompt_len = 1 + len(prompt_b)

    if use_scratchpad:
        lines = target_b.split(b"\n")
        assert len(lines) >= 5, f"Formato inesperado de target scratchpad: {lines}"
        len_pw = len(lines[0]) + 1
        len_pms = len(lines[1]) + 1
        len_final = len(lines[2]) + 1 + len(lines[3]) + 1

        p_start = prompt_len
        pw_end = p_start + len_pw
        pms_end = pw_end + len_pms
        final_end = pms_end + len_final

        field_ids[p_start:pw_end] = 1        # Campo 1: Pw
        field_ids[pw_end:pms_end] = 2        # Campo 2: Pms
        field_ids[pms_end:final_end] = 3     # Campo 3: Watt y MS
        if final_end < max_seq_len:
            field_ids[final_end] = 4         # Campo 4: EOS
    else:
        target_end = min(max_seq_len, prompt_len + len(target_b))
        field_ids[prompt_len:target_end] = 3
        if target_end < max_seq_len:
            field_ids[target_end] = 4

    return field_ids


def compute_v4_decoupled_loss(
    final_logits: torch.Tensor,
    intermediate_logits: List[torch.Tensor],
    halt_probs_list: List[torch.Tensor],
    x_0: torch.Tensor,
    mask_positions: torch.Tensor,
    digit_mask: torch.Tensor,
    t_noise: torch.Tensor,
    cfg: HybridConfig,
    field_ids: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Pérdida desacoplada y normalizada por campo semántico (Opción B).
    """
    assert final_logits.ndim == 2, "La prueba de integridad evalúa la ruta nativa Lazy Logits 2D."

    x_0_sup = x_0[mask_positions]
    dig_sup = digit_mask[mask_positions]
    field_sup = field_ids[mask_positions]

    pos_importance = torch.full_like(x_0_sup, cfg.template_token_weight, dtype=final_logits.dtype)
    pos_importance = torch.where(
        x_0_sup == cfg.pad_token_id,
        torch.full_like(pos_importance, cfg.pad_token_weight),
        pos_importance,
    )
    pos_importance = torch.where(
        dig_sup,
        torch.full_like(pos_importance, cfg.digit_token_weight),
        pos_importance,
    )

    if t_noise.dim() == 1:
        inv_t_full = (1.0 / t_noise.clamp(min=0.10)).unsqueeze(1).expand_as(x_0)
    else:
        inv_t_full = (1.0 / t_noise.clamp(min=0.10)).expand_as(x_0)
    inv_t_sup = inv_t_full[mask_positions]

    def _field_weighted_ce(logits_sup: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        token_loss = F.cross_entropy(logits_sup, x_0_sup, reduction="none")
        w_loss = token_loss * pos_importance * inv_t_sup

        # Máscaras disjuntas por campo
        m_pw = (field_sup == 1)
        m_pms = (field_sup == 2)
        m_fin = (field_sup == 3)
        m_eos = (field_sup == 4)

        loss_pw = (w_loss[m_pw].sum() / pos_importance[m_pw].sum().clamp(min=1.0)) if m_pw.any() else torch.tensor(0.0, device=logits_sup.device)
        loss_pms = (w_loss[m_pms].sum() / pos_importance[m_pms].sum().clamp(min=1.0)) if m_pms.any() else torch.tensor(0.0, device=logits_sup.device)
        loss_fin = (w_loss[m_fin].sum() / pos_importance[m_fin].sum().clamp(min=1.0)) if m_fin.any() else torch.tensor(0.0, device=logits_sup.device)
        loss_eos = (w_loss[m_eos].sum() / pos_importance[m_eos].sum().clamp(min=1.0)) if m_eos.any() else torch.tensor(0.0, device=logits_sup.device)

        w_pw = getattr(cfg, "loss_weight_pw", 1.0)
        w_pms = getattr(cfg, "loss_weight_pms", 1.0)
        w_fin = getattr(cfg, "loss_weight_final", 1.0)
        w_eos = getattr(cfg, "loss_weight_eos", 0.2)

        num = (
            w_pw * loss_pw * float(m_pw.any())
            + w_pms * loss_pms * float(m_pms.any())
            + w_fin * loss_fin * float(m_fin.any())
            + w_eos * loss_eos * float(m_eos.any())
        )
        den = (
            w_pw * float(m_pw.any())
            + w_pms * float(m_pms.any())
            + w_fin * float(m_fin.any())
            + w_eos * float(m_eos.any())
        )
        comb_loss = num / max(den, 1e-6)

        metrics_dict = {
            "loss_pw": float(loss_pw.item()) if m_pw.any() else 0.0,
            "loss_pms": float(loss_pms.item()) if m_pms.any() else 0.0,
            "loss_final_field": float(loss_fin.item()) if m_fin.any() else 0.0,
            "loss_eos": float(loss_eos.item()) if m_eos.any() else 0.0,
        }
        return comb_loss, metrics_dict

    loss_final, field_metrics = _field_weighted_ce(final_logits)

    loss_inter = torch.tensor(0.0, device=x_0.device)
    if len(intermediate_logits) > 1 and cfg.intermediate_loss_weight > 0:
        inter_terms = [_field_weighted_ce(lg)[0] for lg in intermediate_logits[:-1]]
        loss_inter = sum(inter_terms) / len(inter_terms)

    loss_cons = torch.tensor(0.0, device=x_0.device)
    if len(intermediate_logits) > 1 and cfg.consistency_loss_weight > 0:
        norm_denom = pos_importance.sum().clamp(min=1.0)
        teacher_probs = F.softmax(final_logits.detach(), dim=-1)
        kl_terms = []
        for lg in intermediate_logits[:-1]:
            student_log_probs = F.log_softmax(lg, dim=-1)
            kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
            kl_terms.append((kl * pos_importance).sum() / norm_denom)
        loss_cons = sum(kl_terms) / len(kl_terms)

    loss_ponder = torch.tensor(0.0, device=x_0.device)
    if halt_probs_list and cfg.ponder_loss_weight > 0:
        ponder_terms = []
        for lg, h_prob in zip(intermediate_logits, halt_probs_list):
            h_sup = h_prob.squeeze(-1)[mask_positions]
            is_correct = (lg.detach().argmax(dim=-1) == x_0_sup).float()
            halt_target = 0.1 + 0.9 * is_correct
            ponder_terms.append(F.mse_loss(h_sup, halt_target))
        loss_ponder = sum(ponder_terms) / len(ponder_terms)

    total_loss = (
        loss_final
        + cfg.intermediate_loss_weight * loss_inter
        + cfg.consistency_loss_weight * loss_cons
        + cfg.ponder_loss_weight * loss_ponder
    )

    with torch.no_grad():
        preds_sup = final_logits.argmax(dim=-1)
        correct_sup = (preds_sup == x_0_sup)
        dig_correct = (correct_sup & dig_sup).float().sum()
        dig_total = dig_sup.float().sum().clamp(min=1.0)
        all_correct = correct_sup.float().sum()
        all_total = float(max(1, x_0_sup.numel()))

        # Precisión de dígitos desagregada por campo
        m_pw_dig = dig_sup & (field_sup == 1)
        m_pms_dig = dig_sup & (field_sup == 2)
        m_fin_dig = dig_sup & (field_sup == 3)

        acc_pw = (correct_sup & m_pw_dig).float().sum() / m_pw_dig.float().sum().clamp(min=1.0) if m_pw_dig.any() else torch.tensor(0.0)
        acc_pms = (correct_sup & m_pms_dig).float().sum() / m_pms_dig.float().sum().clamp(min=1.0) if m_pms_dig.any() else torch.tensor(0.0)
        acc_fin = (correct_sup & m_fin_dig).float().sum() / m_fin_dig.float().sum().clamp(min=1.0) if m_fin_dig.any() else torch.tensor(0.0)

    metrics = {
        "loss_total": float(total_loss.item()),
        "train_all_acc": float((all_correct / all_total).item()),
        "train_digit_acc": float((dig_correct / dig_total).item()),
        "acc_pw": float(acc_pw.item()),
        "acc_pms": float(acc_pms.item()),
        "acc_final": float(acc_fin.item()),
    }
    metrics.update(field_metrics)
    return total_loss, metrics


def run_prueba_integridad_opcionB():
    print("=" * 78)
    print("  ETAPA 1: PRUEBA DE INTEGRIDAD Y AUDITORÍA DE PARTICIÓN - OPCIÓN B")
    print("=" * 78)

    ram_start = get_current_rss_mib()
    print(f"[Memoria Base] RSS inicial: {ram_start:.1f} MiB")

    cfg = HybridConfig(
        max_seq_len=160,
        use_scratchpad=True,
        use_explicit_operands=True,
        cpu_threads=2,
    )
    # Atributos de Opción B
    cfg.use_decoupled_field_loss = True
    cfg.loss_weight_pw = 1.0
    cfg.loss_weight_pms = 1.0
    cfg.loss_weight_final = 1.0
    cfg.loss_weight_eos = 0.2

    # =====================================================================
    # 1. AUDITORÍA EXHAUSTIVA DE PARTICIÓN DE CAMPOS EN LAS 480 TUPLAS
    # =====================================================================
    print("\n--- 1. Auditoría de Partición Disjunta en las 480 Tuplas de data.py ---")
    ds_train = ProceduralCompositionalDataset(cfg, split="train", num_samples=384)
    ds_val = ProceduralCompositionalDataset(cfg, split="val", num_samples=96)
    all_records = ds_train.unique_records + ds_val.unique_records
    assert len(all_records) == 480, f"Error: {len(all_records)} tuplas != 480"

    overlap_count = 0
    orphan_count = 0
    prompt_pollution_count = 0
    digit_mismatch_count = 0

    for idx, (p_b, t_b, w_val, ms_val) in enumerate(all_records):
        f_ids = build_field_ids(p_b, t_b, cfg.max_seq_len, cfg.use_scratchpad)
        p_len = 1 + len(p_b)
        t_len = len(t_b)
        eos_pos = p_len + t_len

        # 1. Prompt debe ser estrictamente 0
        if (f_ids[:p_len] != 0).any():
            prompt_pollution_count += 1

        # 2. PAD debe ser estrictamente 0
        if eos_pos + 1 < cfg.max_seq_len:
            if (f_ids[eos_pos + 1:] != 0).any():
                orphan_count += 1

        # 3. Delimitador EOS debe ser exactamente 4
        if eos_pos < cfg.max_seq_len:
            assert f_ids[eos_pos] == 4, f"EOS en posición incorrecta en tupla #{idx}"

        # 4. Región de target debe estar compuesta exactamente por {1, 2, 3}
        target_fids = f_ids[p_len:eos_pos]
        unique_target_fids = set(target_fids.tolist())
        if not unique_target_fids.issubset({1, 2, 3}):
            orphan_count += 1

        # 5. Longitudes de campos deben sumar exactamente t_len
        c1_len = (target_fids == 1).sum().item()
        c2_len = (target_fids == 2).sum().item()
        c3_len = (target_fids == 3).sum().item()
        if c1_len + c2_len + c3_len != t_len:
            overlap_count += 1

        # 6. Comprobar que los dígitos en cada campo coinciden con los valores numéricos
        t_str = t_b.decode("utf-8")
        meta = ds_train.record_meta.get((p_b, t_b), None)
        sub_w, sub_ms, tw, tms = meta if meta else (None, None, w_val, ms_val)

        # Extraer dígitos marcados en el tensor completo
        seq = [cfg.bos_token_id] + list(p_b) + list(t_b) + [cfg.eos_token_id]
        seq = seq[:cfg.max_seq_len] + [cfg.pad_token_id] * max(0, cfg.max_seq_len - len(seq))
        seq_t = torch.tensor(seq, dtype=torch.long)

        # Dígitos en campo 1
        d_pw_bytes = bytes(seq_t[(f_ids == 1) & (seq_t >= 48) & (seq_t <= 57)].tolist()).decode("utf-8")
        d_pms_bytes = bytes(seq_t[(f_ids == 2) & (seq_t >= 48) & (seq_t <= 57)].tolist()).decode("utf-8")

        if int(d_pw_bytes) != sub_w or int(d_pms_bytes) != sub_ms:
            digit_mismatch_count += 1

    assert prompt_pollution_count == 0, f"Contaminación en prompt: {prompt_pollution_count}"
    assert orphan_count == 0, f"Tokens huérfanos o sin asignar: {orphan_count}"
    assert overlap_count == 0, f"Solapamientos de campo: {overlap_count}"
    assert digit_mismatch_count == 0, f"Discrepancia en dígitos extraídos: {digit_mismatch_count}"

    print(f"  [OK] 480 tuplas auditadas:")
    print(f"    - Contaminación en Prompt (0..p_len-1): 0 tokens (0.0%)")
    print(f"    - Tokens huérfanos en target: 0 tokens (0.0%)")
    print(f"    - Solapamientos entre campos (Pw, Pms, Final): 0 tokens (0.0%)")
    print(f"    - Consistencia de dígitos extraídos vs metadatos: 480/480 tuplas exactas (100.0%)")

    # Muestra representativa de partición
    p_b0, t_b0, _, _ = all_records[0]
    f_ids0 = build_field_ids(p_b0, t_b0, cfg.max_seq_len, cfg.use_scratchpad)
    p_len0 = 1 + len(p_b0)
    print(f"\n  Muestra de Tupla #1 (Partición en bytes):")
    print(f"    - Prompt: {p_len0} bytes (Field 0)")
    print(f"    - Campo Pw: {(f_ids0 == 1).sum().item()} bytes -> {repr(bytes(list(p_b0 + t_b0)[p_len0-1:p_len0-1+(f_ids0 == 1).sum().item()]).decode('utf-8'))}")
    pms_start = p_len0 + (f_ids0 == 1).sum().item()
    print(f"    - Campo Pms: {(f_ids0 == 2).sum().item()} bytes -> {repr(bytes(list(p_b0 + t_b0)[pms_start-1:pms_start-1+(f_ids0 == 2).sum().item()]).decode('utf-8'))}")
    fin_start = pms_start + (f_ids0 == 2).sum().item()
    print(f"    - Campo Final: {(f_ids0 == 3).sum().item()} bytes -> {repr(bytes(list(p_b0 + t_b0)[fin_start-1:fin_start-1+(f_ids0 == 3).sum().item()]).decode('utf-8'))}")
    print(f"    - Delimitador EOS: posición {fin_start-1+(f_ids0 == 3).sum().item()} (Field 4)")

    # =====================================================================
    # 2. PRUEBA DE OPTIMIZACIÓN Y FLUJO DE GRADIENTES (50 PASOS)
    # =====================================================================
    print("\n--- 2. Prueba Corta de Optimización con Pérdidas Desacopladas (50 Pasos) ---")
    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(cfg.cpu_threads)

    model = TernaryRecursiveDiffusionLM(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.configure_adamw_groups(), lr=cfg.learning_rate, betas=cfg.betas)

    ram_after_model = get_current_rss_mib()
    print(f"  RSS tras instanciar modelo: {ram_after_model:.1f} MiB")

    # DataLoader con soporte para field_ids
    from torch.utils.data import DataLoader

    class DecoupledDataset(torch.utils.data.Dataset):
        def __init__(self, base_ds: ProceduralCompositionalDataset):
            self.base_ds = base_ds

        def __len__(self):
            return len(self.base_ds)

        def __getitem__(self, idx):
            x_0, can_mask, digit_mask = self.base_ds[idx]
            p_b, t_b, _, _ = self.base_ds.samples[idx]
            f_ids = build_field_ids(p_b, t_b, self.base_ds.cfg.max_seq_len, self.base_ds.cfg.use_scratchpad)
            return x_0, can_mask, digit_mask, f_ids

    loader = DataLoader(DecoupledDataset(ds_train), batch_size=16, shuffle=True)
    batch_iter = iter(loader)

    step_times = []
    max_observed_rss = ram_after_model
    model.train()
    step_1_time = 0.0

    for step in range(1, 51):
        try:
            x_0, can_mask, digit_mask, field_ids = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            x_0, can_mask, digit_mask, field_ids = next(batch_iter)

        x_0 = x_0.to(cfg.device)
        can_mask = can_mask.to(cfg.device)
        digit_mask = digit_mask.to(cfg.device)
        field_ids = field_ids.to(cfg.device)

        t0 = time.perf_counter()
        x_t, mask_pos, t_seq = apply_blockwise_diffusion_corruption(x_0, can_mask, cfg)
        sup_mask = mask_pos if cfg.lazy_vocab_projection else None

        out = model(x_t, t_seq, num_recursions=2, supervised_mask=sup_mask)
        loss, metrics = compute_v4_decoupled_loss(
            out.final_logits, out.intermediate_logits, out.halt_probs,
            x_0, mask_pos, digit_mask, t_seq, cfg, field_ids
        )

        assert torch.isfinite(loss), f"Pérdida no finita en paso {step}: {loss.item()}"

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Comprobar gradientes en parámetros clave
        assert model.token_emb.weight.grad is not None, "token_emb no recibió gradiente"
        emb_grad_norm = model.token_emb.weight.grad.norm().item()
        assert emb_grad_norm > 0, "Gradiente de token_emb es cero"

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
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

        if step in (1, 10, 25, 50):
            print(
                f"  Paso {step:02d}/50 | LossTot: {loss.item():.3f} [Pw: {metrics['loss_pw']:.2f}, Pms: {metrics['loss_pms']:.2f}, Fin: {metrics['loss_final_field']:.2f}] | "
                f"DigAcc: {metrics['train_digit_acc']:.2f} (Pw: {metrics['acc_pw']:.2f}, Pms: {metrics['acc_pms']:.2f}, Fin: {metrics['acc_final']:.2f}) | "
                f"GradNorm: {grad_norm.item():.2f} | Tiempo: {elapsed_ms:.1f} ms | RSS: {curr_rss:.1f} MiB"
            )

    steady_avg_ms = sum(step_times) / len(step_times)
    os_peak_mib = get_peak_ram_mib()

    print(f"\n  Resumen de Rendimiento y Memoria (Opción B, L=160, CPU 2 Hilos):")
    print(f"    - Paso 1 (Warmup): {step_1_time:.1f} ms")
    print(f"    - Pasos 2-50 (Régimen Estable): {steady_avg_ms:.1f} ms/paso")
    print(f"    - Máximo RSS observado durante optimización: {max_observed_rss:.1f} MiB")
    print(f"    - Pico acumulado del proceso OS (ru_maxrss): {os_peak_mib:.1f} MiB")

    # =====================================================================
    # 3. SERIALIZACIÓN, RECARGA E IDENTIDAD FUNCIONAL DE INFERENCIA
    # =====================================================================
    print("\n--- 3. Verificación de Guardado, Recarga e Identidad Numérica ---")
    tmp_ckpt = "test_optB_temporal.pt"

    model.prepare_for_eval()
    x_test = x_0[:2].clone()
    t_test = torch.zeros((2, cfg.max_seq_len), device=cfg.device)
    with torch.no_grad():
        out_before = model(x_test, t_test, num_recursions=2)

    t_save0 = time.perf_counter()
    save_dict = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": 50,
        "config": cfg,
    }
    torch.save(save_dict, tmp_ckpt)
    save_ms = (time.perf_counter() - t_save0) * 1000.0
    ckpt_size_mib = os.path.getsize(tmp_ckpt) / (1024 * 1024)
    ram_post_save = get_current_rss_mib()
    print(f"  Checkpoint guardado ({ckpt_size_mib:.2f} MiB en {save_ms:.1f} ms). RSS post-guardado: {ram_post_save:.1f} MiB")

    del model
    del optimizer
    del save_dict

    t_load0 = time.perf_counter()
    loaded_data = torch.load(tmp_ckpt, map_location="cpu", weights_only=False)
    model_reloaded = TernaryRecursiveDiffusionLM(cfg).to(cfg.device)
    model_reloaded.load_state_dict(loaded_data["model_state"])
    model_reloaded.prepare_for_eval()
    load_ms = (time.perf_counter() - t_load0) * 1000.0
    ram_post_reload = get_current_rss_mib()
    print(f"  Checkpoint recargado en {load_ms:.1f} ms. RSS post-recarga: {ram_post_reload:.1f} MiB")

    with torch.no_grad():
        out_after = model_reloaded(x_test, t_test, num_recursions=2)

    max_logit_diff = (out_before.final_logits - out_after.final_logits).abs().max().item()
    print(f"  Diferencia máxima absoluta de logits (pre vs post recarga): {max_logit_diff:.2e}")
    assert max_logit_diff == 0.0, f"Discrepancia numérica en recarga: {max_logit_diff}"
    print("  [OK] Invariancia numérica estricta confirmada (identidad exacta bit a bit).")

    # Limpieza de archivo temporal
    if os.path.exists(tmp_ckpt):
        os.remove(tmp_ckpt)
        print(f"  [Limpieza] Archivo temporal {tmp_ckpt} eliminado con éxito.")

    print("\n" + "=" * 78)
    print("  ETAPA 1 COMPLETADA CON ÉXITO: OPCIÓN B LISTA PARA ENTRENAMIENTO DE 800 PASOS")
    print("=" * 78)


if __name__ == "__main__":
    run_prueba_integridad_opcionB()
