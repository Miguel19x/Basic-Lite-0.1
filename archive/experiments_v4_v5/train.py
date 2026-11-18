# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

import argparse
import math
import random
import time
import resource
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import HybridConfig
from model import TernaryRecursiveDiffusionLM
from data import ProceduralCompositionalDataset, apply_blockwise_diffusion_corruption


def compute_v4_hybrid_loss(
    final_logits: torch.Tensor,
    intermediate_logits: List[torch.Tensor],
    halt_probs_list: List[torch.Tensor],
    x_0: torch.Tensor,
    mask_positions: torch.Tensor,
    digit_mask: torch.Tensor,
    t_noise: torch.Tensor,
    cfg: HybridConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Función de pérdida ponderada contra el techo trivial con soporte dual nativo:
    - Salida perezosa al vocabulario (Lazy Logits 2D: [N_supervisados, V]) -> Ahorra ~80% de proyección y softmax.
    - Salida densa clásica (3D: [B, L, V]) -> Mantiene compatibilidad total hacia atrás.
    """
    # Ruta 1: Proyección perezosa al vocabulario (final_logits ya contiene únicamente las posiciones en mask_positions)
    if final_logits.ndim == 2:
        V = final_logits.size(-1)
        x_0_sup = x_0[mask_positions]
        dig_sup = digit_mask[mask_positions]

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
        norm_denom = pos_importance.sum().clamp(min=1.0)

        if t_noise.dim() == 1:
            inv_t_full = (1.0 / t_noise.clamp(min=0.10)).unsqueeze(1).expand_as(x_0)
        else:
            inv_t_full = (1.0 / t_noise.clamp(min=0.10)).expand_as(x_0)
        inv_t_sup = inv_t_full[mask_positions]

        def _weighted_ce_lazy(logits_sup: torch.Tensor) -> torch.Tensor:
            token_loss = F.cross_entropy(logits_sup, x_0_sup, reduction="none")
            return (token_loss * pos_importance * inv_t_sup).sum() / norm_denom

        loss_final = _weighted_ce_lazy(final_logits)

        loss_inter = torch.tensor(0.0, device=x_0.device)
        if len(intermediate_logits) > 1 and cfg.intermediate_loss_weight > 0:
            inter_terms = [_weighted_ce_lazy(lg) for lg in intermediate_logits[:-1]]
            loss_inter = sum(inter_terms) / len(inter_terms)

        loss_cons = torch.tensor(0.0, device=x_0.device)
        if len(intermediate_logits) > 1 and cfg.consistency_loss_weight > 0:
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

        metrics = {
            "loss_total": float(total_loss.item()),
            "train_all_acc": float((all_correct / all_total).item()),
            "train_digit_acc": float((dig_correct / dig_total).item()),
        }
        return total_loss, metrics

    # Ruta 2: Compatibilidad con tensores completos [B, L, V]
    B, L, V = final_logits.shape
    mask_f = mask_positions.float()
    pos_importance = torch.full_like(mask_f, cfg.template_token_weight)
    pos_importance = torch.where(x_0 == cfg.pad_token_id, torch.full_like(mask_f, cfg.pad_token_weight), pos_importance)
    pos_importance = torch.where(digit_mask, torch.full_like(mask_f, cfg.digit_token_weight), pos_importance)
    eff_mask = mask_f * pos_importance
    norm_denom = eff_mask.sum().clamp(min=1.0)

    if t_noise.dim() == 1:
        inv_t = (1.0 / t_noise.clamp(min=0.10)).unsqueeze(1)
    else:
        inv_t = (1.0 / t_noise.clamp(min=0.10))

    def _weighted_ce(logits: torch.Tensor) -> torch.Tensor:
        token_loss = F.cross_entropy(logits.view(-1, V), x_0.view(-1), reduction="none").view(B, L)
        return (token_loss * eff_mask * inv_t).sum() / norm_denom

    loss_final = _weighted_ce(final_logits)

    loss_inter = torch.tensor(0.0, device=x_0.device)
    if len(intermediate_logits) > 1 and cfg.intermediate_loss_weight > 0:
        inter_terms = [_weighted_ce(lg) for lg in intermediate_logits[:-1]]
        loss_inter = sum(inter_terms) / len(inter_terms)

    loss_cons = torch.tensor(0.0, device=x_0.device)
    if len(intermediate_logits) > 1 and cfg.consistency_loss_weight > 0:
        teacher_probs = F.softmax(final_logits.detach(), dim=-1)
        kl_terms = []
        for lg in intermediate_logits[:-1]:
            student_log_probs = F.log_softmax(lg, dim=-1)
            kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
            kl_terms.append((kl * eff_mask).sum() / norm_denom)
        loss_cons = sum(kl_terms) / len(kl_terms)

    loss_ponder = torch.tensor(0.0, device=x_0.device)
    if halt_probs_list and cfg.ponder_loss_weight > 0:
        ponder_terms = []
        for lg, h_prob in zip(intermediate_logits, halt_probs_list):
            is_correct = (lg.detach().argmax(dim=-1) == x_0).float().unsqueeze(-1)
            halt_target = 0.1 + 0.9 * is_correct
            if h_prob.shape[1] != halt_target.shape[1] and halt_target.shape[1] % h_prob.shape[1] == 0:
                p_sz = halt_target.shape[1] // h_prob.shape[1]
                halt_target = halt_target.view(B, h_prob.shape[1], p_sz, 1).mean(dim=2)
            ponder_terms.append(F.mse_loss(h_prob, halt_target))
        loss_ponder = sum(ponder_terms) / len(ponder_terms)

    total_loss = (
        loss_final
        + cfg.intermediate_loss_weight * loss_inter
        + cfg.consistency_loss_weight * loss_cons
        + cfg.ponder_loss_weight * loss_ponder
    )

    with torch.no_grad():
        preds = final_logits.argmax(dim=-1)
        masked_digits = mask_positions & digit_mask
        dig_correct = ((preds == x_0) & masked_digits).float().sum()
        dig_total = masked_digits.float().sum().clamp(min=1.0)
        all_correct = ((preds == x_0) & mask_positions).float().sum()
        all_total = mask_f.sum().clamp(min=1.0)

    metrics = {
        "loss_total": float(total_loss.item()),
        "train_all_acc": float((all_correct / all_total).item()),
        "train_digit_acc": float((dig_correct / dig_total).item()),
    }
    return total_loss, metrics


@torch.no_grad()
def evaluate_digit_and_overall_accuracy(
    model: TernaryRecursiveDiffusionLM,
    val_loader: DataLoader,
    cfg: HybridConfig,
    recursions_to_eval: Optional[Tuple[int, ...]] = None,
) -> Dict[str, float]:
    """
    Mide por separado all_acc y digit_acc usando proyección perezosa sobre mask_pos.
    En modo desarrollo (cfg.dev_mode=True), evalúa únicamente la profundidad principal (R=2).
    """
    model.prepare_for_eval()
    if recursions_to_eval is None:
        recursions_to_eval = cfg.eval_recursions_dev if cfg.dev_mode else cfg.eval_recursions_full

    results = {}
    for r_eval in recursions_to_eval:
        all_ok, all_tot, dig_ok, dig_tot = 0.0, 0.0, 0.0, 0.0
        for x_0, can_mask, digit_mask in val_loader:
            x_t, mask_pos, t_vec = apply_blockwise_diffusion_corruption(x_0, can_mask, cfg, fixed_t=0.50)
            sup_mask = mask_pos if cfg.lazy_vocab_projection else None
            logits, _, _, _, _ = model(
                x_t,
                t_vec,
                num_recursions=r_eval,
                record_telemetry=(r_eval == recursions_to_eval[0] and all_tot == 0),
                supervised_mask=sup_mask,
            )

            if logits.ndim == 2:
                preds_sup = logits.argmax(dim=-1)
                x_0_sup = x_0[mask_pos]
                dig_sup = digit_mask[mask_pos]
                correct_sup = (preds_sup == x_0_sup)
                all_ok += float(correct_sup.float().sum().item())
                all_tot += float(x_0_sup.numel())
                dig_ok += float((correct_sup & dig_sup).float().sum().item())
                dig_tot += float(dig_sup.float().sum().item())
            else:
                preds = logits.argmax(dim=-1)
                masked_digits = mask_pos & digit_mask
                all_ok += float(((preds == x_0) & mask_pos).float().sum().item())
                all_tot += float(mask_pos.float().sum().item())
                dig_ok += float(((preds == x_0) & masked_digits).float().sum().item())
                dig_tot += float(masked_digits.float().sum().item())

        results[f"all_r{r_eval}"] = all_ok / max(1.0, all_tot)
        results[f"dig_r{r_eval}"] = dig_ok / max(1.0, dig_tot)

    for r_default in (1, 2, 4):
        if f"dig_r{r_default}" not in results:
            ref_r = recursions_to_eval[0]
            results[f"dig_r{r_default}"] = results[f"dig_r{ref_r}"]
            results[f"all_r{r_default}"] = results[f"all_r{ref_r}"]

    model.train()
    return results


def update_bitnet_schedule(optimizer: torch.optim.Optimizer, step: int, cfg: HybridConfig) -> float:
    warmup_steps = max(10, int(0.10 * cfg.max_steps))
    second_stage_start = int(0.80 * cfg.max_steps)
    if step <= warmup_steps:
        lr_mult = step / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, cfg.max_steps - warmup_steps)
        lr_mult = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    current_lr = cfg.learning_rate * lr_mult
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
        if param_group["weight_decay"] > 0 and step >= second_stage_start:
            param_group["weight_decay"] = 0.0
    return current_lr


def get_peak_ram_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_stage_profiler(cfg: HybridConfig, n_iters: int = 5) -> Dict[str, float]:
    """
    Perfilador ligero por etapas sobre pocas iteraciones para separar
    corrupción de datos, forward, pérdida, backward y optimizador.
    """
    torch.set_num_threads(cfg.cpu_threads)
    ds = ProceduralCompositionalDataset(cfg, split="train", num_samples=(n_iters + 2) * cfg.batch_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
    model = TernaryRecursiveDiffusionLM(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.configure_adamw_groups(), lr=cfg.learning_rate, betas=cfg.betas)

    stage_times = {
        "data_corrupt_ms": 0.0,
        "forward_ms": 0.0,
        "loss_ms": 0.0,
        "backward_ms": 0.0,
        "optim_ms": 0.0,
    }
    it = iter(loader)

    # 1 iteración de calentamiento
    x_0, can_mask, digit_mask = next(it)
    x_t1, mask_pos1, t1 = apply_blockwise_diffusion_corruption(x_0, can_mask, cfg)
    sup_m = mask_pos1 if cfg.lazy_vocab_projection else None
    lg1, int1, hlt1, _, _ = model(x_t1, t1, supervised_mask=sup_m)
    l1, _ = compute_v4_hybrid_loss(lg1, int1, hlt1, x_0, mask_pos1, digit_mask, t1, cfg)
    optimizer.zero_grad(set_to_none=True)
    l1.backward()
    optimizer.step()

    for _ in range(n_iters):
        x_0, can_mask, digit_mask = next(it)

        t0 = time.perf_counter()
        x_t1, mask_pos1, t1 = apply_blockwise_diffusion_corruption(x_0, can_mask, cfg)
        stage_times["data_corrupt_ms"] += (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        sup1 = mask_pos1 if cfg.lazy_vocab_projection else None
        logits1, inter1, halt1, _, carried1 = model(x_t1, t1, supervised_mask=sup1)

        # Paso 2 con predicción desacoplada del modelo (sin filtración de ground-truth)
        if logits1.ndim == 2:
            preds_sup1 = logits1.detach().argmax(dim=-1)
            preds1 = x_t1.clone()
            preds1[mask_pos1] = preds_sup1
        else:
            preds1 = logits1.detach().argmax(dim=-1)

        frac_keep = torch.empty_like(t1).uniform_(0.35, 0.85)
        keep_cond = (torch.rand_like(x_0, dtype=torch.float32) < frac_keep.unsqueeze(1)) if t1.dim() == 1 else (torch.rand_like(x_0, dtype=torch.float32) < frac_keep)
        keep_mask = keep_cond & mask_pos1
        for b in range(x_0.size(0)):
            if not keep_mask[b].any() and mask_pos1[b].any():
                idx_m = torch.nonzero(mask_pos1[b], as_tuple=False).squeeze(-1)
                keep_mask[b, idx_m[0]] = True

        x_t2 = x_t1.clone()
        revealed_pos = mask_pos1 & ~keep_mask
        x_t2[revealed_pos] = preds1[revealed_pos]
        x_t2[keep_mask] = cfg.mask_token_id
        t2 = (t1 * frac_keep).clamp(min=0.05)
        dt = (t1 - t2).clamp(min=0.01)

        sup2 = keep_mask if cfg.lazy_vocab_projection else None
        logits2, inter2, halt2, _, _ = model(
            x_t2, t2, delta_t=dt, prev_states=carried1, record_telemetry=True, supervised_mask=sup2
        )
        stage_times["forward_ms"] += (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        loss1, _ = compute_v4_hybrid_loss(logits1, inter1, halt1, x_0, mask_pos1, digit_mask, t1, cfg)
        loss2, _ = compute_v4_hybrid_loss(logits2, inter2, halt2, x_0, keep_mask, digit_mask, t2, cfg)
        total_loss = 0.45 * loss1 + 0.55 * loss2
        stage_times["loss_ms"] += (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        stage_times["backward_ms"] += (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if torch.isfinite(grad_norm):
            optimizer.step()
        stage_times["optim_ms"] += (time.perf_counter() - t0) * 1000.0

    for k in stage_times:
        stage_times[k] /= max(1, n_iters)
    stage_times["total_step_ms"] = sum(stage_times.values())
    tern_stats = model.collect_ternary_telemetry()
    stage_times["moe_active_slot_ratio"] = tern_stats.get("moe_active_slot_ratio", 1.0)
    return stage_times


def benchmark_thread_budget(thread_candidates: Tuple[int, ...] = (1, 2, 4), n_iters: int = 4) -> Dict[int, float]:
    """Mide empíricamente 1, 2 y 4 hilos sobre la misma carga de trabajo."""
    results = {}
    for th in thread_candidates:
        cfg_th = HybridConfig(cpu_threads=th)
        stats = run_stage_profiler(cfg_th, n_iters=n_iters)
        results[th] = stats["total_step_ms"]
        print(
            f"[Presupuesto Hilos] threads={th} -> {stats['total_step_ms']:.1f} ms/paso "
            f"(Fwd: {stats['forward_ms']:.1f} ms | Loss: {stats['loss_ms']:.1f} ms | "
            f"Bwd: {stats['backward_ms']:.1f} ms | Plazas MoE activas: {100*stats['moe_active_slot_ratio']:.1f}%)"
        )
    return results


def train(
    dev_mode: bool = False,
    max_steps_override: Optional[int] = None,
    threads_override: Optional[int] = None,
):
    cfg = HybridConfig(dev_mode=dev_mode)
    if max_steps_override is not None:
        cfg.max_steps = max_steps_override
    if threads_override is not None:
        cfg.cpu_threads = threads_override

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)

    val_count = cfg.val_samples_dev if cfg.dev_mode else cfg.val_samples_full
    train_ds = ProceduralCompositionalDataset(cfg, split="train", num_samples=3200)
    val_ds = ProceduralCompositionalDataset(cfg, split="val", num_samples=val_count)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = TernaryRecursiveDiffusionLM(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.configure_adamw_groups(), lr=cfg.learning_rate, betas=cfg.betas)

    unique_params = sum(p.numel() for p in model.parameters())
    mode_label = "DESARROLLO (Rápido)" if cfg.dev_mode else "EXPERIMENTO (Completo)"
    print(
        f"[TRDM-v4 | Modo: {mode_label}] Parámetros únicos: {unique_params:,} | "
        f"Hilos CPU: {cfg.cpu_threads} | LazyLogits: {cfg.lazy_vocab_projection} | "
        f"GrafoMoEDisperso: {cfg.use_sparse_module_graph}"
    )
    print("[Nota] Techo trivial de plantilla+PAD en AllAcc = 86.7%. La métrica real de cálculo es DigAcc (solo dígitos).")

    step = 0
    train_time_accum = 0.0
    eval_time_accum = 0.0

    while step < cfg.max_steps:
        for x_0, can_mask, digit_mask in train_loader:
            t_step0 = time.perf_counter()
            step += 1
            lr_now = update_bitnet_schedule(optimizer, step, cfg)
            record_tel = (step % cfg.eval_every == 0) or (step == 1)

            r_train = random.choice([1, 2, 3]) if cfg.randomize_train_recursions else cfg.num_recursions

            # Paso encadenado 1 (ruido alto t1)
            x_t1, mask_pos1, t1 = apply_blockwise_diffusion_corruption(x_0, can_mask, cfg)
            sup_mask1 = mask_pos1 if cfg.lazy_vocab_projection else None
            logits1, inter1, halt1, _, carried1 = model(
                x_t1, t1, num_recursions=r_train, record_telemetry=False, supervised_mask=sup_mask1
            )
            loss1, _ = compute_v4_hybrid_loss(
                logits1, inter1, halt1, x_0, mask_pos1, digit_mask, t1, cfg
            )

            # Paso encadenado 2:
            # Las posiciones que se revelan entre el paso 1 y el paso 2 reciben la PREDICCIÓN de la red
            # (con .detach()), NO la solución perfecta x_0, preservando el contrato de difusión.
            if logits1.ndim == 2:
                preds_sup1 = logits1.detach().argmax(dim=-1)
                preds1 = x_t1.clone()
                preds1[mask_pos1] = preds_sup1
            else:
                preds1 = logits1.detach().argmax(dim=-1)

            frac_keep = torch.empty_like(t1).uniform_(0.35, 0.85)
            keep_cond = (torch.rand_like(x_0, dtype=torch.float32) < frac_keep.unsqueeze(1)) if t1.dim() == 1 else (torch.rand_like(x_0, dtype=torch.float32) < frac_keep)
            keep_mask = keep_cond & mask_pos1
            for b in range(x_0.size(0)):
                if not keep_mask[b].any() and mask_pos1[b].any():
                    idx_m = torch.nonzero(mask_pos1[b], as_tuple=False).squeeze(-1)
                    keep_mask[b, idx_m[0]] = True

            x_t2 = x_t1.clone()
            revealed_pos = mask_pos1 & ~keep_mask
            x_t2[revealed_pos] = preds1[revealed_pos]
            x_t2[keep_mask] = cfg.mask_token_id

            t2 = (t1 * frac_keep).clamp(min=0.05)
            dt = (t1 - t2).clamp(min=0.01)

            sup_mask2 = keep_mask if cfg.lazy_vocab_projection else None
            logits2, inter2, halt2, _, _ = model(
                x_t2,
                t2,
                delta_t=dt,
                prev_states=carried1,
                num_recursions=r_train,
                record_telemetry=record_tel,
                supervised_mask=sup_mask2,
            )
            loss2, metrics2 = compute_v4_hybrid_loss(
                logits2, inter2, halt2, x_0, keep_mask, digit_mask, t2, cfg
            )

            total_loss = 0.45 * loss1 + 0.55 * loss2
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if torch.isfinite(grad_norm):
                optimizer.step()

            train_time_accum += time.perf_counter() - t_step0

            if record_tel:
                t_ev0 = time.perf_counter()
                val_m = evaluate_digit_and_overall_accuracy(model, val_loader, cfg)
                ev_dt = time.perf_counter() - t_ev0
                eval_time_accum += ev_dt

                tern_stats = model.collect_ternary_telemetry()
                sec_per_step = train_time_accum / max(1, step)
                moe_active_pct = 100.0 * tern_stats.get("moe_active_slot_ratio", 1.0)

                print(
                    f"Paso {step:03d}/{cfg.max_steps} | LR: {lr_now:.2e} | Loss: {total_loss.item():.3f} | "
                    f"TrainDig: {metrics2['train_digit_acc']:.2f} | "
                    f"OOD DigAcc(R=1/2/4): {val_m['dig_r1']:.2f}/{val_m['dig_r2']:.2f}/{val_m['dig_r4']:.2f} | "
                    f"OOD AllAcc(R=2): {val_m['all_r2']:.2f} | "
                    f"MoE Activo: {moe_active_pct:.0f}% | "
                    f"BITCOS: {tern_stats.get('bitcos_bpw', 0):.2f} bpw | "
                    f"Train: {sec_per_step*1000:.0f}ms/p | Eval: {ev_dt:.2f}s | RAM: {get_peak_ram_mib():.1f} MiB",
                    flush=True,
                )

            if step >= cfg.max_steps:
                break

    torch.save({"model_state": model.state_dict(), "config": cfg}, cfg.checkpoint_path)
    packed_bytes = model.export_bitcos_archive(cfg.bitcos_export_path)
    print(
        f"[Guardado] Checkpoint: {cfg.checkpoint_path} | BITCOSv2: {packed_bytes / 1024:.1f} KiB | "
        f"Tiempo Train Total: {train_time_accum:.1f}s | Tiempo Eval Total: {eval_time_accum:.1f}s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento y Perfilador con Presupuesto TRDM-v4")
    parser.add_argument("--dev", action="store_true", help="Activa modo de desarrollo barato (val=64, R=2)")
    parser.add_argument("--steps", type=int, default=None, help="Sobrescribe max_steps")
    parser.add_argument("--threads", type=int, default=None, help="Sobrescribe cpu_threads")
    parser.add_argument("--profile", action="store_true", help="Ejecuta el perfilador por etapas y termina")
    parser.add_argument("--bench-threads", action="store_true", help="Compara rendimiento con 1, 2 y 4 hilos")
    args = parser.parse_args()

    if args.bench_threads:
        benchmark_thread_budget()
    elif args.profile:
        cfg_p = HybridConfig(cpu_threads=args.threads if args.threads else 2)
        st = run_stage_profiler(cfg_p)
        print("=== Perfilado por Etapas (Media por paso de 2 pasadas encadenadas) ===")
        for k, v in st.items():
            if k.endswith("_ms"):
                print(f"  {k:<20s}: {v:.2f} ms")
            else:
                print(f"  {k:<20s}: {100*v:.1f}%")
    else:
        train(dev_mode=args.dev, max_steps_override=args.steps, threads_override=args.threads)
