import math
import time
import resource
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import HybridConfig
from model import TernaryRecursiveDiffusionLM
from data import ProceduralNaturalTextDataset, apply_discrete_diffusion_corruption


def update_bitnet_schedule(optimizer: torch.optim.Optimizer, step: int, cfg: HybridConfig) -> float:
    """
    Planificador en dos etapas para BitNet b1.58 + AdamW:
    - Calentamiento lineal (primer 10% de pasos).
    - Decaimiento coseno posterior.
    - Desactivación de weight decay en el último 20% del entrenamiento para estabilizar STE.
    """
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


def compute_v2_hybrid_loss(
    final_logits: torch.Tensor,
    intermediate_logits: List[torch.Tensor],
    halt_probs_list: List[torch.Tensor],
    x_0: torch.Tensor,
    mask_positions: torch.Tensor,
    t_noise: torch.Tensor,
    cfg: HybridConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    B, L, V = final_logits.shape
    mask_f = mask_positions.float()
    total_masked = mask_f.sum().clamp(min=1.0)

    def _masked_weighted_ce(logits: torch.Tensor) -> torch.Tensor:
        token_loss = F.cross_entropy(logits.view(-1, V), x_0.view(-1), reduction="none").view(B, L)
        weights = (1.0 / t_noise).unsqueeze(1)
        return (token_loss * mask_f * weights).sum() / total_masked

    loss_final = _masked_weighted_ce(final_logits)

    loss_inter = torch.tensor(0.0, device=x_0.device)
    if len(intermediate_logits) > 1 and cfg.intermediate_loss_weight > 0:
        inter_terms = [_masked_weighted_ce(lg) for lg in intermediate_logits[:-1]]
        loss_inter = sum(inter_terms) / len(inter_terms)

    loss_cons = torch.tensor(0.0, device=x_0.device)
    if len(intermediate_logits) > 1 and cfg.consistency_loss_weight > 0:
        teacher_probs = F.softmax(final_logits.detach(), dim=-1)
        kl_terms = []
        for lg in intermediate_logits[:-1]:
            student_log_probs = F.log_softmax(lg, dim=-1)
            kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
            kl_masked = (kl * mask_f).sum() / total_masked
            kl_terms.append(kl_masked)
        loss_cons = sum(kl_terms) / len(kl_terms)

    loss_ponder = torch.tensor(0.0, device=x_0.device)
    if halt_probs_list and cfg.ponder_loss_weight > 0:
        ponder_terms = []
        for lg, h_prob in zip(intermediate_logits, halt_probs_list):
            is_correct = (lg.detach().argmax(dim=-1) == x_0).float().unsqueeze(-1)
            halt_target = 0.1 + 0.9 * is_correct
            ponder_terms.append(F.mse_loss(h_prob, halt_target))
        loss_ponder = sum(ponder_terms) / len(ponder_terms)

    total_loss = (
        loss_final
        + cfg.intermediate_loss_weight * loss_inter
        + cfg.consistency_loss_weight * loss_cons
        + cfg.ponder_loss_weight * loss_ponder
    )

    depth_acc = {}
    with torch.no_grad():
        for r_idx, lg in enumerate(intermediate_logits):
            preds = lg.argmax(dim=-1)
            correct = ((preds == x_0) & mask_positions).float().sum()
            depth_acc[f"acc_r{r_idx+1}"] = float((correct / total_masked).item())

    metrics = {
        "loss_total": float(total_loss.item()),
        "loss_diff": float(loss_final.item()),
        "loss_cons": float(loss_cons.item()),
        "loss_ponder": float(loss_ponder.item()),
        **depth_acc,
    }
    return total_loss, metrics


@torch.no_grad()
def evaluate_model(
    model: TernaryRecursiveDiffusionLM, val_loader: DataLoader, cfg: HybridConfig
) -> Dict[str, float]:
    model.eval()
    results = {}
    for fixed_t in (0.50, 1.00):
        total_acc, count = 0.0, 0
        for x_0, can_mask in val_loader:
            x_t, mask_pos, t_vec = apply_discrete_diffusion_corruption(x_0, can_mask, cfg, fixed_t=fixed_t)
            logits, _, _, _, _ = model(x_t, t_vec, record_telemetry=(fixed_t == 0.50 and count == 0))
            preds = logits.argmax(dim=-1)
            acc = ((preds == x_0) & mask_pos).float().sum() / mask_pos.float().sum().clamp(min=1.0)
            total_acc += float(acc.item())
            count += 1
        results[f"val_acc_t{int(fixed_t*100)}"] = total_acc / max(1, count)
    model.train()
    return results


def get_peak_ram_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def train():
    cfg = HybridConfig()
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)

    train_ds = ProceduralNaturalTextDataset(cfg, split="train", num_samples=2400)
    val_ds = ProceduralNaturalTextDataset(cfg, split="val", num_samples=160)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = TernaryRecursiveDiffusionLM(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.configure_adamw_groups(), lr=cfg.learning_rate, betas=cfg.betas)

    unique_params = sum(p.numel() for p in model.parameters())
    print(f"[TRDM-v3] Parámetros únicos: {unique_params:,} | GQA {cfg.n_heads}:{cfg.n_kv_heads} | BlockSize: {cfg.block_size} | Longitud L={cfg.max_seq_len}")

    step = 0
    t0 = time.perf_counter()

    while step < cfg.max_steps:
        for x_0, can_mask in train_loader:
            step += 1
            record_tel = (step % cfg.eval_every == 0) or (step == 1)

            x_t, mask_pos, t_vec = apply_discrete_diffusion_corruption(x_0, can_mask, cfg)
            final_logits, inter_logits, halt_probs, _, _ = model(x_t, t_vec, record_telemetry=record_tel)
            loss, metrics = compute_v2_hybrid_loss(
                final_logits, inter_logits, halt_probs, x_0, mask_pos, t_vec, cfg
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            if torch.isfinite(grad_norm):
                current_lr = update_bitnet_schedule(optimizer, step, cfg)
                optimizer.step()
            else:
                current_lr = optimizer.param_groups[0]["lr"]

            if record_tel:
                val_metrics = evaluate_model(model, val_loader, cfg)
                tern_stats = model.collect_ternary_telemetry()
                act_ratios = [round(a, 2) for a in model.last_active_token_ratios]
                elapsed = time.perf_counter() - t0

                print(
                    f"Paso {step:03d}/{cfg.max_steps} | "
                    f"LR: {current_lr:.2e} | "
                    f"Loss: {metrics['loss_total']:.3f} (Cons:{metrics['loss_cons']:.3f}) | "
                    f"Acc(r1->r{cfg.num_recursions}): {metrics['acc_r1']:.2f}->{metrics[f'acc_r{cfg.num_recursions}']:.2f} | "
                    f"Val(50/100%): {val_metrics['val_acc_t50']:.2f}/{val_metrics['val_acc_t100']:.2f} | "
                    f"BITCOS: {tern_stats.get('bitcos_bpw', 0):.2f} bpw | "
                    f"MoR Activos: {act_ratios} | RAM: {get_peak_ram_mb():.1f} MB | {elapsed:.1f}s"
                )

            if step >= cfg.max_steps:
                break

    torch.save({"model_state": model.state_dict(), "config": cfg}, cfg.checkpoint_path)
    packed_bytes = model.export_bitcos_archive(cfg.bitcos_export_path)
    print(f"[ Guardado ] Checkpoint: {cfg.checkpoint_path} | Archivo BITCOSv2: {packed_bytes / 1024:.1f} KB")


if __name__ == "__main__":
    train()
