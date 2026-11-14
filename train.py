import math
import time
import resource
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import HybridConfig
from model import TernaryRecursiveDiffusionLM
from data import ProceduralCompositionalDataset, apply_discrete_diffusion_corruption


def corrupt_batch(
    x_0: torch.Tensor, can_mask: torch.Tensor, t_vec: torch.Tensor, cfg: HybridConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = x_0.size(0)
    rand_u = torch.rand_like(x_0.float())
    mask_pos = (rand_u < t_vec.unsqueeze(1)) & can_mask
    for b in range(B):
        if not mask_pos[b].any() and can_mask[b].any():
            valid_idx = torch.nonzero(can_mask[b], as_tuple=False).squeeze(-1)
            chosen = valid_idx[torch.randint(0, len(valid_idx), (1,))]
            mask_pos[b, chosen] = True
    x_t = x_0.clone()
    x_t[mask_pos] = cfg.mask_token_id
    return x_t, mask_pos


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
        # Ponderación 1/t normalizada por tokens enmascarados
        weights = (1.0 / t_noise.clamp(min=0.05)).unsqueeze(1)
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
def evaluate_model_h1(
    model: TernaryRecursiveDiffusionLM, val_loader: DataLoader, cfg: HybridConfig
) -> Dict[str, float]:
    """
    Evaluación rigurosa sobre combinaciones estrictamente reservadas (Held-Out)
    probando la Hipótesis H1 (escalado en inferencia R=1, 2, 4 sin reentrenar).
    """
    model.eval()
    results = {}
    for r_infer in (1, 2, 4):
        total_acc, count = 0.0, 0
        for x_0, can_mask in val_loader:
            # Evaluación a nivel de ruido medio (t=0.50)
            x_t, mask_pos, t_vec = apply_discrete_diffusion_corruption(x_0, can_mask, cfg, fixed_t=0.50)
            dt_vec = torch.full_like(t_vec, 0.25)
            logits, _, _, _, _ = model(
                x_t, t_vec, delta_t=dt_vec, num_recursions=r_infer, record_telemetry=False
            )
            preds = logits.argmax(dim=-1)
            acc = ((preds == x_0) & mask_pos).float().sum() / mask_pos.float().sum().clamp(min=1.0)
            total_acc += float(acc.item())
            count += 1
        results[f"val_ood_acc_R{r_infer}"] = total_acc / max(1, count)
    model.train()
    return results


def get_peak_ram_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def train():
    cfg = HybridConfig()
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)

    # Dataset composicional con 96 combinaciones reservadas (held-out)
    train_ds = ProceduralCompositionalDataset(cfg, split="train", num_samples=2400)
    val_ds = ProceduralCompositionalDataset(cfg, split="val", num_samples=320)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = TernaryRecursiveDiffusionLM(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.configure_adamw_groups(), lr=cfg.learning_rate, betas=cfg.betas)

    unique_params = sum(p.numel() for p in model.parameters())
    print(f"[TRDM-v3 Saneado] Parámetros únicos: {unique_params:,} | R_train={cfg.num_recursions} | GQA {cfg.n_heads}:{cfg.n_kv_heads} | BlockSize: {cfg.block_size}")

    step = 0
    t0 = time.perf_counter()

    while step < cfg.max_steps:
        for x_0, can_mask in train_loader:
            step += 1
            record_tel = (step % cfg.eval_every == 0) or (step == 1)

            # -------------------------------------------------------------
            # Paso A: Ruido inicial t_1 ~ U(0.35, 1.0)
            # -------------------------------------------------------------
            t_1 = torch.empty(cfg.batch_size, device=cfg.device).uniform_(0.35, 1.0)
            x_t1, mask_1 = corrupt_batch(x_0, can_mask, t_1, cfg)

            final_lg_1, inter_1, h_1, _, carried_1 = model(x_t1, t_1, record_telemetry=record_tel)
            loss_1, metrics_1 = compute_v2_hybrid_loss(final_lg_1, inter_1, h_1, x_0, mask_1, t_1, cfg)

            # -------------------------------------------------------------
            # Paso B: Paso encadenado con memoria transportada y delta_t variable
            # Entrena activamente carry_gate, delta_t y la evaluación con t_2 -> 0.0
            # -------------------------------------------------------------
            t_2 = t_1 * torch.empty(cfg.batch_size, device=cfg.device).uniform_(0.0, 0.6)
            delta_t = t_1 - t_2
            x_t2, mask_2 = corrupt_batch(x_0, can_mask, t_2, cfg)

            final_lg_2, inter_2, h_2, _, _ = model(
                x_t2, t_2, delta_t=delta_t, prev_states=carried_1, record_telemetry=False
            )
            loss_2, metrics_2 = compute_v2_hybrid_loss(final_lg_2, inter_2, h_2, x_0, mask_2, t_2, cfg)

            total_loss = 0.5 * (loss_1 + loss_2)

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            if torch.isfinite(grad_norm):
                current_lr = update_bitnet_schedule(optimizer, step, cfg)
                optimizer.step()
            else:
                current_lr = optimizer.param_groups[0]["lr"]

            if record_tel:
                h1_val_metrics = evaluate_model_h1(model, val_loader, cfg)
                tern_stats = model.collect_ternary_telemetry()
                act_ratios = [round(a, 2) for a in model.last_active_token_ratios]
                elapsed = time.perf_counter() - t0

                print(
                    f"Paso {step:03d}/{cfg.max_steps} | "
                    f"LR: {current_lr:.2e} | "
                    f"Loss: {total_loss.item():.3f} | "
                    f"H1 OOD Acc(R=1/2/4): {h1_val_metrics['val_ood_acc_R1']:.2f}/{h1_val_metrics['val_ood_acc_R2']:.2f}/{h1_val_metrics['val_ood_acc_R4']:.2f} | "
                    f"BITCOS: {tern_stats.get('bitcos_bpw', 0):.2f} bpw | "
                    f"MoR: {act_ratios} | RAM: {get_peak_ram_mb():.1f} MB | {elapsed:.1f}s",
                    flush=True,
                )

            if step >= cfg.max_steps:
                break

    torch.save({"model_state": model.state_dict(), "config": cfg}, cfg.checkpoint_path)
    packed_bytes = model.export_bitcos_archive(cfg.bitcos_export_path)
    print(f"[ Guardado ] Checkpoint: {cfg.checkpoint_path} | Archivo BITCOSv2: {packed_bytes / 1024:.1f} KB")


if __name__ == "__main__":
    train()
