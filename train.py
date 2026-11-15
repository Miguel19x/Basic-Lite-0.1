import math
import random
import time
import resource
from typing import Dict, List, Tuple
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
    Función de pérdida ponderada contra el techo trivial:
    Asigna peso x6.0 a los bytes de dígitos ('0'..'9') y x0.20 a los tokens PAD.
    """
    B, L, V = final_logits.shape
    mask_f = mask_positions.float()

    # Mapa de importancia por posición supervisada
    pos_importance = torch.full_like(mask_f, cfg.template_token_weight)
    pos_importance = torch.where(x_0 == cfg.pad_token_id, torch.full_like(mask_f, cfg.pad_token_weight), pos_importance)
    pos_importance = torch.where(digit_mask, torch.full_like(mask_f, cfg.digit_token_weight), pos_importance)

    eff_mask = mask_f * pos_importance
    norm_denom = eff_mask.sum().clamp(min=1.0)

    def _weighted_ce(logits: torch.Tensor) -> torch.Tensor:
        token_loss = F.cross_entropy(logits.view(-1, V), x_0.view(-1), reduction="none").view(B, L)
        inv_t = (1.0 / t_noise.clamp(min=0.10)).unsqueeze(1)
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
    model: TernaryRecursiveDiffusionLM, val_loader: DataLoader, cfg: HybridConfig
) -> Dict[str, float]:
    """
    Mide por separado para R in {1, 2, 4}:
    - all_acc: exactitud global en posiciones enmascaradas (cuyo techo trivial es ~86.7%)
    - digit_acc: exactitud EXCLUSIVA sobre los bytes de dígitos ('0'..'9')
    """
    model.prepare_for_eval()
    results = {}
    for r_eval in (1, 2, 4):
        all_ok, all_tot, dig_ok, dig_tot = 0.0, 0.0, 0.0, 0.0
        for x_0, can_mask, digit_mask in val_loader:
            x_t, mask_pos, t_vec = apply_blockwise_diffusion_corruption(x_0, can_mask, cfg, fixed_t=0.50)
            logits, _, _, _, _ = model(
                x_t, t_vec, num_recursions=r_eval, record_telemetry=(r_eval == 2 and all_tot == 0)
            )
            preds = logits.argmax(dim=-1)
            masked_digits = mask_pos & digit_mask

            all_ok += float(((preds == x_0) & mask_pos).float().sum().item())
            all_tot += float(mask_pos.float().sum().item())
            dig_ok += float(((preds == x_0) & masked_digits).float().sum().item())
            dig_tot += float(masked_digits.float().sum().item())

        results[f"all_r{r_eval}"] = all_ok / max(1.0, all_tot)
        results[f"dig_r{r_eval}"] = dig_ok / max(1.0, dig_tot)
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


def train():
    cfg = HybridConfig()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)

    train_ds = ProceduralCompositionalDataset(cfg, split="train", num_samples=3200)
    val_ds = ProceduralCompositionalDataset(cfg, split="val", num_samples=384)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = TernaryRecursiveDiffusionLM(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.configure_adamw_groups(), lr=cfg.learning_rate, betas=cfg.betas)

    unique_params = sum(p.numel() for p in model.parameters())
    print(f"[TRDM-v4] Parámetros únicos: {unique_params:,} | Split: 384 Train / 96 Held-Out | Peso dígitos: x{cfg.digit_token_weight}")
    print("[Nota] Techo trivial de plantilla+PAD en AllAcc = 86.7%. La métrica real de cálculo es DigAcc (solo dígitos).")

    step = 0
    t0 = time.perf_counter()

    while step < cfg.max_steps:
        for x_0, can_mask, digit_mask in train_loader:
            step += 1
            lr_now = update_bitnet_schedule(optimizer, step, cfg)
            record_tel = (step % cfg.eval_every == 0) or (step == 1)

            r_train = random.choice([1, 2, 3]) if cfg.randomize_train_recursions else cfg.num_recursions

            # Paso encadenado 1 (ruido alto t1) -> Paso 2 (ruido menor t2 con carry_gate y delta_t)
            x_t1, mask_pos1, t1 = apply_blockwise_diffusion_corruption(x_0, can_mask, cfg)
            logits1, inter1, halt1, _, carried1 = model(
                x_t1, t1, num_recursions=r_train, record_telemetry=False
            )
            loss1, _ = compute_v4_hybrid_loss(
                logits1, inter1, halt1, x_0, mask_pos1, digit_mask, t1, cfg
            )

            frac_keep = torch.empty_like(t1).uniform_(0.35, 0.85)
            keep_mask = (torch.rand_like(x_0, dtype=torch.float32) < frac_keep.unsqueeze(1)) & mask_pos1
            for b in range(x_0.size(0)):
                if not keep_mask[b].any() and mask_pos1[b].any():
                    idx_m = torch.nonzero(mask_pos1[b], as_tuple=False).squeeze(-1)
                    keep_mask[b, idx_m[0]] = True

            x_t2 = x_0.clone()
            x_t2[keep_mask] = cfg.mask_token_id
            t2 = (t1 * frac_keep).clamp(min=0.05)
            dt = (t1 - t2).clamp(min=0.01)

            logits2, inter2, halt2, _, _ = model(
                x_t2, t2, delta_t=dt, prev_states=carried1, num_recursions=r_train, record_telemetry=record_tel
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

            if record_tel:
                val_m = evaluate_digit_and_overall_accuracy(model, val_loader, cfg)
                tern_stats = model.collect_ternary_telemetry()
                elapsed = time.perf_counter() - t0
                print(
                    f"Paso {step:03d}/{cfg.max_steps} | LR: {lr_now:.2e} | Loss: {total_loss.item():.3f} | "
                    f"TrainDig: {metrics2['train_digit_acc']:.2f} | "
                    f"OOD DigAcc(R=1/2/4): {val_m['dig_r1']:.2f}/{val_m['dig_r2']:.2f}/{val_m['dig_r4']:.2f} | "
                    f"OOD AllAcc(R=2): {val_m['all_r2']:.2f} | "
                    f"BITCOS: {tern_stats.get('bitcos_bpw', 0):.2f} bpw | RAM: {get_peak_ram_mib():.1f} MiB | {elapsed:.1f}s",
                    flush=True,
                )

            if step >= cfg.max_steps:
                break

    torch.save({"model_state": model.state_dict(), "config": cfg}, cfg.checkpoint_path)
    packed_bytes = model.export_bitcos_archive(cfg.bitcos_export_path)
    print(f"[Guardado] Checkpoint: {cfg.checkpoint_path} | BITCOSv2: {packed_bytes / 1024:.1f} KiB")


if __name__ == "__main__":
    train()
