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
from model import TernaryRecursiveDiffusionLM_v5
from data import ProceduralCompositionalDataset, apply_blockwise_diffusion_corruption
from train import compute_v4_hybrid_loss, update_bitnet_schedule, get_peak_ram_mib


@torch.no_grad()
def evaluate_v5_accuracy(
    model: TernaryRecursiveDiffusionLM_v5,
    val_loader: DataLoader,
    cfg: HybridConfig,
    recursions_to_eval: Tuple[int, ...] = (1, 2, 4),
) -> Dict[str, float]:
    """
    Evalúa la exactitud de TRDM-v5 en dígitos (DigAcc) y total (AllAcc)
    a través de diferentes profundidades recursivas R in {1, 2, 4}.
    """
    model.prepare_for_eval()
    results = {}

    for r_eval in recursions_to_eval:
        all_ok, all_tot, dig_ok, dig_tot = 0.0, 0.0, 0.0, 0.0
        for x_0, can_mask, digit_mask in val_loader:
            x_t, mask_pos, t_vec = apply_blockwise_diffusion_corruption(
                x_0, can_mask, cfg, fixed_t=0.50
            )
            # TRDM-v5 forward
            out = model(x_t, t_vec, num_recursions=r_eval)
            logits = out.final_logits

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


def train_v5(
    dev_mode: bool = False,
    max_steps_override: Optional[int] = None,
    threads_override: Optional[int] = None,
):
    cfg = HybridConfig(dev_mode=dev_mode)
    if max_steps_override is not None:
        cfg.max_steps = max_steps_override
    if threads_override is not None:
        cfg.cpu_threads = threads_override

    # Asegurar configuración TRDM-v5
    cfg.checkpoint_path = "trdm_v5_checkpoint.pt"
    cfg.bitcos_export_path = "trdm_v5_paged_bitcos.bin"
    cfg.use_byte_patching = True
    cfg.byte_patch_size = 4
    cfg.use_mla = True

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)

    val_count = cfg.val_samples_dev if cfg.dev_mode else cfg.val_samples_full
    train_ds = ProceduralCompositionalDataset(cfg, split="train", num_samples=3200)
    val_ds = ProceduralCompositionalDataset(cfg, split="val", num_samples=val_count)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = TernaryRecursiveDiffusionLM_v5(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.configure_adamw_groups(), lr=cfg.learning_rate, betas=cfg.betas)

    unique_params = sum(p.numel() for p in model.parameters())
    mode_label = "DESARROLLO (Rápido)" if cfg.dev_mode else "EXPERIMENTO (Completo)"
    print(
        f"[TRDM-v5 | Modo: {mode_label}] Parámetros únicos: {unique_params:,} | "
        f"Hilos CPU: {cfg.cpu_threads} | BytePatching: P={cfg.byte_patch_size} | "
        f"MLA: Latent={cfg.mla_latent_dim} | PagedMoE: MaxResident={cfg.max_resident_experts}/{cfg.num_experts}"
    )

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
            out1 = model(x_t1, t1, num_recursions=r_train, record_telemetry=False)
            logits1, inter1, halt1, _, carried1 = (
                out1.final_logits,
                out1.intermediate_logits,
                out1.halt_probs,
                out1.kv_caches,
                out1.carried_states,
            )
            loss1, _ = compute_v4_hybrid_loss(
                logits1, inter1, halt1, x_0, mask_pos1, digit_mask, t1, cfg
            )

            # Paso encadenado 2: Predicciones desacopladas de la red
            preds1 = logits1.detach().argmax(dim=-1)

            frac_keep = torch.empty_like(t1).uniform_(0.35, 0.85)
            keep_cond = (
                (torch.rand_like(x_0, dtype=torch.float32) < frac_keep.unsqueeze(1))
                if t1.dim() == 1
                else (torch.rand_like(x_0, dtype=torch.float32) < frac_keep)
            )
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

            out2 = model(
                x_t2,
                t2,
                delta_t=dt,
                prev_states=carried1,
                num_recursions=r_train,
                record_telemetry=record_tel,
            )
            logits2, inter2, halt2 = out2.final_logits, out2.intermediate_logits, out2.halt_probs

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
                val_m = evaluate_v5_accuracy(model, val_loader, cfg)
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
    export_info = model.export_paged_bitcos(cfg.bitcos_export_path, format_type="fast")
    packed_bytes = export_info.get("total_bytes", 0)
    print(
        f"[Guardado TRDM-v5] Checkpoint: {cfg.checkpoint_path} | PagedBitCOS: {packed_bytes / 1024:.1f} KiB | "
        f"Tiempo Train Total: {train_time_accum:.1f}s | Tiempo Eval Total: {eval_time_accum:.1f}s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de Calibración TRDM-v5")
    parser.add_argument("--dev", action="store_true", help="Modo de desarrollo rápido")
    parser.add_argument("--steps", type=int, default=200, help="Número de pasos de calibración")
    parser.add_argument("--threads", type=int, default=2, help="Hilos CPU")
    args = parser.parse_args()

    train_v5(dev_mode=args.dev, max_steps_override=args.steps, threads_override=args.threads)
