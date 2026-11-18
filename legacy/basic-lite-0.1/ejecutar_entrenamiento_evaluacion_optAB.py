import math
import os
import random
import re
import resource
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import HybridConfig
from model import TernaryRecursiveDiffusionLM
from data import ProceduralCompositionalDataset
from trdm.diffusion.corruption import apply_blockwise_diffusion_corruption
from evaluar_muestreador_corregido import (
    generate_block_causal_corrected,
    parse_subtotals_dual,
    parse_finals_dual,
)
from trdm.data.parsers import parse_scratchpad_prediction


def get_current_rss_mib() -> float:
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:
        return 0.0


def get_peak_ram_mib() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024.0


def compute_isolated_digit_acc(pred_str: str, target_str: str) -> Tuple[int, int]:
    correct, total = 0, 0
    for idx, ch in enumerate(target_str):
        if ch.isdigit():
            total += 1
            if idx < len(pred_str) and pred_str[idx] == ch:
                correct += 1
    return correct, total


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


def build_field_ids(prompt_b: bytes, target_b: bytes, max_seq_len: int, use_scratchpad: bool) -> torch.Tensor:
    field_ids = torch.zeros(max_seq_len, dtype=torch.long)
    prompt_len = 1 + len(prompt_b)

    if use_scratchpad:
        lines = target_b.split(b"\n")
        assert len(lines) >= 5, f"Formato inesperado de target: {lines}"
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


class DecoupledFieldDataset(Dataset):
    def __init__(self, base_ds: ProceduralCompositionalDataset):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x_0, can_mask, digit_mask = self.base_ds[idx]
        p_b, t_b, _, _ = self.base_ds.samples[idx]
        f_ids = build_field_ids(p_b, t_b, self.base_ds.cfg.max_seq_len, self.base_ds.cfg.use_scratchpad)
        return x_0, can_mask, digit_mask, f_ids


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





def train_optAB_800(cfg: HybridConfig) -> Dict[str, float]:
    print("\n" + "=" * 78)
    print("  INICIANDO ENTRENAMIENTO: OPCIÓN A+B (160 POSICIONES, PÉRDIDAS DESACOPLADAS)")
    print(f"  max_seq_len={cfg.max_seq_len} | use_explicit_operands={cfg.use_explicit_operands} | max_steps={cfg.max_steps}")
    print(f"  loss_weights: pw={cfg.loss_weight_pw}, pms={cfg.loss_weight_pms}, final={cfg.loss_weight_final}, eos={cfg.loss_weight_eos}")
    print("=" * 78)

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)

    train_ds = ProceduralCompositionalDataset(cfg, split="train", num_samples=3200)
    train_loader = DataLoader(DecoupledFieldDataset(train_ds), batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = TernaryRecursiveDiffusionLM(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.configure_adamw_groups(), lr=cfg.learning_rate, betas=cfg.betas)

    step = 0
    train_time_accum = 0.0
    step_times: List[float] = []

    model.train()
    t_start_total = time.perf_counter()

    while step < cfg.max_steps:
        for x_0, can_mask, digit_mask, field_ids in train_loader:
            t_step0 = time.perf_counter()
            step += 1
            lr_now = update_bitnet_schedule(optimizer, step, cfg)
            r_train = cfg.num_recursions

            x_0 = x_0.to(cfg.device)
            can_mask = can_mask.to(cfg.device)
            digit_mask = digit_mask.to(cfg.device)
            field_ids = field_ids.to(cfg.device)

            # Paso 1: Ruido alto t1
            x_t1, mask_pos1, t1 = apply_blockwise_diffusion_corruption(x_0, can_mask, cfg)
            sup_mask1 = mask_pos1 if cfg.lazy_vocab_projection else None
            logits1, inter1, halt1, _, carried1 = model(
                x_t1, t1, num_recursions=r_train, record_telemetry=False, supervised_mask=sup_mask1
            )
            loss1, metrics1 = compute_v4_decoupled_loss(logits1, inter1, halt1, x_0, mask_pos1, digit_mask, t1, cfg, field_ids)

            # Paso 2: Encadenamiento con predicción desacoplada (.detach())
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
                x_t2, t2, delta_t=dt, prev_states=carried1, num_recursions=r_train,
                record_telemetry=False, supervised_mask=sup_mask2
            )
            loss2, metrics2 = compute_v4_decoupled_loss(logits2, inter2, halt2, x_0, keep_mask, digit_mask, t2, cfg, field_ids)

            total_loss = 0.45 * loss1 + 0.55 * loss2
            assert torch.isfinite(total_loss), f"Pérdida no finita en paso {step}: {total_loss.item()}"

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if torch.isfinite(grad_norm):
                optimizer.step()

            elapsed_step = time.perf_counter() - t_step0
            train_time_accum += elapsed_step
            if step > 1:
                step_times.append(elapsed_step * 1000.0)

            if step % 100 == 0 or step == 1:
                ram_now = get_peak_ram_mib()
                ms_step = (train_time_accum / step) * 1000.0
                print(
                    f"  [Opción A+B] Paso {step:03d}/{cfg.max_steps} | LR: {lr_now:.2e} | "
                    f"Loss: {total_loss.item():.3f} [Pw: {metrics2['loss_pw']:.2f}, Pms: {metrics2['loss_pms']:.2f}, Fin: {metrics2['loss_final_field']:.2f}] | "
                    f"DigAcc: {metrics2['train_digit_acc']:.2f} (Pw: {metrics2['acc_pw']:.2f}, Pms: {metrics2['acc_pms']:.2f}, Fin: {metrics2['acc_final']:.2f}) | "
                    f"GradNorm: {grad_norm.item():.2f} | Tiempo: {ms_step:.1f} ms/p | RAM: {ram_now:.1f} MiB",
                    flush=True
                )

            if step >= cfg.max_steps:
                break

    total_wall_s = time.perf_counter() - t_start_total

    # Guardar checkpoint
    checkpoint_dict = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "config": cfg,
        "variant_name": "OpcionAB_160",
    }
    torch.save(checkpoint_dict, cfg.checkpoint_path)
    packed_bytes = model.export_bitcos_archive(cfg.bitcos_export_path)

    steady_step_ms = sum(step_times) / max(1, len(step_times))
    print(
        f"  [Guardado Opción A+B] Checkpoint: {cfg.checkpoint_path} | BITCOSv2: {packed_bytes/1024:.1f} KiB | "
        f"Tiempo Total: {total_wall_s:.1f} s (~{total_wall_s/60.0:.1f} min) | Régimen: {steady_step_ms:.1f} ms/paso"
    )

    return {
        "total_wall_s": total_wall_s,
        "steady_step_ms": steady_step_ms,
        "peak_ram_mib": get_peak_ram_mib(),
        "final_loss": float(total_loss.item()),
        "bitcos_kib": packed_bytes / 1024.0,
    }


def evaluate_variant(
    cfg: HybridConfig,
    variant_name: str,
    train_records_96: List[Tuple[bytes, bytes, int, int]],
    heldout_records_96: List[Tuple[bytes, bytes, int, int]],
) -> Dict[str, Dict[str, float]]:
    print(f"\n" + "=" * 78)
    print(f"  EVALUACIÓN PRINCIPAL (R=2): {variant_name.upper()}")
    print(f"  Checkpoint: {cfg.checkpoint_path} | Dataset: Opción A+B")
    print("=" * 78)

    model = TernaryRecursiveDiffusionLM(cfg).to(cfg.device)
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.load_bitcos_archive(cfg.bitcos_export_path)
    model.prepare_for_eval()

    splits = [
        ("Train (Muestra fija de 96 casos)", train_records_96),
        ("Held-Out OOD (96 casos reservados)", heldout_records_96),
    ]

    results_by_split = {}

    for split_label, records in splits:
        n = len(records)
        print(f"\n--- Evaluando {split_label} ({n} casos) ---")

        # 1. Reconstrucción Estática (Masked Argmax)
        ds = ProceduralCompositionalDataset(cfg, split="train" if "Train" in split_label else "val", num_samples=96)
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        rec_dig_ok, rec_dig_tot = 0, 0
        with torch.no_grad():
            for x_0, can_mask, digit_mask in loader:
                x_t, m_pos, t_vec = apply_blockwise_diffusion_corruption(x_0, can_mask, cfg, fixed_t=0.50)
                out = model(x_t, t_vec, num_recursions=2, supervised_mask=m_pos if cfg.lazy_vocab_projection else None)
                if out.final_logits.ndim == 2:
                    p_sup = out.final_logits.argmax(dim=-1)
                    c_sup = (p_sup == x_0[m_pos])
                    d_sup = digit_mask[m_pos]
                    rec_dig_ok += int((c_sup & d_sup).sum().item())
                    rec_dig_tot += int(d_sup.sum().item())
                else:
                    p = out.final_logits.argmax(dim=-1)
                    m_d = m_pos & digit_mask
                    rec_dig_ok += int(((p == x_0) & m_d).sum().item())
                    rec_dig_tot += int(m_d.sum().item())
        reconst_dig_acc = 100.0 * rec_dig_ok / max(1, rec_dig_tot)

        # 2. Generación Libre con Muestreador Corregido
        c1_strict_valid, c1_strict_em = 0, 0
        c1_diag_exclusive, c1_diag_em = 0, 0
        c1_pw_em, c1_pms_em = 0, 0
        c1_pw_dig_c, c1_pw_dig_t = 0, 0
        c1_pms_dig_c, c1_pms_dig_t = 0, 0

        c2_strict_valid, c2_strict_em = 0, 0
        c2_diag_exclusive, c2_diag_em = 0, 0
        c2_w_em, c2_ms_em = 0, 0
        c2_w_dig_c, c2_w_dig_t = 0, 0
        c2_ms_dig_c, c2_ms_dig_t = 0, 0

        c4_full_strict_em = 0
        c4_full_diag_em = 0

        for prompt_b, target_b, tw, tms in records:
            p_str = prompt_b.decode("utf-8")
            t_str = target_b.decode("utf-8")
            _, tr_pw, tr_pms, _, _ = parse_scratchpad_prediction(t_str, "")

            # C1: Subtotales (gen_length=24)
            gen1, _, _ = generate_block_causal_corrected(
                model, p_str, cfg, gen_length=24, num_recursions_override=2, restrict_vocab=True
            )
            cont1 = gen1[len(p_str):] if gen1.startswith(p_str) else gen1
            st1, p1_pw, p1_pms = parse_subtotals_dual(cont1)

            pw_ok = (p1_pw == tr_pw)
            pms_ok = (p1_pms == tr_pms)
            sub_both = (pw_ok and pms_ok)

            if pw_ok: c1_pw_em += 1
            if pms_ok: c1_pms_em += 1

            if st1 == "STRICT_VALID":
                c1_strict_valid += 1
                if sub_both: c1_strict_em += 1
            elif st1 == "UNAMBIGUOUS_RECOVERABLE":
                c1_diag_exclusive += 1
                if sub_both: c1_diag_em += 1

            m_pw = re.search(r"pw[:_\s]*(\d+)", cont1, re.I)
            if m_pw:
                c, t = compute_isolated_digit_acc(m_pw.group(1), str(tr_pw))
                c1_pw_dig_c += c; c1_pw_dig_t += t
            else:
                c1_pw_dig_t += len(str(tr_pw))
            m_pms = re.search(r"pms[:_\s]*(\d+)", cont1, re.I)
            if m_pms:
                c, t = compute_isolated_digit_acc(m_pms.group(1), str(tr_pms))
                c1_pms_dig_c += c; c1_pms_dig_t += t
            else:
                c1_pms_dig_t += len(str(tr_pms))

            # C2: Factores + Subtotales Oráculo -> Final (gen_length=24)
            prompt_c2 = p_str + f"pw: {tr_pw}\npms: {tr_pms}\n"
            gen2, _, _ = generate_block_causal_corrected(
                model, prompt_c2, cfg, gen_length=24, num_recursions_override=2, restrict_vocab=True
            )
            cont2 = gen2[len(prompt_c2):] if gen2.startswith(prompt_c2) else gen2
            st2, p2_w, p2_ms = parse_finals_dual(cont2)

            w_ok = (p2_w == tw)
            ms_ok = (p2_ms == tms)
            resp_both = (w_ok and ms_ok)

            if w_ok: c2_w_em += 1
            if ms_ok: c2_ms_em += 1

            if st2 == "STRICT_VALID":
                c2_strict_valid += 1
                if resp_both: c2_strict_em += 1
            elif st2 == "UNAMBIGUOUS_RECOVERABLE":
                c2_diag_exclusive += 1
                if resp_both: c2_diag_em += 1

            m_w = re.search(r"watt[:_\s]*(\d+)", cont2, re.I)
            if m_w:
                c, t = compute_isolated_digit_acc(m_w.group(1), str(tw))
                c2_w_dig_c += c; c2_w_dig_t += t
            else:
                c2_w_dig_t += len(str(tw))
            m_ms = re.search(r"ms[:_\s]*(\d+)", cont2, re.I)
            if m_ms:
                c, t = compute_isolated_digit_acc(m_ms.group(1), str(tms))
                c2_ms_dig_c += c; c2_ms_dig_t += t
            else:
                c2_ms_dig_t += len(str(tms))

            # C4: Cadena Completa Integrada (gen_length=48)
            gen4, _, _ = generate_block_causal_corrected(
                model, p_str, cfg, gen_length=48, num_recursions_override=2, restrict_vocab=True
            )
            is_fmt4, p4_sw, p4_sms, p4_w, p4_ms = parse_scratchpad_prediction(gen4, p_str)
            all_correct = (p4_sw == tr_pw and p4_sms == tr_pms and p4_w == tw and p4_ms == tms)
            if is_fmt4 and all_correct:
                c4_full_strict_em += 1
            elif all_correct:
                c4_full_diag_em += 1

        c1_cobertura = c1_strict_valid + c1_diag_exclusive
        c1_global_em = c1_strict_em + c1_diag_em
        c2_cobertura = c2_strict_valid + c2_diag_exclusive
        c2_global_em = c2_strict_em + c2_diag_em
        c4_global_em = c4_full_strict_em + c4_full_diag_em

        pw_dig_acc = 100.0 * c1_pw_dig_c / max(1, c1_pw_dig_t)
        pms_dig_acc = 100.0 * c1_pms_dig_c / max(1, c1_pms_dig_t)
        w_dig_acc = 100.0 * c2_w_dig_c / max(1, c2_w_dig_t)
        ms_dig_acc = 100.0 * c2_ms_dig_c / max(1, c2_ms_dig_t)

        print(f"  Reconstrucción Estática DigAcc: {reconst_dig_acc:.1f}%")
        print(f"  C1 (Subtotales):")
        print(f"    - Cobertura Total: {c1_cobertura}/{n} ({100.0*c1_cobertura/n:.1f}%) [Estricto: {c1_strict_valid}, Recup. Diagnóstico Exclusivo: {c1_diag_exclusive}]")
        print(f"    - Subtotal EM Global (Pw y Pms): {c1_global_em}/{n} ({100.0*c1_global_em/n:.1f}%) [Estricto: {c1_strict_em}, Diagnóstico: {c1_diag_em}]")
        print(f"    - Desglose EM: Pw EM = {c1_pw_em}/{n} ({100.0*c1_pw_em/n:.1f}%), Pms EM = {c1_pms_em}/{n} ({100.0*c1_pms_em/n:.1f}%)")
        print(f"    - DigAcc Libre: Pw DigAcc = {pw_dig_acc:.1f}%, Pms DigAcc = {pms_dig_acc:.1f}%")
        print(f"  C2 (Oráculo -> Respuesta Final):")
        print(f"    - Cobertura Total: {c2_cobertura}/{n} ({100.0*c2_cobertura/n:.1f}%) [Estricto: {c2_strict_valid}, Recup. Diagnóstico Exclusivo: {c2_diag_exclusive}]")
        print(f"    - Respuesta Final EM Global (Watt y MS): {c2_global_em}/{n} ({100.0*c2_global_em/n:.1f}%) [Estricto: {c2_strict_em}, Diagnóstico: {c2_diag_em}]")
        print(f"    - Desglose EM: Watt EM = {c2_w_em}/{n} ({100.0*c2_w_em/n:.1f}%), MS EM = {c2_ms_em}/{n} ({100.0*c2_ms_em/n:.1f}%)")
        print(f"    - DigAcc Libre: Watt DigAcc = {w_dig_acc:.1f}%, MS DigAcc = {ms_dig_acc:.1f}%")
        print(f"  C4 (Cadena Completa Integrada):")
        print(f"    - Full Chain EM Global: {c4_global_em}/{n} ({100.0*c4_global_em/n:.1f}%) [Estricto: {c4_full_strict_em}, Diagnóstico: {c4_full_diag_em}]")

        split_key = "train" if "Train" in split_label else "heldout"
        results_by_split[split_key] = {
            "reconst_dig_acc": reconst_dig_acc,
            "c1_cobertura": c1_cobertura,
            "c1_strict_valid": c1_strict_valid,
            "c1_diag_exclusive": c1_diag_exclusive,
            "c1_global_em": c1_global_em,
            "c1_strict_em": c1_strict_em,
            "c1_diag_em": c1_diag_em,
            "c1_pw_em": c1_pw_em,
            "c1_pms_em": c1_pms_em,
            "c1_pw_dig_acc": pw_dig_acc,
            "c1_pms_dig_acc": pms_dig_acc,
            "c2_cobertura": c2_cobertura,
            "c2_strict_valid": c2_strict_valid,
            "c2_diag_exclusive": c2_diag_exclusive,
            "c2_global_em": c2_global_em,
            "c2_strict_em": c2_strict_em,
            "c2_diag_em": c2_diag_em,
            "c2_w_em": c2_w_em,
            "c2_ms_em": c2_ms_em,
            "c2_w_dig_acc": w_dig_acc,
            "c2_ms_dig_acc": ms_dig_acc,
            "c4_global_em": c4_global_em,
            "c4_full_strict_em": c4_full_strict_em,
            "c4_full_diag_em": c4_full_diag_em,
        }

    return results_by_split


def run_pipeline():
    print("=" * 80)
    print("  EXPERIMENTO CONTROLADO: VARIANTE OPCIÓN A+B (PÉRDIDAS DESACOPLADAS)")
    print("=" * 80)

    cfg_setup = HybridConfig(max_seq_len=160, use_explicit_operands=True)
    train_ds_ref = ProceduralCompositionalDataset(cfg_setup, split="train", num_samples=384)
    val_ds_ref = ProceduralCompositionalDataset(cfg_setup, split="val", num_samples=96)

    rng = random.Random(42)
    train_indices = rng.sample(range(len(train_ds_ref.unique_records)), 96)

    optAB_train_96 = [train_ds_ref.unique_records[i] for i in train_indices]
    optAB_val_96 = list(val_ds_ref.unique_records)

    cfg_optAB = HybridConfig(
        max_seq_len=160,
        use_scratchpad=True,
        use_explicit_operands=True,
        max_steps=800,
        checkpoint_path="trdm_v4_optAB_checkpoint.pt",
        bitcos_export_path="trdm_v4_optAB_bitcos.bin",
    )
    cfg_optAB.use_decoupled_field_loss = True
    cfg_optAB.loss_weight_pw = 1.0
    cfg_optAB.loss_weight_pms = 1.0
    cfg_optAB.loss_weight_final = 1.0
    cfg_optAB.loss_weight_eos = 0.2

    # 1. Entrenar Opción A+B
    stats_optAB = train_optAB_800(cfg_optAB)

    # 2. Evaluar Opción A+B
    eval_optAB = evaluate_variant(cfg_optAB, "OpcionAB_160", optAB_train_96, optAB_val_96)

    # 3. Cuadro comparativo final
    print("\n" + "=" * 80)
    print("  RESULTADOS DE OPCIÓN A+B CONCLUIDOS CON ÉXITO")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
