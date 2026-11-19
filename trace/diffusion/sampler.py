# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

import math
import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from trace.data.parsers import parse_scratchpad_prediction


@torch.no_grad()
def generate_block_causal_with_kv_cache(
    model,
    prompt_text: str,
    cfg,
    gen_length: Optional[int] = None,
    num_recursions_override: Optional[int] = None,
) -> Tuple[str, List[str], int]:
    """
    Decodificación no autorregresiva Block-Causal con reutilización estricta de caché KV.
    Procesa la secuencia bloque a bloque (tamaño B_blk=16) mediante difusión discreta local.
    """
    model.eval()
    R_infer = num_recursions_override if num_recursions_override is not None else cfg.num_recursions
    prompt_bytes = list(prompt_text.encode("utf-8"))

    if gen_length is None:
        gen_length = 48 if getattr(cfg, "use_scratchpad", False) else 24

    seq = [cfg.bos_token_id] + prompt_bytes + [cfg.mask_token_id] * gen_length
    seq = seq[: cfg.max_seq_len]
    B_blk = cfg.block_size
    if len(seq) % B_blk != 0:
        pad_to_blk = B_blk - (len(seq) % B_blk)
        seq = (seq + [cfg.mask_token_id] * pad_to_blk)[: cfg.max_seq_len]

    total_len = len(seq)
    x_full = torch.tensor([seq], dtype=torch.long, device=cfg.device)
    num_blocks = total_len // B_blk

    kv_caches_committed = None
    committed_len = 0
    history: List[str] = []
    total_block_evals = 0

    for b_idx in range(num_blocks):
        b_start = b_idx * B_blk
        b_end = (b_idx + 1) * B_blk
        block_tokens = x_full[:, b_start:b_end].clone()

        if not (block_tokens == cfg.mask_token_id).any():
            t_zero = torch.zeros((1, B_blk), device=cfg.device)
            dt_zero = torch.zeros((1, B_blk), device=cfg.device)
            _, _, _, kv_caches_committed, _ = model(
                block_tokens,
                t_zero,
                delta_t=dt_zero,
                num_recursions=R_infer,
                kv_caches_per_r=kv_caches_committed,
                start_pos=committed_len,
            )
            committed_len = b_end
            total_block_evals += R_infer
            continue

        steps = cfg.diffusion_steps_per_block
        editable_positions = (block_tokens[0] == cfg.mask_token_id).clone()
        initial_masks = int(editable_positions.sum().item())
        carried_states = None
        prev_ratio = 1.0

        for s_idx in range(steps):
            masked_pos = (block_tokens[0] == cfg.mask_token_id)
            num_masked = int(masked_pos.sum().item())
            if num_masked == 0:
                break

            mask_ratio = num_masked / max(1, initial_masks)
            t_tensor = torch.full((1, B_blk), mask_ratio, device=cfg.device)
            t_tensor[0, ~masked_pos] = 0.0
            dt_val = max(0.01, prev_ratio - mask_ratio)
            dt_tensor = torch.full((1, B_blk), dt_val, device=cfg.device)
            prev_ratio = mask_ratio

            logits, inter_logits, _, _, carried_states = model(
                block_tokens,
                t_tensor,
                delta_t=dt_tensor,
                prev_states=carried_states,
                num_recursions=R_infer,
                kv_caches_per_r=kv_caches_committed,
                start_pos=committed_len,
            )
            total_block_evals += len(inter_logits)

            logits[:, :, cfg.mask_token_id] = -1e9
            logits[:, :, cfg.bos_token_id] = -1e9

            # PAD solo es válido en posiciones estrictamente posteriores a un token EOS
            eos_in_past = (x_full[:, :b_start] == cfg.eos_token_id).any().item()
            if not eos_in_past:
                is_eos_in_block = (block_tokens[0] == cfg.eos_token_id)
                can_be_pad = torch.zeros(B_blk, dtype=torch.bool, device=cfg.device)
                if is_eos_in_block.any():
                    first_eos_idx = torch.nonzero(is_eos_in_block, as_tuple=False)[0].item()
                    if first_eos_idx + 1 < B_blk:
                        can_be_pad[first_eos_idx + 1:] = True
                logits[0, ~can_be_pad, cfg.pad_token_id] = -1e9

            probs = F.softmax(logits[0], dim=-1)
            confidences, best_tokens = probs.max(dim=-1)

            editable_indices = torch.nonzero(editable_positions, as_tuple=False).squeeze(-1)
            progress = (s_idx + 1) / steps
            target_masks_rem = int(math.floor(initial_masks * math.cos(progress * math.pi / 2.0)))

            if s_idx == steps - 1 or target_masks_rem == 0:
                block_tokens[0, editable_indices] = best_tokens[editable_indices]
            elif getattr(cfg, "use_dynamic_remasking", True):
                # Re-enmascaramiento dinámico estilo LLaDA/CDLM
                num_to_keep = max(1, len(editable_indices) - target_masks_rem)
                edit_conf = confidences[editable_indices]
                _, top_order = torch.topk(edit_conf, k=min(num_to_keep, len(editable_indices)), largest=True)
                keep_pos = editable_indices[top_order]

                block_tokens[0, editable_indices] = cfg.mask_token_id
                block_tokens[0, keep_pos] = best_tokens[keep_pos]
            else:
                masked_indices = torch.nonzero(masked_pos, as_tuple=False).squeeze(-1)
                num_to_reveal = max(1, len(masked_indices) - target_masks_rem)
                masked_conf = confidences[masked_indices]
                _, top_reveal_order = torch.topk(masked_conf, k=min(num_to_reveal, len(masked_indices)), largest=True)
                chosen_positions = masked_indices[top_reveal_order]
                block_tokens[0, chosen_positions] = best_tokens[chosen_positions]

            x_full[:, b_start:b_end] = block_tokens

            raw_bytes = bytearray()
            for tok in x_full[0, 1:total_len].tolist():
                if tok in (cfg.eos_token_id, cfg.pad_token_id):
                    break
                if tok < 256:
                    raw_bytes.append(tok)
            history.append(raw_bytes.decode("utf-8", errors="replace"))

        if (block_tokens[0] == cfg.eos_token_id).any() or (block_tokens[0] == cfg.pad_token_id).any():
            break

        t_zero = torch.zeros((1, B_blk), device=cfg.device)
        dt_zero = torch.zeros((1, B_blk), device=cfg.device)
        _, _, _, kv_caches_committed, _ = model(
            block_tokens,
            t_zero,
            delta_t=dt_zero,
            num_recursions=R_infer,
            kv_caches_per_r=kv_caches_committed,
            start_pos=committed_len,
        )
        committed_len = b_end
        total_block_evals += R_infer

    final_text = history[-1] if history else prompt_text
    return final_text, history, total_block_evals


def evaluate_exact_match_split(
    model,
    records: List[Tuple[bytes, bytes, int, int]],
    cfg,
    r_infer: int,
) -> Dict[str, float]:
    n = len(records)
    fmt_ok, watt_em, ms_em, both_em, dig_correct, dig_total = 0, 0, 0, 0, 0, 0
    sub_w_em, sub_ms_em, sub_both_em, all_em = 0, 0, 0, 0
    has_scratchpad_data = False
    t0 = time.perf_counter()

    for prompt_b, target_b, true_watt, true_ms in records:
        prompt_str = prompt_b.decode("utf-8")
        target_str = target_b.decode("utf-8")
        gen_text, _, _ = generate_block_causal_with_kv_cache(
            model, prompt_str, cfg, num_recursions_override=r_infer
        )
        is_valid_fmt, pred_sub_w, pred_sub_ms, pred_w, pred_ms = parse_scratchpad_prediction(
            gen_text, prompt_str
        )
        if is_valid_fmt:
            fmt_ok += 1
        if pred_w == true_watt:
            watt_em += 1
        if pred_ms == true_ms:
            ms_em += 1
        if pred_w == true_watt and pred_ms == true_ms:
            both_em += 1

        # Si el ground-truth contiene subtotales
        _, true_sub_w, true_sub_ms, _, _ = parse_scratchpad_prediction(target_str, "")
        if true_sub_w is not None and true_sub_ms is not None:
            has_scratchpad_data = True
            sw_ok = (pred_sub_w == true_sub_w)
            sms_ok = (pred_sub_ms == true_sub_ms)
            if sw_ok:
                sub_w_em += 1
            if sms_ok:
                sub_ms_em += 1
            if sw_ok and sms_ok:
                sub_both_em += 1
            if sw_ok and sms_ok and (pred_w == true_watt) and (pred_ms == true_ms):
                all_em += 1

        # Exactitud posicional exclusiva en caracteres de dígitos
        continuation = gen_text[len(prompt_str):] if gen_text.startswith(prompt_str) else ""
        for idx_c, ch in enumerate(target_str):
            if ch.isdigit():
                dig_total += 1
                if idx_c < len(continuation) and continuation[idx_c] == ch:
                    dig_correct += 1

    latency_ms = ((time.perf_counter() - t0) / max(1, n)) * 1000.0
    p_both = both_em / max(1, n)
    se_both = math.sqrt(p_both * (1.0 - p_both) / max(1, n))

    res = {
        "n": n,
        "fmt_pct": 100.0 * fmt_ok / max(1, n),
        "watt_em_pct": 100.0 * watt_em / max(1, n),
        "ms_em_pct": 100.0 * ms_em / max(1, n),
        "both_em_pct": 100.0 * p_both,
        "both_se_pct": 100.0 * se_both,
        "digit_acc_pct": 100.0 * dig_correct / max(1, dig_total),
        "latency_ms": latency_ms,
    }
    if has_scratchpad_data:
        res["sub_w_em_pct"] = 100.0 * sub_w_em / max(1, n)
        res["sub_ms_em_pct"] = 100.0 * sub_ms_em / max(1, n)
        res["sub_both_em_pct"] = 100.0 * sub_both_em / max(1, n)
        res["all_em_pct"] = 100.0 * all_em / max(1, n)

    return res
