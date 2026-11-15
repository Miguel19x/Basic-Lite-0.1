import math
import time
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F

from config import HybridConfig
from model import TernaryRecursiveDiffusionLM
from data import ProceduralCompositionalDataset, parse_watt_ms_prediction


@torch.no_grad()
def generate_block_causal_with_kv_cache(
    model: TernaryRecursiveDiffusionLM,
    prompt_text: str,
    cfg: HybridConfig,
    gen_length: int = 24,
    num_recursions_override: int = None,
) -> Tuple[str, List[str], int]:
    model.eval()
    R_infer = num_recursions_override if num_recursions_override is not None else cfg.num_recursions
    prompt_bytes = list(prompt_text.encode("utf-8"))

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
            t_zero = torch.tensor([0.0], device=cfg.device)
            dt_zero = torch.tensor([0.0], device=cfg.device)
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
            mask_pos = (block_tokens[0] == cfg.mask_token_id)
            num_masked = int(mask_pos.sum().item())
            if num_masked == 0:
                break

            mask_ratio = num_masked / max(1, initial_masks)
            t_tensor = torch.tensor([mask_ratio], device=cfg.device)
            dt_tensor = torch.tensor([max(0.01, prev_ratio - mask_ratio)], device=cfg.device)
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
            probs = F.softmax(logits[0], dim=-1)
            confidences, best_tokens = probs.max(dim=-1)

            edit_indices = torch.nonzero(editable_positions, as_tuple=False).squeeze(-1)
            block_tokens[0, edit_indices] = best_tokens[edit_indices]

            progress = (s_idx + 1) / steps
            target_rem = int(math.floor(initial_masks * math.cos(progress * math.pi / 2.0)))
            if s_idx < steps - 1 and target_rem > 0:
                edit_conf = confidences[edit_indices]
                _, lowest_order = torch.topk(edit_conf, k=min(target_rem, len(edit_indices)), largest=False)
                remask_pos = edit_indices[lowest_order]
                block_tokens[0, remask_pos] = cfg.mask_token_id

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

        t_zero = torch.tensor([0.0], device=cfg.device)
        dt_zero = torch.tensor([0.0], device=cfg.device)
        _, _, _, kv_caches_committed, _ = model(
            block_tokens,
            t_zero,
            delta_t=dt_zero,
            num_recursions=R_infer,
            kv_caches_per_r=kv_caches_committed,
            start_pos=committed_len,
        )
        committed_len = b_end

    final_text = history[-1] if history else prompt_text
    return final_text, history, total_block_evals


def evaluate_exact_match_split(
    model: TernaryRecursiveDiffusionLM,
    records: List[Tuple[bytes, bytes, int, int]],
    cfg: HybridConfig,
    r_infer: int,
) -> Dict[str, float]:
    n = len(records)
    fmt_ok, watt_em, ms_em, both_em, dig_correct, dig_total = 0, 0, 0, 0, 0, 0
    t0 = time.perf_counter()

    for prompt_b, target_b, true_watt, true_ms in records:
        prompt_str = prompt_b.decode("utf-8")
        target_str = target_b.decode("utf-8")
        gen_text, _, _ = generate_block_causal_with_kv_cache(
            model, prompt_str, cfg, num_recursions_override=r_infer
        )
        is_valid_fmt, pred_w, pred_ms = parse_watt_ms_prediction(gen_text, prompt_str)
        if is_valid_fmt:
            fmt_ok += 1
        if pred_w == true_watt:
            watt_em += 1
        if pred_ms == true_ms:
            ms_em += 1
        if pred_w == true_watt and pred_ms == true_ms:
            both_em += 1

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

    return {
        "n": n,
        "fmt_pct": 100.0 * fmt_ok / max(1, n),
        "watt_em_pct": 100.0 * watt_em / max(1, n),
        "ms_em_pct": 100.0 * ms_em / max(1, n),
        "both_em_pct": 100.0 * p_both,
        "both_se_pct": 100.0 * se_both,
        "digit_acc_pct": 100.0 * dig_correct / max(1, dig_total),
        "latency_ms": latency_ms,
    }


if __name__ == "__main__":
    cfg = HybridConfig()
    model = TernaryRecursiveDiffusionLM(cfg)
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.load_bitcos_archive(cfg.bitcos_export_path)
    model.prepare_for_eval()

    train_ds = ProceduralCompositionalDataset(cfg, split="train", num_samples=100)
    val_ds = ProceduralCompositionalDataset(cfg, split="val", num_samples=100)

    # 1. Prueba de memorización sobre 96 combinaciones de entrenamiento
    train_subset = train_ds.unique_records[:96]
    m_train = evaluate_exact_match_split(model, train_subset, cfg, r_infer=cfg.num_recursions)
    print(f"=== 1. Control de Memorización en Entrenamiento (96 tuplas Train, R={cfg.num_recursions}) ===")
    print(
        f"Formato: {m_train['fmt_pct']:.1f}% | DigAcc: {m_train['digit_acc_pct']:.1f}% | "
        f"Watt EM: {m_train['watt_em_pct']:.1f}% | MS EM: {m_train['ms_em_pct']:.1f}% | "
        f"Full Tuple EM: {m_train['both_em_pct']:.1f}% (±{m_train['both_se_pct']:.1f}%)"
    )

    # 2. Evaluación rigurosa de la Hipótesis H1 sobre las 96 combinaciones Held-Out completas
    print("\n=== 2. Evaluación Hipótesis H1 en las 96 Tuplas Reservadas (Held-Out OOD) ===")
    print(f"{'R_infer':<8} | {'Fmt %':<7} | {'DigAcc %':<9} | {'Watt EM %':<10} | {'MS EM %':<8} | {'Full EM % (±SE)':<18} | {'Latencia':<9}")
    print("-" * 82)
    for r_val in (1, 2, 4, 8):
        m_val = evaluate_exact_match_split(model, val_ds.unique_records, cfg, r_infer=r_val)
        em_str = f"{m_val['both_em_pct']:.1f}% (±{m_val['both_se_pct']:.1f}%)"
        print(
            f"R = {r_val:<4} | {m_val['fmt_pct']:>5.1f}% | {m_val['digit_acc_pct']:>7.1f}% | "
            f"{m_val['watt_em_pct']:>8.1f}% | {m_val['ms_em_pct']:>6.1f}% | {em_str:<18} | {m_val['latency_ms']:>6.1f} ms"
        )
