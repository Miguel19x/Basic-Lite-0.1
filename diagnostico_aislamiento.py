import math
import re
import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from config import HybridConfig
from model import TernaryRecursiveDiffusionLM
from data import ProceduralCompositionalDataset, parse_watt_ms_prediction


@torch.no_grad()
def run_diffusion_on_sequence(
    model: TernaryRecursiveDiffusionLM,
    x_full: torch.Tensor,
    cfg: HybridConfig,
    r_infer: int,
    editable_mask: Optional[torch.Tensor] = None,
    steps_override: Optional[int] = None,
) -> torch.Tensor:
    """
    Ejecuta el pipeline de difusión causal por bloques.
    Si se pasa editable_mask (forma 1, L), únicamente las posiciones en True
    serán editables; las demás permanecen congeladas en su valor actual.
    """
    model.eval()
    B_blk = cfg.block_size
    total_len = x_full.size(1)
    num_blocks = total_len // B_blk

    kv_caches_committed = None
    committed_len = 0

    for b_idx in range(num_blocks):
        b_start = b_idx * B_blk
        b_end = (b_idx + 1) * B_blk
        block_tokens = x_full[:, b_start:b_end].clone()

        # Determinar posiciones editables en este bloque
        if editable_mask is not None:
            blk_editable = editable_mask[:, b_start:b_end] & (block_tokens == cfg.mask_token_id)
        else:
            blk_editable = (block_tokens == cfg.mask_token_id)

        if not blk_editable.any():
            t_zero = torch.zeros((1, B_blk), device=cfg.device)
            dt_zero = torch.zeros((1, B_blk), device=cfg.device)
            _, _, _, kv_caches_committed, _ = model(
                block_tokens,
                t_zero,
                delta_t=dt_zero,
                num_recursions=r_infer,
                kv_caches_per_r=kv_caches_committed,
                start_pos=committed_len,
            )
            committed_len = b_end
            continue

        steps = steps_override if steps_override is not None else cfg.diffusion_steps_per_block
        editable_positions = blk_editable[0].clone()
        initial_masks = int(editable_positions.sum().item())
        carried_states = None
        prev_ratio = 1.0

        for s_idx in range(steps):
            masked_pos = (block_tokens[0] == cfg.mask_token_id) & editable_positions
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
                num_recursions=r_infer,
                kv_caches_per_r=kv_caches_committed,
                start_pos=committed_len,
            )

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

        if (block_tokens[0] == cfg.eos_token_id).any() or (block_tokens[0] == cfg.pad_token_id).any():
            break

        t_zero = torch.zeros((1, B_blk), device=cfg.device)
        dt_zero = torch.zeros((1, B_blk), device=cfg.device)
        _, _, _, kv_caches_committed, _ = model(
            block_tokens,
            t_zero,
            delta_t=dt_zero,
            num_recursions=r_infer,
            kv_caches_per_r=kv_caches_committed,
            start_pos=committed_len,
        )
        committed_len = b_end

    return x_full


def decode_tokens_to_text(x_full: torch.Tensor, cfg: HybridConfig) -> str:
    total_len = x_full.size(1)
    raw_bytes = bytearray()
    for tok in x_full[0, 1:total_len].tolist():
        if tok in (cfg.eos_token_id, cfg.pad_token_id):
            break
        if tok < 256:
            raw_bytes.append(tok)
    return raw_bytes.decode("utf-8", errors="replace")


def compute_strictly_hidden_digit_acc(
    gen_continuation: str,
    target_str: str,
    condition: str,
) -> Tuple[int, int]:
    """
    AUDITORÍA RIGUROSA DEL DENOMINADOR:
    Calcula (dig_correct, dig_total) EXCLUSIVAMENTE sobre las posiciones originalmente ocultas.
    Descarta por completo cualquier posición que haya sido fijada o entregada como contexto.
    """
    match_t = re.match(r"^watt:\s*(\d+)\s*\nms:\s*(\d+)\s*$", target_str.strip())
    if not match_t:
        return 0, 0

    w_str = match_t.group(1)
    ms_str = match_t.group(2)

    w_start = target_str.find(w_str)
    w_end = w_start + len(w_str)

    ms_start = target_str.find(ms_str, w_end)
    ms_end = ms_start + len(ms_str)

    ranges_to_eval = []
    if condition in ("free", "clamped_template", "staggered_w_then_ms", "staggered_ms_then_w"):
        ranges_to_eval = [(w_start, w_end), (ms_start, ms_end)]
    elif condition == "known_watt":
        # Watt estaba fijado; evaluar ÚNICAMENTE los dígitos de MS
        ranges_to_eval = [(ms_start, ms_end)]
    elif condition == "known_ms":
        # MS estaba fijado; evaluar ÚNICAMENTE los dígitos de Watt
        ranges_to_eval = [(w_start, w_end)]

    correct, total = 0, 0
    for start, end in ranges_to_eval:
        for idx in range(start, end):
            ch = target_str[idx]
            if ch.isdigit():
                total += 1
                if idx < len(gen_continuation) and gen_continuation[idx] == ch:
                    correct += 1

    return correct, total


def run_staggered_generation(
    model: TernaryRecursiveDiffusionLM,
    prompt_bytes: bytes,
    cfg: HybridConfig,
    r_infer: int,
    order: str = "w_then_ms",
) -> str:
    """
    DIFUSIÓN ESCALONADA POR CAMPOS (SIN valores verdaderos):
    1. Fase 1: Genera el campo 1 en 2 pasos de difusión; el campo 2 permanece enmascarado.
    2. Fase 2: Congela la PREDICCIÓN generada por la red para el campo 1 (sea correcta o errónea).
    3. Fase 3: Desruida el campo 2 en 2 pasos de difusión condicionado a su propia predicción.
    Costo total: 4 pasos (idéntico a la difusión simultánea estándar).
    """
    bos = [cfg.bos_token_id]
    eos = [cfg.eos_token_id]
    p_bytes = list(prompt_bytes)

    # Longitudes representativas para la plantilla fija (3 dígitos watt, 2 dígitos ms)
    # Sin filtrar longitudes verdaderas: siempre usa el mismo tamaño de slot
    w_slot = 3
    ms_slot = 2

    seq = (
        bos
        + p_bytes
        + list(b"watt: ")
        + [cfg.mask_token_id] * w_slot
        + list(b"\nms: ")
        + [cfg.mask_token_id] * ms_slot
        + list(b"\n")
        + eos
    )
    B_blk = cfg.block_size
    if len(seq) % B_blk != 0:
        pad_needed = B_blk - (len(seq) % B_blk)
        seq = (seq + [cfg.pad_token_id] * pad_needed)[: cfg.max_seq_len]

    x_full = torch.tensor([seq], dtype=torch.long, device=cfg.device)

    # Identificar índices de los slots de watt y ms
    w_start = 1 + len(p_bytes) + len(b"watt: ")
    w_end = w_start + w_slot
    ms_start = w_end + len(b"\nms: ")
    ms_end = ms_start + ms_slot

    mask_w = torch.zeros_like(x_full, dtype=torch.bool)
    mask_w[0, w_start:w_end] = True

    mask_ms = torch.zeros_like(x_full, dtype=torch.bool)
    mask_ms[0, ms_start:ms_end] = True

    if order == "w_then_ms":
        # Fase 1: Solo desruidar watt (2 pasos)
        x_full = run_diffusion_on_sequence(model, x_full, cfg, r_infer, editable_mask=mask_w, steps_override=2)
        # Fase 2: Desruidar ms condicionado al watt auto-predicho (2 pasos)
        x_full = run_diffusion_on_sequence(model, x_full, cfg, r_infer, editable_mask=mask_ms, steps_override=2)
    elif order == "ms_then_w":
        # Fase 1: Solo desruidar ms (2 pasos)
        x_full = run_diffusion_on_sequence(model, x_full, cfg, r_infer, editable_mask=mask_ms, steps_override=2)
        # Fase 2: Desruidar watt condicionado al ms auto-predicho (2 pasos)
        x_full = run_diffusion_on_sequence(model, x_full, cfg, r_infer, editable_mask=mask_w, steps_override=2)
    else:
        raise ValueError(f"Orden desconocido: {order}")

    return decode_tokens_to_text(x_full, cfg)


def build_clamped_sequence(
    prompt_bytes: bytes,
    cfg: HybridConfig,
    condition: str,
    true_watt: int,
    true_ms: int,
) -> torch.Tensor:
    bos = [cfg.bos_token_id]
    eos = [cfg.eos_token_id]
    p_bytes = list(prompt_bytes)

    if condition == "free":
        seq = bos + p_bytes + [cfg.mask_token_id] * 24
    elif condition == "clamped_template":
        w_len = len(str(true_watt))
        ms_len = len(str(true_ms))
        seq = (
            bos
            + p_bytes
            + list(b"watt: ")
            + [cfg.mask_token_id] * w_len
            + list(b"\nms: ")
            + [cfg.mask_token_id] * ms_len
            + list(b"\n")
            + eos
        )
    elif condition == "known_watt":
        ms_len = len(str(true_ms))
        target_known_w = f"watt: {true_watt}\nms: ".encode("utf-8")
        seq = (
            bos
            + p_bytes
            + list(target_known_w)
            + [cfg.mask_token_id] * ms_len
            + list(b"\n")
            + eos
        )
    elif condition == "known_ms":
        w_len = len(str(true_watt))
        seq = (
            bos
            + p_bytes
            + list(b"watt: ")
            + [cfg.mask_token_id] * w_len
            + list(f"\nms: {true_ms}\n".encode("utf-8"))
            + eos
        )
    else:
        raise ValueError(f"Condición desconocida: {condition}")

    B_blk = cfg.block_size
    seq = seq[: cfg.max_seq_len]
    if len(seq) % B_blk != 0:
        pad_needed = B_blk - (len(seq) % B_blk)
        fill_tok = cfg.mask_token_id if condition == "free" else cfg.pad_token_id
        seq = (seq + [fill_tok] * pad_needed)[: cfg.max_seq_len]

    return torch.tensor([seq], dtype=torch.long, device=cfg.device)


def evaluate_rigorous_suite(
    model: TernaryRecursiveDiffusionLM,
    dataset_records: List[Tuple[bytes, bytes, int, int]],
    cfg: HybridConfig,
    r_infer: int,
) -> None:
    print(f"\n" + "=" * 96)
    print(f"AUDITORÍA METODOLÓGICA FORMAL (N=96 HELD-OUT OOD) | PROFUNDIDAD R = {r_infer}")
    print("=" * 96)

    conditions = [
        ("free", "1. Generación Simultánea Libre (Baseline)"),
        ("clamped_template", "2. Plantilla Fija (Ambos Ocultos)"),
        ("known_watt", "3. Watt Conocido (Solo MS Oculto - Denominador Corregido)"),
        ("known_ms", "4. MS Conocido (Solo Watt Oculto - Denominador Corregido)"),
        ("staggered_w_then_ms", "5. Escalonada W -> MS (Auto-Predicho, Cero Filtración)"),
        ("staggered_ms_then_w", "6. Escalonada MS -> W (Auto-Predicho, Cero Filtración)"),
    ]

    for cond_key, cond_name in conditions:
        n = len(dataset_records)
        n11, n10, n01, n00 = 0, 0, 0, 0
        fmt_ok = 0
        hidden_dig_corr, hidden_dig_tot = 0, 0
        t0 = time.perf_counter()

        for p_b, t_b, tw, tms in dataset_records:
            p_str = p_b.decode("utf-8")
            t_str = t_b.decode("utf-8")

            if cond_key == "staggered_w_then_ms":
                gen_txt = run_staggered_generation(model, p_b, cfg, r_infer, order="w_then_ms")
            elif cond_key == "staggered_ms_then_w":
                gen_txt = run_staggered_generation(model, p_b, cfg, r_infer, order="ms_then_w")
            else:
                x_seq = build_clamped_sequence(p_b, cfg, cond_key, tw, tms)
                x_out = run_diffusion_on_sequence(model, x_seq, cfg, r_infer)
                gen_txt = decode_tokens_to_text(x_out, cfg)

            is_fmt, pw, pms = parse_watt_ms_prediction(gen_txt, p_str)
            if is_fmt:
                fmt_ok += 1

            w_ok = (pw == tw)
            ms_ok = (pms == tms)

            if w_ok and ms_ok:
                n11 += 1
            elif w_ok and not ms_ok:
                n10 += 1
            elif not w_ok and ms_ok:
                n01 += 1
            else:
                n00 += 1

            continuation = gen_txt[len(p_str):] if gen_txt.startswith(p_str) else ""
            c_dig, t_dig = compute_strictly_hidden_digit_acc(continuation, t_str, cond_key)
            hidden_dig_corr += c_dig
            hidden_dig_tot += t_dig

        dt = (time.perf_counter() - t0) * 1000.0 / max(1, n)
        fmt_pct = 100.0 * fmt_ok / n
        dig_acc_hidden = 100.0 * hidden_dig_corr / max(1, hidden_dig_tot)

        print(f"\n--> {cond_name}:")
        print(f"    Formato: {fmt_pct:>5.1f}% | DigAcc(Solo Ocultos): {dig_acc_hidden:>5.1f}% | Latencia: {dt:.1f} ms/caso")
        print(f"    Matriz: [n11 Ambos OK]={n11:>2} ({100*n11/n:>4.1f}%) | [n10 Solo Watt]={n10:>2} ({100*n10/n:>4.1f}%) | [n01 Solo MS]={n01:>2} ({100*n01/n:>4.1f}%) | [n00 Ninguno]={n00:>2} ({100*n00/n:>4.1f}%)")


def main():
    cfg = HybridConfig()
    model = TernaryRecursiveDiffusionLM(cfg)
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.load_bitcos_archive(cfg.bitcos_export_path)
    model.prepare_for_eval()

    val_ds = ProceduralCompositionalDataset(cfg, split="val", num_samples=96)
    # Ejecutar para R=2 y R=8
    for r_test in (2, 8):
        evaluate_rigorous_suite(model, val_ds.unique_records, cfg, r_infer=r_test)


if __name__ == "__main__":
    main()
