import math
import random
import re
import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from config import HybridConfig
from model import TernaryRecursiveDiffusionLM
from data import ProceduralCompositionalDataset
from trdm.diffusion.sampler import generate_block_causal_with_kv_cache
from trdm.data.parsers import parse_scratchpad_prediction
from train import compute_v4_hybrid_loss
from trdm.diffusion.corruption import apply_blockwise_diffusion_corruption


# =====================================================================
# 1. MÁSCARA ESTRICTA DE VOCABULARIO DE TAREA (ASCII PERMITIDO)
# =====================================================================
def get_task_allowed_mask(cfg: HybridConfig) -> torch.Tensor:
    """
    Construye la máscara booleana estricta para el vocabulario de la tarea:
    - Dígitos ASCII ('0'..'9')
    - Caracteres de plantilla requeridos ('p', 'w', 'm', 's', 'a', 't', ':', ' ', '\n')
    - Tokens especiales permitidos bajo política estricta (PAD, EOS)
    - Prohíbe de forma absoluta MASK (257), BOS (258) y bytes de control/no-ASCII.
    """
    allowed = torch.zeros(cfg.vocab_size, dtype=torch.bool)
    # 1. Dígitos
    for b in range(ord('0'), ord('9') + 1):
        allowed[b] = True
    # 2. Caracteres de plantilla
    for ch in "pwmsat: \n":
        allowed[ord(ch)] = True
    # 3. Tokens especiales gobernados
    allowed[cfg.eos_token_id] = True
    allowed[cfg.pad_token_id] = True
    return allowed


# =====================================================================
# 2. MUESTREADOR CORREGIDO (Vocabulario de Tarea + Terminación Robusta)
# =====================================================================
@torch.no_grad()
def generate_block_causal_corrected(
    model,
    prompt_text: str,
    cfg: HybridConfig,
    gen_length: Optional[int] = None,
    num_recursions_override: Optional[int] = None,
    restrict_vocab: bool = True,
    min_tokens_before_eos: int = 10,
) -> Tuple[str, List[str], int]:
    """
    Muestreador corregido:
    - Aplica máscara de vocabulario de tarea cuando restrict_vocab=True.
    - Elimina el break prematuro de bloques por EOS/PAD provisional.
    - Solo permite EOS tras min_tokens_before_eos posiciones generadas.
    - Restringe PAD estrictamente a posiciones posteriores a un EOS aceptado.
    """
    model.eval()
    R_infer = num_recursions_override if num_recursions_override is not None else cfg.num_recursions
    prompt_bytes = list(prompt_text.encode("utf-8"))
    prompt_len = len(prompt_bytes)

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

    allowed_mask = get_task_allowed_mask(cfg).to(cfg.device) if restrict_vocab else None

    kv_caches_committed = None
    committed_len = 0
    history: List[str] = []
    total_block_evals = 0

    # Registro de EOS secuencial aceptado
    eos_accepted = False
    eos_accepted_global_pos = None

    for b_idx in range(num_blocks):
        b_start = b_idx * B_blk
        b_end = (b_idx + 1) * B_blk
        block_tokens = x_full[:, b_start:b_end].clone()

        # Si el bloque ya es 100% prompt conocido, solo se añade al KV cache
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

            # Restricciones fundamentales universales
            logits[:, :, cfg.mask_token_id] = -1e9
            logits[:, :, cfg.bos_token_id] = -1e9

            # Restricción de vocabulario de tarea
            if allowed_mask is not None:
                logits[:, :, ~allowed_mask] = -1e9

            # Restricción posicional de EOS (no permitir EOS antes de longitud mínima)
            for pos_in_blk in range(B_blk):
                global_pos = b_start + pos_in_blk
                tokens_gen_so_far = global_pos - (1 + prompt_len)
                if tokens_gen_so_far < min_tokens_before_eos:
                    logits[0, pos_in_blk, cfg.eos_token_id] = -1e9

            # Restricción estricta de PAD: solo si EOS ya ocurrió previamente
            can_be_pad = torch.zeros(B_blk, dtype=torch.bool, device=cfg.device)
            if eos_accepted:
                can_be_pad[:] = True
            else:
                # Comprobar si hay EOS confirmado en este bloque
                is_eos = (block_tokens[0] == cfg.eos_token_id)
                if is_eos.any():
                    first_e = torch.nonzero(is_eos, as_tuple=False)[0].item()
                    if first_e + 1 < B_blk:
                        can_be_pad[first_e + 1:] = True
            logits[0, ~can_be_pad, cfg.pad_token_id] = -1e9

            probs = F.softmax(logits[0], dim=-1)
            confidences, best_tokens = probs.max(dim=-1)

            editable_indices = torch.nonzero(editable_positions, as_tuple=False).squeeze(-1)
            progress = (s_idx + 1) / steps
            target_masks_rem = int(math.floor(initial_masks * math.cos(progress * math.pi / 2.0)))

            if s_idx == steps - 1 or target_masks_rem == 0:
                block_tokens[0, editable_indices] = best_tokens[editable_indices]
            else:
                masked_indices = torch.nonzero(masked_pos, as_tuple=False).squeeze(-1)
                num_to_reveal = max(1, len(masked_indices) - target_masks_rem)
                masked_conf = confidences[masked_indices]
                _, top_reveal_order = torch.topk(masked_conf, k=min(num_to_reveal, len(masked_indices)), largest=True)
                chosen_positions = masked_indices[top_reveal_order]
                block_tokens[0, chosen_positions] = best_tokens[chosen_positions]

            x_full[:, b_start:b_end] = block_tokens

            # Actualizar estado de EOS confirmado
            if not eos_accepted:
                is_eos_now = (block_tokens[0] == cfg.eos_token_id)
                if is_eos_now.any():
                    e_idx = torch.nonzero(is_eos_now, as_tuple=False)[0].item()
                    global_e = b_start + e_idx
                    if global_e - (1 + prompt_len) >= min_tokens_before_eos:
                        eos_accepted = True
                        eos_accepted_global_pos = global_e

            raw_bytes = bytearray()
            for tok in x_full[0, 1:total_len].tolist():
                if tok in (cfg.eos_token_id, cfg.pad_token_id):
                    break
                if tok < 256:
                    raw_bytes.append(tok)
            history.append(raw_bytes.decode("utf-8", errors="replace"))

        # Fin de difusión del bloque: registrar en KV cache sin abortar
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

        # Si ya se aceptó EOS y todo el resto de este bloque es PAD, se puede detener
        if eos_accepted and (block_tokens[0] == cfg.pad_token_id).all():
            break

    final_text = history[-1] if history else prompt_text
    return final_text, history, total_block_evals


# =====================================================================
# 3. PARSERS DUALES: ESTRICTO Y EXTRACTOR DIAGNÓSTICO TRI-ESTADO
# =====================================================================
_STRICT_SUB_RE = re.compile(r"^pw:\s*(\d+)\s*\npms:\s*(\d+)\s*$", re.MULTILINE)
_STRICT_FINAL_RE = re.compile(r"^watt:\s*(\d+)\s*\nms:\s*(\d+)\s*$", re.MULTILINE)

# Patrones de campo diagnóstico ciegos al ground truth
_FIELD_PW_RE = re.compile(r"p[a-z_]*[:_\s]*(\d+)", re.IGNORECASE)
_FIELD_PMS_RE = re.compile(r"pm[a-z_]*[:_\s]*(\d+)", re.IGNORECASE)
_FIELD_WATT_RE = re.compile(r"w[a-z_]*[:_\s]*(\d+)", re.IGNORECASE)
_FIELD_MS_RE = re.compile(r"m[s_]*[:_\s]*(\d+)", re.IGNORECASE)


def parse_subtotals_dual(
    text: str,
) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Retorna (status, pred_pw, pred_pms)
    status in {"STRICT_VALID", "UNAMBIGUOUS_RECOVERABLE", "AMBIGUOUS_OR_CORRUPT"}
    """
    # 1. Chequeo Estricto
    m_strict = _STRICT_SUB_RE.search(text)
    if m_strict:
        try:
            return "STRICT_VALID", int(m_strict.group(1)), int(m_strict.group(2))
        except ValueError:
            pass

    # 2. Chequeo Diagnóstico No Sesgado
    # Buscar campo 1 (pw) y campo 2 (pms)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) >= 2:
        m1 = _FIELD_PW_RE.search(lines[0])
        m2 = _FIELD_PMS_RE.search(lines[1])
        if m1 and m2:
            try:
                v1 = int(m1.group(1))
                v2 = int(m2.group(2) if m2.lastindex >= 2 else m2.group(1))
                # Verificar que no haya números extra en conflicto en la misma línea
                all_nums1 = re.findall(r"\d+", lines[0])
                all_nums2 = re.findall(r"\d+", lines[1])
                if len(all_nums1) == 1 and len(all_nums2) == 1:
                    return "UNAMBIGUOUS_RECOVERABLE", v1, v2
            except ValueError:
                pass

    return "AMBIGUOUS_OR_CORRUPT", None, None


def parse_finals_dual(
    text: str,
) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Retorna (status, pred_watt, pred_ms)
    status in {"STRICT_VALID", "UNAMBIGUOUS_RECOVERABLE", "AMBIGUOUS_OR_CORRUPT"}
    """
    # 1. Chequeo Estricto
    m_strict = _STRICT_FINAL_RE.search(text)
    if m_strict:
        try:
            return "STRICT_VALID", int(m_strict.group(1)), int(m_strict.group(2))
        except ValueError:
            pass

    # 2. Chequeo Diagnóstico No Sesgado
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) >= 2:
        m1 = _FIELD_WATT_RE.search(lines[0])
        m2 = _FIELD_MS_RE.search(lines[1])
        if m1 and m2:
            try:
                v1 = int(m1.group(1))
                v2 = int(m2.group(1))
                all_nums1 = re.findall(r"\d+", lines[0])
                all_nums2 = re.findall(r"\d+", lines[1])
                if len(all_nums1) == 1 and len(all_nums2) == 1:
                    return "UNAMBIGUOUS_RECOVERABLE", v1, v2
            except ValueError:
                pass

    return "AMBIGUOUS_OR_CORRUPT", None, None


# =====================================================================
# 4. PROTOCOLO DE EVALUACIÓN COMPARATIVA (HITO 1)
# =====================================================================
def run_evaluation_suite():
    cfg = HybridConfig()
    cfg.seed = 42
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)

    print("=" * 76)
    print("  HITO 1: EVALUACIÓN COMPARATIVA DE MUESTREADORES Y AUDITORÍA DE PARSERS")
    print("  (Sin Reentrenamiento: Evaluando Checkpoints Existentes)")
    print("=" * 76)

    # 1. Cargar Checkpoint Principal de 800 Pasos
    print("\n[1/3] Cargando Checkpoint Principal (trdm_v4_checkpoint.pt, 800 Pasos)...")
    model_800 = TernaryRecursiveDiffusionLM(cfg).to(cfg.device)
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    model_800.load_state_dict(ckpt["model_state"])
    model_800.load_bitcos_archive(cfg.bitcos_export_path)
    model_800.prepare_for_eval()

    train_ds = ProceduralCompositionalDataset(cfg, split="train", num_samples=100)
    val_ds = ProceduralCompositionalDataset(cfg, split="val", num_samples=100)

    rng = random.Random(cfg.seed)
    train_records = rng.sample(train_ds.unique_records, 96)
    heldout_records = list(val_ds.unique_records)

    splits = [
        ("Train (96 casos)", train_records),
        ("Held-Out OOD (96 casos)", heldout_records),
    ]

    for split_label, records in splits:
        print(f"\n" + "=" * 76)
        print(f"  EVALUACIÓN EN: {split_label}")
        print("=" * 76)

        for sampler_name, use_corrected, restrict in [
            ("Muestreador Original (Libre / Sin Restricción)", False, False),
            ("Muestreador Corregido (Vocabulario de Tarea + Terminación Robusta)", True, True),
        ]:
            print(f"\n>>> Condición: {sampler_name} <<<")
            n = len(records)

            # Contadores C1 (Subtotales)
            c1_strict_fmt, c1_strict_em = 0, 0
            c1_recov, c1_recov_em = 0, 0
            c1_corrupt = 0

            # Contadores C2 (Oráculo -> Respuesta Final)
            c2_strict_fmt, c2_strict_em = 0, 0
            c2_recov, c2_recov_em = 0, 0
            c2_corrupt = 0

            for prompt_b, target_b, tw, tms in records:
                p_str = prompt_b.decode("utf-8")
                t_str = target_b.decode("utf-8")
                _, tr_pw, tr_pms, _, _ = parse_scratchpad_prediction(t_str, "")

                # C1: Factores -> Generar Subtotales
                if use_corrected:
                    gen1, _, _ = generate_block_causal_corrected(
                        model_800, p_str, cfg, gen_length=24, num_recursions_override=2, restrict_vocab=restrict
                    )
                else:
                    gen1, _, _ = generate_block_causal_with_kv_cache(
                        model_800, p_str, cfg, gen_length=24, num_recursions_override=2
                    )
                cont1 = gen1[len(p_str):] if gen1.startswith(p_str) else gen1
                st1, p1_pw, p1_pms = parse_subtotals_dual(cont1)

                if st1 == "STRICT_VALID":
                    c1_strict_fmt += 1
                    if p1_pw == tr_pw and p1_pms == tr_pms:
                        c1_strict_em += 1
                elif st1 == "UNAMBIGUOUS_RECOVERABLE":
                    c1_recov += 1
                    if p1_pw == tr_pw and p1_pms == tr_pms:
                        c1_recov_em += 1
                else:
                    c1_corrupt += 1

                # C2: Factores + Subtotales Oráculo -> Generar Final
                prompt_c2 = p_str + f"pw: {tr_pw}\npms: {tr_pms}\n"
                if use_corrected:
                    gen2, _, _ = generate_block_causal_corrected(
                        model_800, prompt_c2, cfg, gen_length=24, num_recursions_override=2, restrict_vocab=restrict
                    )
                else:
                    gen2, _, _ = generate_block_causal_with_kv_cache(
                        model_800, prompt_c2, cfg, gen_length=24, num_recursions_override=2
                    )
                cont2 = gen2[len(prompt_c2):] if gen2.startswith(prompt_c2) else gen2
                st2, p2_w, p2_ms = parse_finals_dual(cont2)

                if st2 == "STRICT_VALID":
                    c2_strict_fmt += 1
                    if p2_w == tw and p2_ms == tms:
                        c2_strict_em += 1
                elif st2 == "UNAMBIGUOUS_RECOVERABLE":
                    c2_recov += 1
                    if p2_w == tw and p2_ms == tms:
                        c2_recov_em += 1
                else:
                    c2_corrupt += 1

            # Imprimir tabla de resultados para esta combinación
            print(f"  [Condición 1: Solo Subtotales (pw, pms)]")
            print(f"    - Formato Estricto Válido: {c1_strict_fmt}/{n} ({100.0*c1_strict_fmt/n:.1f}%) | EM Estricto: {c1_strict_em}/{n} ({100.0*c1_strict_em/n:.1f}%)")
            print(f"    - Recuperable No Ambiguo:  {c1_recov}/{n} ({100.0*c1_recov/n:.1f}%) | EM Diagnóstico: {c1_recov_em}/{n} ({100.0*c1_recov_em/n:.1f}%)")
            print(f"    - Ambiguo / No Extraíble:  {c1_corrupt}/{n} ({100.0*c1_corrupt/n:.1f}%)")
            c1_tot_em = c1_strict_em + c1_recov_em
            print(f"    - EM Numérico Global (Estricto + Diagnóstico): {c1_tot_em}/{n} ({100.0*c1_tot_em/n:.1f}%)")

            print(f"  [Condición 2: Respuesta con Subtotales ORÁCULO (watt, ms)]")
            print(f"    - Formato Estricto Válido: {c2_strict_fmt}/{n} ({100.0*c2_strict_fmt/n:.1f}%) | EM Estricto: {c2_strict_em}/{n} ({100.0*c2_strict_em/n:.1f}%)")
            print(f"    - Recuperable No Ambiguo:  {c2_recov}/{n} ({100.0*c2_recov/n:.1f}%) | EM Diagnóstico: {c2_recov_em}/{n} ({100.0*c2_recov_em/n:.1f}%)")
            print(f"    - Ambiguo / No Extraíble:  {c2_corrupt}/{n} ({100.0*c2_corrupt/n:.1f}%)")
            c2_tot_em = c2_strict_em + c2_recov_em
            print(f"    - EM Numérico Global (Estricto + Diagnóstico): {c2_tot_em}/{n} ({100.0*c2_tot_em/n:.1f}%)")

    # 2. Control Cruzado sobre el Modelo de 8 Casos
    print("\n" + "=" * 76)
    print("  CONTROL CRUZADO: MODELO SOBREAJUSTADO (8 CASOS)")
    print("=" * 76)
    print("Entrenando modelo de 8 casos en memoria (400 pasos)...")

    model_8 = TernaryRecursiveDiffusionLM(cfg).to(cfg.device)
    opt_8 = torch.optim.AdamW(model_8.configure_adamw_groups(), lr=2.0e-3, betas=(0.9, 0.95))
    sched_8 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_8, T_max=400, eta_min=1e-4)

    fixed_8 = train_records[:8]
    batch_x0, batch_can_mask, batch_digit_mask = [], [], []
    for prompt_b, target_b, _, _ in fixed_8:
        seq = [cfg.bos_token_id] + list(prompt_b) + list(target_b) + [cfg.eos_token_id]
        p_len = 1 + len(prompt_b)
        seq = seq[: cfg.max_seq_len]
        seq = seq + [cfg.pad_token_id] * (cfg.max_seq_len - len(seq))
        c_mask = torch.zeros(cfg.max_seq_len, dtype=torch.bool)
        c_mask[p_len:] = True
        d_mask = torch.zeros(cfg.max_seq_len, dtype=torch.bool)
        t_end = min(cfg.max_seq_len, p_len + len(target_b))
        for p in range(p_len, t_end):
            if 48 <= seq[p] <= 57:
                d_mask[p] = True
        batch_x0.append(torch.tensor(seq, dtype=torch.long))
        batch_can_mask.append(c_mask)
        batch_digit_mask.append(d_mask)

    x0_8 = torch.stack(batch_x0).to(cfg.device)
    cmask_8 = torch.stack(batch_can_mask).to(cfg.device)
    dmask_8 = torch.stack(batch_digit_mask).to(cfg.device)

    model_8.train()
    for s in range(1, 401):
        x_t, m_pos, t_seq = apply_blockwise_diffusion_corruption(x0_8, cmask_8, cfg)
        out1 = model_8(x_t, t_seq, num_recursions=2, supervised_mask=m_pos)
        l1, _ = compute_v4_hybrid_loss(
            out1.final_logits, out1.intermediate_logits, out1.halt_probs,
            x0_8, m_pos, dmask_8, t_seq, cfg
        )
        with torch.no_grad():
            if getattr(cfg, "lazy_vocab_projection", True):
                p1_sup = out1.final_logits.argmax(dim=-1)
                x_pred = x_t.clone()
                x_pred[m_pos] = p1_sup
            else:
                x_pred = out1.final_logits.argmax(dim=-1)
            km = m_pos & (x_pred != x0_8)
        if km.any():
            x_ref = torch.where(km, cfg.mask_token_id, x_pred)
            out2 = model_8(x_ref, t_seq * 0.5, num_recursions=2, supervised_mask=km)
            l2, _ = compute_v4_hybrid_loss(
                out2.final_logits, out2.intermediate_logits, out2.halt_probs,
                x0_8, km, dmask_8, t_seq * 0.5, cfg
            )
            tot_l = 0.45 * l1 + 0.55 * l2
        else:
            tot_l = l1
        opt_8.zero_grad()
        tot_l.backward()
        torch.nn.utils.clip_grad_norm_(model_8.parameters(), 1.0)
        opt_8.step()
        sched_8.step()

    model_8.prepare_for_eval()

    for sampler_name, use_corrected, restrict in [
        ("Control 8 Casos: Muestreador Original (Libre)", False, False),
        ("Control 8 Casos: Muestreador Corregido (Vocabulario de Tarea + Terminación Robusta)", True, True),
    ]:
        print(f"\n>>> {sampler_name} <<<")
        c2_strict_fmt, c2_strict_em = 0, 0
        c2_recov, c2_recov_em = 0, 0
        c2_corrupt = 0

        for idx, (prompt_b, target_b, tw, tms) in enumerate(fixed_8):
            p_str = prompt_b.decode("utf-8")
            t_str = target_b.decode("utf-8")
            _, tr_pw, tr_pms, _, _ = parse_scratchpad_prediction(t_str, "")

            prompt_c2 = p_str + f"pw: {tr_pw}\npms: {tr_pms}\n"
            if use_corrected:
                gen2, _, _ = generate_block_causal_corrected(
                    model_8, prompt_c2, cfg, gen_length=24, num_recursions_override=2, restrict_vocab=restrict
                )
            else:
                gen2, _, _ = generate_block_causal_with_kv_cache(
                    model_8, prompt_c2, cfg, gen_length=24, num_recursions_override=2
                )
            cont2 = gen2[len(prompt_c2):] if gen2.startswith(prompt_c2) else gen2
            st2, p2_w, p2_ms = parse_finals_dual(cont2)

            if st2 == "STRICT_VALID":
                c2_strict_fmt += 1
                if p2_w == tw and p2_ms == tms:
                    c2_strict_em += 1
            elif st2 == "UNAMBIGUOUS_RECOVERABLE":
                c2_recov += 1
                if p2_w == tw and p2_ms == tms:
                    c2_recov_em += 1
            else:
                c2_corrupt += 1

            if idx < 3:
                print(f"  [Ejemplo #{idx+1}] Target: watt={tw}, ms={tms} | Status={st2} (w={p2_w}, ms={p2_ms}) | Cont: {repr(cont2)}")

        print(f"  Resultados Condición Oráculo (8 Casos):")
        print(f"    - Formato Estricto: {c2_strict_fmt}/8 ({100.0*c2_strict_fmt/8:.1f}%) | EM Estricto: {c2_strict_em}/8 ({100.0*c2_strict_em/8:.1f}%)")
        print(f"    - Recuperable No Ambiguo: {c2_recov}/8 ({100.0*c2_recov/8:.1f}%) | EM Diagnóstico: {c2_recov_em}/8 ({100.0*c2_recov_em/8:.1f}%)")
        print(f"    - Ambiguo / Corrupto: {c2_corrupt}/8 ({100.0*c2_corrupt/8:.1f}%)")
        tot = c2_strict_em + c2_recov_em
        print(f"    - Total Valores Correctos Emitidos: {tot}/8 ({100.0*tot/8:.1f}%)")


if __name__ == "__main__":
    run_evaluation_suite()
