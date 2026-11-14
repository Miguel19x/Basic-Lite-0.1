import math
import os
from typing import List, Tuple
import torch
import torch.nn.functional as F

from config import HybridConfig
from model import TernaryRecursiveDiffusionLM


@torch.no_grad()
def generate_block_causal_with_kv_cache(
    model: TernaryRecursiveDiffusionLM,
    prompt_text: str,
    cfg: HybridConfig,
    gen_length: int = 16,
) -> Tuple[str, List[str], int]:
    """
    Generador Block-Causal TRDM-v3:
    1. Congela en caché KV los bloques que pertenecen al prompt.
    2. En el bloque activo, aplica difusión con Reconsideración de Tokens (Remasking de baja confianza).
    3. Transporta el estado latente (z_R, y_R) desconectado entre pasos sucesivos.
    4. Condiciona por salto delta_t y detiene limpiamente al consolidar el punto o [EOS].
    """
    model.eval()
    prompt_bytes = list(prompt_text.encode("utf-8"))
    seq = [cfg.bos_token_id] + prompt_bytes + [cfg.mask_token_id] * gen_length
    seq = seq[: cfg.max_seq_len]
    prompt_len = 1 + len(prompt_bytes)
    total_len = len(seq)

    x_full = torch.tensor([seq], dtype=torch.long, device=cfg.device)
    B_blk = cfg.block_size
    num_blocks = math.ceil(total_len / B_blk)

    kv_caches_committed = None
    committed_len = 0
    history: List[str] = []
    total_block_evals = 0

    for b_idx in range(num_blocks):
        b_start = b_idx * B_blk
        b_end = min((b_idx + 1) * B_blk, total_len)
        block_tokens = x_full[:, b_start:b_end]

        # Bloques del prompt (sin máscaras): solo calculamos su caché KV
        if not (block_tokens == cfg.mask_token_id).any():
            t_zero = torch.tensor([0.0], device=cfg.device)
            dt_zero = torch.tensor([0.0], device=cfg.device)
            _, _, _, kv_caches_committed, _ = model(
                block_tokens,
                t_zero,
                delta_t=dt_zero,
                kv_caches_per_r=kv_caches_committed,
                start_pos=committed_len,
            )
            committed_len = b_end
            total_block_evals += 1
            continue

        # Bloque activo con máscaras
        editable_positions = (block_tokens[0] == cfg.mask_token_id).clone()
        initial_masks_in_block = int(editable_positions.sum().item())
        carried_states = None
        prev_ratio = 1.0
        steps = cfg.diffusion_steps_per_block

        for s_idx in range(steps):
            mask_pos = (block_tokens[0] == cfg.mask_token_id)
            num_masked = int(mask_pos.sum().item())
            if num_masked == 0:
                break

            mask_ratio = num_masked / max(1, initial_masks_in_block)
            t_tensor = torch.tensor([mask_ratio], device=cfg.device)
            dt_tensor = torch.tensor([max(0.0, prev_ratio - mask_ratio)], device=cfg.device)
            prev_ratio = mask_ratio

            logits, inter_logits, _, _, carried_states = model(
                block_tokens,
                t_tensor,
                delta_t=dt_tensor,
                prev_states=carried_states,
                kv_caches_per_r=kv_caches_committed,
                start_pos=committed_len,
            )
            total_block_evals += len(inter_logits)

            logits[:, :, cfg.mask_token_id] = -1e9
            logits[:, :, cfg.bos_token_id] = -1e9
            probs = F.softmax(logits[0], dim=-1)
            confidences, best_tokens = probs.max(dim=-1)

            # Política de Reconsideración (Low-Confidence Remasking)
            edit_indices = torch.nonzero(editable_positions, as_tuple=False).squeeze(-1)
            block_tokens[0, edit_indices] = best_tokens[edit_indices]

            progress = (s_idx + 1) / steps
            target_rem = int(math.floor(initial_masks_in_block * math.cos(progress * math.pi / 2.0)))
            if s_idx < steps - 1 and target_rem > 0:
                edit_conf = confidences[edit_indices]
                _, lowest_order = torch.topk(edit_conf, k=min(target_rem, len(edit_indices)), largest=False)
                remask_pos = edit_indices[lowest_order]
                block_tokens[0, remask_pos] = cfg.mask_token_id

            x_full[:, b_start:b_end] = block_tokens

            # Decodificar respuesta limpia hasta el primer delimitador de fin (. o \n o EOS)
            resp_tokens = x_full[0, prompt_len:total_len].tolist()
            resp_chars = []
            for tok in resp_tokens:
                if tok == cfg.mask_token_id:
                    resp_chars.append("█")
                elif tok in (cfg.eos_token_id, cfg.pad_token_id):
                    break
                elif tok < 256:
                    ch = chr(tok)
                    resp_chars.append(ch)
                    if ch in ('.', '\n'):
                        break
            history.append("".join(resp_chars).strip())

        # Consolidar estados K, V finales del bloque en la caché
        t_zero = torch.tensor([0.0], device=cfg.device)
        dt_zero = torch.tensor([0.0], device=cfg.device)
        _, _, _, kv_caches_committed, _ = model(
            block_tokens,
            t_zero,
            delta_t=dt_zero,
            kv_caches_per_r=kv_caches_committed,
            start_pos=committed_len,
        )
        committed_len = b_end

        # Si el bloque activo consolidó el fin de la respuesta, detener la generación
        current_resp_text = history[-1] if history else ""
        if "." in current_resp_text or (block_tokens[0] == cfg.eos_token_id).any():
            break

    final_resp_tokens = x_full[0, prompt_len:total_len].tolist()
    final_chars = []
    for tok in final_resp_tokens:
        if tok in (cfg.mask_token_id, cfg.eos_token_id, cfg.pad_token_id):
            break
        elif tok < 256:
            ch = chr(tok)
            final_chars.append(ch)
            if ch in ('.', '\n'):
                break

    final_text = "".join(final_chars).strip()
    return final_text, history, total_block_evals


if __name__ == "__main__":
    cfg = HybridConfig()
    model = TernaryRecursiveDiffusionLM(cfg)
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    if os.path.exists(cfg.bitcos_export_path):
        loaded = model.load_bitcos_archive(cfg.bitcos_export_path)
        print(f"[BITCOSv2] {loaded} capas ternarias cargadas en modo de inferencia entera int8/int32.")
    else:
        print("[Modo FP32/QAT] Evaluando sin archivo BITCOS binario.")

    prompts_prueba = [
        ("Pregunta: Cual es la capital de Francia?\nRespuesta: ", "Paris"),
        ("Pregunta: De que color es el cielo despejado?\nRespuesta: ", "Azul"),
        ("Pregunta: Que animal ladra y cuida la casa?\nRespuesta: ", "El perro"),
        ("Pregunta: Cual es el opuesto de frio?\nRespuesta: ", "Caliente"),
        ("Pregunta: Que profesional apaga los incendios?\nRespuesta: ", "El bombero"),
    ]

    print("=========================================================================")
    print("  TRDM-v3: DIFUSIÓN BLOCK-CAUSAL + REMASKING + GQA + C_cap(t) + DELTA_t  ")
    print("=========================================================================\n")

    for idx, (p_text, esperado) in enumerate(prompts_prueba, 1):
        print(f"--- Caso #{idx} (Esperado: '{esperado}') ---")
        print(f"Prompt : {repr(p_text.strip())}")
        texto_final, rondas, evals = generate_block_causal_with_kv_cache(model, p_text, cfg)
        print("Evolución por paso activo:")
        for r_i, r_str in enumerate(rondas, 1):
            print(f"  [Paso {r_i}] -> '{r_str}'")
        print(f"Respuesta final limpia: '{texto_final}' | Evaluaciones de bloque: {evals}\n")
