import math
import os
from typing import List, Tuple
import torch
import torch.nn.functional as F

from config import HybridConfig
from model import TernaryRecursiveDiffusionLM
from data import ProceduralCompositionalDataset


def decode_tokens_clean(tokens: List[int], cfg: HybridConfig) -> str:
    """Decodificación limpia por bytes UTF-8 sin truncamientos heurísticos de texto."""
    byte_seq = []
    for tok in tokens:
        if tok in (cfg.eos_token_id, cfg.pad_token_id):
            break
        if tok < 256:
            byte_seq.append(tok)
    return bytes(byte_seq).decode("utf-8", errors="replace")


@torch.no_grad()
def generate_block_causal_with_kv_cache(
    model: TernaryRecursiveDiffusionLM,
    prompt_text: str,
    cfg: HybridConfig,
    gen_length: int = 24,
    num_recursions: int = None,
) -> Tuple[str, List[str], int]:
    """
    Generador Block-Causal TRDM-v3 Saneado:
    1. Procesa bloques del prompt y congela K, V en la caché.
    2. En el bloque activo, aplica difusión con Reconsideración de Tokens (Remasking).
    3. Transporta el estado latente (z_R, y_R) desconectado entre pasos.
    4. Detención natural en cuanto se consolida [EOS] o [PAD] (sin heurísticas de texto).
    """
    model.eval()
    R = num_recursions if num_recursions is not None else cfg.num_recursions
    prompt_bytes = list(prompt_text.encode("utf-8"))

    # Secuencia inicial: prompt + posiciones [MASK] alineadas a múltiplos de bloque
    prompt_len = 1 + len(prompt_bytes)
    needed_len = prompt_len + gen_length
    total_len = min(cfg.max_seq_len, math.ceil(needed_len / cfg.block_size) * cfg.block_size)
    num_masks = max(1, total_len - prompt_len)
    seq = [cfg.bos_token_id] + prompt_bytes + [cfg.mask_token_id] * num_masks
    seq = seq[:total_len]

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

        # Bloque del prompt sin máscaras: consolidamos su caché KV una única vez
        if not (block_tokens == cfg.mask_token_id).any():
            t_zero = torch.tensor([0.0], device=cfg.device)
            dt_zero = torch.tensor([0.0], device=cfg.device)
            _, _, _, kv_caches_committed, _ = model(
                block_tokens,
                t_zero,
                delta_t=dt_zero,
                num_recursions=R,
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
                num_recursions=R,
                kv_caches_per_r=kv_caches_committed,
                start_pos=committed_len,
            )
            total_block_evals += len(inter_logits)

            logits[:, :, cfg.mask_token_id] = -1e9
            logits[:, :, cfg.bos_token_id] = -1e9
            probs = F.softmax(logits[0], dim=-1)
            confidences, best_tokens = probs.max(dim=-1)

            # Reconsideración de Tokens (Remasking de baja confianza)
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

            # Historial decodificado limpiamente
            resp_slice = x_full[0, prompt_len:total_len].tolist()
            history.append(decode_tokens_clean(resp_slice, cfg))

        # Consolidar caché KV del bloque finalizado
        t_zero = torch.tensor([0.0], device=cfg.device)
        dt_zero = torch.tensor([0.0], device=cfg.device)
        _, _, _, kv_caches_committed, _ = model(
            block_tokens,
            t_zero,
            delta_t=dt_zero,
            num_recursions=R,
            kv_caches_per_r=kv_caches_committed,
            start_pos=committed_len,
        )
        committed_len = b_end

        # Corte natural LLaDA: si el bloque contiene [EOS], detener sin procesar más bloques
        if (block_tokens[0] == cfg.eos_token_id).any():
            break

    final_resp_tokens = x_full[0, prompt_len:total_len].tolist()
    final_text = decode_tokens_clean(final_resp_tokens, cfg)
    return final_text, history, total_block_evals


def test_hypothesis_h1(model: TernaryRecursiveDiffusionLM, cfg: HybridConfig):
    """
    Prueba de la Hipótesis H1:
    Modelo entrenado con R=2, evaluado con R in {1, 2, 4, 8} sobre combinaciones
    rigurosamente reservadas (Held-Out) jamás vistas en entrenamiento.
    """
    print("\n" + "=" * 75)
    print("  TEST HIPÓTESIS H1: ESCALADO DE CÓMPUTO EN INFERENCIA (R in {1, 2, 4, 8}) ")
    print("=" * 75)

    held_out_cases = [
        # Combinaciones reservadas: nodo, perfil, redundancia, carga -> esperado
        ("nodo: server\nperfil: turbo\nred: dual\ncarga: pico\n", "watt: 87\nms: 16\n"),
        ("nodo: compute\nperfil: eco\nred: simple\ncarga: pico\n", "watt: 22\nms: 40\n"),
        ("nodo: router\nperfil: edge\nred: mesh\ncarga: pico\n", "watt: 53\nms: 7\n"),
        ("nodo: storage\nperfil: batch\nred: triple\ncarga: media\n", "watt: 35\nms: 22\n"),
    ]

    for c_idx, (prompt, target_esperado) in enumerate(held_out_cases, 1):
        print(f"\n--- Caso OOD #{c_idx} (Held-Out) ---")
        print(f"Prompt   : {repr(prompt)}")
        print(f"Esperado : {repr(target_esperado.strip())}")

        for r_test in (1, 2, 4, 8):
            salida, _, evals = generate_block_causal_with_kv_cache(
                model, prompt, cfg, num_recursions=r_test
            )
            acierto = (salida.strip() == target_esperado.strip())
            icono = "✓ EXACTO" if acierto else "✗ DIF"
            print(f"  R={r_test} | {icono} | Salida: {repr(salida.strip())} | Evals: {evals}")


if __name__ == "__main__":
    cfg = HybridConfig()
    model = TernaryRecursiveDiffusionLM(cfg)
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    if os.path.exists(cfg.bitcos_export_path):
        loaded = model.load_bitcos_archive(cfg.bitcos_export_path)
        print(f"[BITCOSv2] {loaded} capas ternarias cargadas en modo de inferencia entera int8/int32.")
    else:
        print("[Modo FP32/QAT] Evaluando sin archivo binario BITCOS.")

    test_hypothesis_h1(model, cfg)
