import random
import re
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from config import HybridConfig


class ProceduralCompositionalDataset(Dataset):
    """
    Universo exacto de 480 combinaciones (4 nodos x 6 cargas x 4 redundancias x 5 perfiles):
    - 384 combinaciones de entrenamiento (80.0%)
    - 96 combinaciones reservadas Held-Out (20.0%)
    """
    NODOS = [("sensor", 10, 4), ("gateway", 28, 9), ("switch", 55, 3), ("router", 90, 14)]
    CARGAS = [
        ("idle", 0.4, 1.0), ("baja", 0.7, 1.2), ("media", 1.0, 1.5),
        ("alta", 1.4, 2.0), ("pico", 1.8, 2.7), ("saturada", 2.2, 3.5),
    ]
    REDUNDANCIAS = [("ninguna", 0, 0), ("simple", 8, 2), ("doble", 18, 5), ("mesh", 30, 9)]
    PERFILES = [("eco", -5, 3), ("base", 0, 0), ("turbo", 15, -2), ("edge", 6, -1), ("batch", 10, 4)]

    # 24 tripletas (nodo, carga, perfil) válidas x 4 redundancias = 96 tuplas Held-Out (20.0%)
    HELD_OUT_TRIPLETS = {
        ("sensor", "idle", "turbo"), ("sensor", "baja", "edge"), ("sensor", "media", "batch"),
        ("sensor", "alta", "eco"), ("sensor", "pico", "base"), ("sensor", "saturada", "turbo"),
        ("gateway", "idle", "edge"), ("gateway", "baja", "batch"), ("gateway", "media", "eco"),
        ("gateway", "alta", "base"), ("gateway", "pico", "turbo"), ("gateway", "saturada", "edge"),
        ("switch", "idle", "batch"), ("switch", "baja", "eco"), ("switch", "media", "base"),
        ("switch", "alta", "turbo"), ("switch", "pico", "edge"), ("switch", "saturada", "batch"),
        ("router", "idle", "eco"), ("router", "baja", "base"), ("router", "media", "turbo"),
        ("router", "alta", "edge"), ("router", "pico", "batch"), ("router", "saturada", "eco"),
    }

    def __init__(self, cfg: HybridConfig, split: str = "train", num_samples: int = 2400):
        super().__init__()
        self.cfg = cfg
        self.split = split

        train_pool, val_pool = [], []
        for n_name, b_w, b_ms in self.NODOS:
            for c_name, m_w, m_ms in self.CARGAS:
                for r_name, a_w, a_ms in self.REDUNDANCIAS:
                    for p_name, o_w, o_ms in self.PERFILES:
                        watt = max(5, int(round(b_w * m_w + a_w + o_w)))
                        ms = max(1, int(round(b_ms * m_ms + a_ms + o_ms)))
                        prompt_str = f"nodo: {n_name}\ncarga: {c_name}\nred: {r_name}\nperfil: {p_name}\n"
                        target_str = f"watt: {watt}\nms: {ms}\n"
                        record = (prompt_str.encode("utf-8"), target_str.encode("utf-8"), watt, ms)
                        if (n_name, c_name, p_name) in self.HELD_OUT_TRIPLETS:
                            val_pool.append(record)
                        else:
                            train_pool.append(record)

        assert len(train_pool) == 384, f"Error de partición: train_pool={len(train_pool)} != 384"
        assert len(val_pool) == 96, f"Error de partición: val_pool={len(val_pool)} != 96"

        base_pool = train_pool if split == "train" else val_pool
        self.unique_records: List[Tuple[bytes, bytes, int, int]] = list(base_pool)

        rng = random.Random(cfg.seed if split == "train" else cfg.seed + 999)
        self.samples: List[Tuple[bytes, bytes, int, int]] = [
            rng.choice(base_pool) for _ in range(num_samples)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_b, target_b, _, _ = self.samples[idx]
        seq = [self.cfg.bos_token_id] + list(prompt_b) + list(target_b) + [self.cfg.eos_token_id]
        prompt_len = 1 + len(prompt_b)

        seq = seq[: self.cfg.max_seq_len]
        pad_len = self.cfg.max_seq_len - len(seq)
        seq = seq + [self.cfg.pad_token_id] * pad_len

        x_0 = torch.tensor(seq, dtype=torch.long)

        # can_mask cubre toda la región de respuesta (incluyendo EOS y PAD posteriores)
        can_mask = torch.zeros(self.cfg.max_seq_len, dtype=torch.bool)
        can_mask[prompt_len:] = True

        # digit_mask identifica exactamente los bytes de dígitos ASCII ('0'..'9') en la respuesta
        digit_mask = torch.zeros(self.cfg.max_seq_len, dtype=torch.bool)
        target_end = min(self.cfg.max_seq_len, prompt_len + len(target_b))
        for pos in range(prompt_len, target_end):
            if 48 <= seq[pos] <= 57:
                digit_mask[pos] = True

        return x_0, can_mask, digit_mask


def apply_blockwise_diffusion_corruption(
    x_0: torch.Tensor,
    can_mask: torch.Tensor,
    cfg: HybridConfig,
    fixed_t: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aplica corrupción por bloques con soporte para bloques limpios (t=0.0)
    para alinear el entrenamiento con el llenado de caché KV en inferencia.
    """
    B, L = x_0.shape
    device = x_0.device
    B_blk = cfg.block_size
    num_blocks = (L + B_blk - 1) // B_blk

    if fixed_t is not None:
        t_block = torch.full((B, num_blocks), fixed_t, device=device)
    else:
        t_block = torch.empty((B, num_blocks), device=device).uniform_(
            cfg.min_mask_rate, cfg.max_mask_rate
        )
        zero_Selector = torch.rand((B, num_blocks), device=device) < 0.15
        t_block = torch.where(zero_Selector, torch.zeros_like(t_block), t_block)

    t_per_token = t_block.repeat_interleave(B_blk, dim=1)[:, :L]
    rand_u = torch.rand((B, L), device=device)
    mask_positions = (rand_u < t_per_token) & can_mask

    for b in range(B):
        if not mask_positions[b].any() and can_mask[b].any():
            valid_idx = torch.nonzero(can_mask[b], as_tuple=False).squeeze(-1)
            chosen = valid_idx[torch.randint(0, len(valid_idx), (1,), device=device)]
            mask_positions[b, chosen] = True
            t_per_token[b, chosen] = max(cfg.min_mask_rate, float(t_per_token[b, chosen].item()))

    x_t = x_0.clone()
    x_t[mask_positions] = cfg.mask_token_id
    t_seq = t_per_token.mean(dim=1).clamp(min=0.05)
    return x_t, mask_positions, t_seq


_PARSE_RE = re.compile(r"watt:\s*(\d+)\s*\nms:\s*(\d+)")

def parse_watt_ms_prediction(generated_text: str, prompt_text: str) -> Tuple[bool, Optional[int], Optional[int]]:
    """
    Extrae los enteros predichos para watt y ms de la región generada tras el prompt.
    Devuelve (formato_estricto_valido, pred_watt, pred_ms).
    """
    continuation = generated_text[len(prompt_text):] if generated_text.startswith(prompt_text) else generated_text
    match = _PARSE_RE.search(continuation)
    if not match:
        return False, None, None
    try:
        w_val = int(match.group(1))
        ms_val = int(match.group(2))
        strict_ok = continuation.strip().startswith("watt:") and ("\nms:" in continuation)
        return strict_ok, w_val, ms_val
    except ValueError:
        return False, None, None
