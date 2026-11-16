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
        self.record_meta: Dict[Tuple[bytes, bytes], Tuple[int, int, int, int]] = {}

        for n_name, b_w, b_ms in self.NODOS:
            for c_name, m_w, m_ms in self.CARGAS:
                for r_name, a_w, a_ms in self.REDUNDANCIAS:
                    for p_name, o_w, o_ms in self.PERFILES:
                        sub_w = int(round(b_w * m_w))
                        sub_ms = int(round(b_ms * m_ms))

                        if getattr(cfg, "use_scratchpad", False):
                            # Consistencia aditiva estricta: registros disjuntos y compactos (pw y pms)
                            watt = max(5, sub_w + a_w + o_w)
                            ms = max(1, sub_ms + a_ms + o_ms)
                            target_str = f"pw: {sub_w}\npms: {sub_ms}\nwatt: {watt}\nms: {ms}\n"
                        else:
                            watt = max(5, int(round(b_w * m_w + a_w + o_w)))
                            ms = max(1, int(round(b_ms * m_ms + a_ms + o_ms)))
                            target_str = f"watt: {watt}\nms: {ms}\n"

                        if getattr(cfg, "use_explicit_operands", False):
                            # Opción A: inclusión de operandos numéricos estructurados mediante transformación fija
                            prompt_str = (
                                f"nodo: {n_name}\ncarga: {c_name}\nred: {r_name}\nperfil: {p_name}\n"
                                f"bw: {b_w} mw: {m_w} aw: {a_w} ow: {o_w} bms: {b_ms} mms: {m_ms} ams: {a_ms} oms: {o_ms}\n"
                            )
                        else:
                            prompt_str = f"nodo: {n_name}\ncarga: {c_name}\nred: {r_name}\nperfil: {p_name}\n"

                        p_bytes = prompt_str.encode("utf-8")
                        t_bytes = target_str.encode("utf-8")
                        record = (p_bytes, t_bytes, watt, ms)
                        self.record_meta[(p_bytes, t_bytes)] = (sub_w, sub_ms, watt, ms)

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
        raw_len = 1 + len(prompt_b) + len(target_b) + 1
        assert raw_len <= self.cfg.max_seq_len, (
            f"Error de longitud: secuencia completa ({raw_len} bytes) "
            f"supera max_seq_len ({self.cfg.max_seq_len}). "
            f"Abortando para prevenir truncamiento silencioso de registros o respuesta."
        )

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


from trdm.diffusion.corruption import apply_blockwise_diffusion_corruption
from trdm.data.parsers import (
    parse_watt_ms_prediction,
    parse_scratchpad_prediction,
)
