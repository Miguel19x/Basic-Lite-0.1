import random
from typing import List, Tuple, Set
import torch
from torch.utils.data import Dataset
from config import HybridConfig


class ProceduralCompositionalDataset(Dataset):
    """
    Dataset composicional estructurado para evaluar generalización real fuera de distribución (OOD).
    Genera tuplas de configuración técnica a partir de 4 ejes independientes:
      - 6 Nodos x 5 Perfiles x 4 Redundancias x 4 Cargas = 480 combinaciones posibles.
    Reserva un 20% de las combinaciones (96 tuplas) de forma estricta para validación y test.
    El modelo jamás ve esas combinaciones en entrenamiento.

    Supervisión de longitud estilo LLaDA:
      - Las posiciones de respuesta se supervisan hacia los bytes objetivo.
      - La posición final se supervisa hacia el token [EOS].
      - Las posiciones posteriores hasta el límite de secuencia se supervisan hacia [PAD].
      Esto entrena al modelo a predecir el fin de secuencia de forma autónoma.
    """

    NODOS = ["server", "switch", "storage", "compute", "sensor", "router"]
    PERFILES = [
        ("eco", 2, 8),       # (nombre, base_potencia, base_latencia)
        ("base", 5, 6),
        ("turbo", 10, 3),
        ("edge", 4, 2),
        ("batch", 8, 10),
    ]
    REDUNDANCIAS = [
        ("simple", 1.0, 0),  # (nombre, mult_potencia, delta_latencia)
        ("dual", 1.5, 1),
        ("triple", 2.0, 2),
        ("mesh", 2.5, -1),
    ]
    CARGAS = [
        ("baja", 1),
        ("media", 2),
        ("alta", 3),
        ("pico", 5),
    ]

    @classmethod
    def get_held_out_tuples(cls) -> Set[Tuple[str, str, str]]:
        # Reservamos combinaciones sistemáticas nodo-perfil-carga para OOD
        return {
            ("server", "turbo", "pico"),
            ("server", "batch", "alta"),
            ("compute", "turbo", "alta"),
            ("compute", "eco", "pico"),
            ("router", "edge", "pico"),
            ("router", "mesh", "alta"),
            ("storage", "batch", "media"),
            ("storage", "turbo", "pico"),
            ("switch", "base", "alta"),
            ("sensor", "eco", "baja"),
        }

    def __init__(self, cfg: HybridConfig, split: str = "train", num_samples: int = 2400):
        super().__init__()
        self.cfg = cfg
        rng = random.Random(cfg.seed if split == "train" else cfg.seed + 9999)
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        held_out = self.get_held_out_tuples()

        while len(self.samples) < num_samples:
            nodo = rng.choice(self.NODOS)
            perfil, base_pot, base_lat = rng.choice(self.PERFILES)
            red, red_pot, red_lat = rng.choice(self.REDUNDANCIAS)
            carga, mult_carga = rng.choice(self.CARGAS)

            is_reserved = (nodo, perfil, carga) in held_out
            if (split == "train" and is_reserved) or (split in ("val", "test") and not is_reserved):
                continue

            # Fórmulas de correlación física deterministas
            nodo_bonus = 12 if nodo in ("server", "compute") else 3
            vatios = int(round((base_pot * mult_carga * red_pot) + nodo_bonus))
            latencia = max(1, int(round((base_lat * mult_carga) + red_lat - (2 if nodo in ("switch", "router") else 0))))

            prompt_str = f"nodo: {nodo}\nperfil: {perfil}\nred: {red}\ncarga: {carga}\n"
            target_str = f"watt: {vatios}\nms: {latencia}\n"

            p_bytes = list(prompt_str.encode("utf-8"))
            r_bytes = list(target_str.encode("utf-8"))

            prompt_len = 1 + len(p_bytes)
            target_len = len(r_bytes)
            eos_pos = prompt_len + target_len

            # Construir secuencia [BOS] + Prompt + Respuesta + [EOS] + [PAD]...
            seq = [cfg.bos_token_id] + p_bytes + r_bytes + [cfg.eos_token_id]
            seq = seq[: cfg.max_seq_len]
            pad_len = cfg.max_seq_len - len(seq)
            seq = seq + [cfg.pad_token_id] * pad_len

            x_0 = torch.tensor(seq, dtype=torch.long)
            can_mask = torch.zeros(cfg.max_seq_len, dtype=torch.bool)

            # Supervisión de longitud estilo LLaDA:
            # Se permite enmascarar desde el inicio de la respuesta hasta el final de la secuencia
            # Los tokens objetivo supervisan el texto; eos_pos supervisa EOS; el resto supervisa PAD.
            can_mask[prompt_len : cfg.max_seq_len] = True

            self.samples.append((x_0, can_mask))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def apply_discrete_diffusion_corruption(
    x_0: torch.Tensor,
    can_mask: torch.Tensor,
    cfg: HybridConfig,
    fixed_t: float = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Corrompe x_0 muestreando t ~ U(min_mask_rate, max_mask_rate) y reemplazando
    cada posición de respuesta elegible por [MASK] con probabilidad t.
    """
    B, L = x_0.shape
    device = x_0.device

    if fixed_t is not None:
        t = torch.full((B,), fixed_t, device=device)
    else:
        t = torch.empty(B, device=device).uniform_(cfg.min_mask_rate, cfg.max_mask_rate)

    rand_u = torch.rand(B, L, device=device)
    mask_positions = (rand_u < t.unsqueeze(1)) & can_mask

    # Garantizar al menos una posición enmascarada para evitar gradientes nulos
    for b in range(B):
        if not mask_positions[b].any() and can_mask[b].any():
            valid_indices = torch.nonzero(can_mask[b], as_tuple=False).squeeze(-1)
            chosen = valid_indices[torch.randint(0, len(valid_indices), (1,))]
            mask_positions[b, chosen] = True

    x_t = x_0.clone()
    x_t[mask_positions] = cfg.mask_token_id
    return x_t, mask_positions, t
