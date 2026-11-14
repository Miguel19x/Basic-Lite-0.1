import random
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from config import HybridConfig


class ProceduralNaturalTextDataset(Dataset):
    """
    Dataset de texto en lenguaje natural en espanol estructurado en pares de
    Pregunta y Respuesta sobre conocimiento factual (geografia, colores, animales,
    antónimos, profesiones y ciencia).

    Permite interpretar de forma directa y visual si el modelo genera palabras
    reales y sintacticamente correctas ronda a ronda durante la difusion discreta.
    """

    HECHOS = [
        # (categoria, pregunta_base, pregunta_variante, respuesta)
        ("geografia", "Cual es la capital de Francia", "Dime la capital de Francia", "Paris"),
        ("geografia", "Cual es la capital de Espana", "Dime la capital de Espana", "Madrid"),
        ("geografia", "Cual es la capital de Italia", "Dime la capital de Italia", "Roma"),
        ("geografia", "Cual es la capital de Alemania", "Dime la capital de Alemania", "Berlin"),
        ("geografia", "Cual es la capital de Japon", "Dime la capital de Japon", "Tokio"),
        ("geografia", "Cual es la capital de Portugal", "Dime la capital de Portugal", "Lisboa"),

        ("color", "De que color es el cielo despejado", "Que color tiene el cielo claro", "Azul"),
        ("color", "De que color es la hierba fresca", "Que color tiene la hierba viva", "Verde"),
        ("color", "De que color es la nieve limpia", "Que color tiene la nieve pura", "Blanca"),
        ("color", "De que color es el carbon quemado", "Que color tiene el carbon puro", "Negro"),
        ("color", "De que color es el Sol brillante", "Que color tiene el Sol de dia", "Amarillo"),

        ("animal", "Que animal ladra y cuida la casa", "Cual es el animal que ladra", "El perro"),
        ("animal", "Que animal maulla y ronronea suave", "Cual es el animal que maulla", "El gato"),
        ("animal", "Que animal muge en el prado", "Cual es el animal que muge", "La vaca"),
        ("animal", "Que animal relincha en el campo", "Cual es el animal que relincha", "El caballo"),

        ("opuesto", "Cual es el opuesto de frio", "Dime el antónimo de frio", "Caliente"),
        ("opuesto", "Cual es el opuesto de rapido", "Dime el antónimo de rapido", "Lento"),
        ("opuesto", "Cual es el opuesto de alto", "Dime el antónimo de alto", "Bajo"),
        ("opuesto", "Cual es el opuesto de grande", "Dime el antónimo de grande", "Pequeno"),

        ("profesion", "Que profesional apaga los incendios", "Quien apaga el fuego en la ciudad", "El bombero"),
        ("profesion", "Que profesional cura a los enfermos", "Quien atiende la salud del paciente", "El medico"),
        ("profesion", "Que profesional ensena en la escuela", "Quien da clases a los alumnos", "El maestro"),

        ("ciencia", "Que sentido usamos para ver", "Cual es el sentido de la vision", "La vista"),
        ("ciencia", "Que sentido usamos para oir", "Cual es el sentido de la audicion", "El oido"),
        ("ciencia", "En que estacion hace mucho calor", "Cual es la epoca mas calurosa del ano", "El verano"),
        ("ciencia", "En que estacion nieva y hace frio", "Cual es la epoca mas fria del ano", "El invierno"),
    ]

    def __init__(self, cfg: HybridConfig, split: str = "train", num_samples: int = 2400):
        super().__init__()
        self.cfg = cfg
        rng = random.Random(cfg.seed if split == "train" else cfg.seed + 999)
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for _ in range(num_samples):
            cat, preg_base, preg_var, resp = rng.choice(self.HECHOS)

            # Para entrenamiento usamos la plantilla principal; para validacion usamos variantes
            if split == "train":
                p_text = f"Pregunta: {preg_base}?\nRespuesta: "
            else:
                p_text = f"P: {preg_var}?\nR: "

            r_text = f"{resp}.\n"

            p_bytes = list(p_text.encode("utf-8"))
            r_bytes = list(r_text.encode("utf-8"))

            seq = [cfg.bos_token_id] + p_bytes + r_bytes + [cfg.eos_token_id]
            prompt_len = 1 + len(p_bytes)
            target_len = len(r_bytes) + 1  # Incluye el token [EOS]

            seq = seq[: cfg.max_seq_len]
            pad_len = cfg.max_seq_len - len(seq)
            seq = seq + [cfg.pad_token_id] * pad_len

            x_0 = torch.tensor(seq, dtype=torch.long)
            can_mask = torch.zeros(cfg.max_seq_len, dtype=torch.bool)
            # Solo se enmascaran las posiciones de la respuesta y su [EOS]
            can_mask[prompt_len : min(prompt_len + target_len, cfg.max_seq_len)] = True

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
    cada posicion de respuesta por [MASK] con probabilidad t.
    """
    B, L = x_0.shape
    device = x_0.device

    if fixed_t is not None:
        t = torch.full((B,), fixed_t, device=device)
    else:
        t = torch.empty(B, device=device).uniform_(cfg.min_mask_rate, cfg.max_mask_rate)

    rand_u = torch.rand(B, L, device=device)
    mask_positions = (rand_u < t.unsqueeze(1)) & can_mask

    for b in range(B):
        if not mask_positions[b].any() and can_mask[b].any():
            valid_indices = torch.nonzero(can_mask[b], as_tuple=False).squeeze(-1)
            chosen = valid_indices[torch.randint(0, len(valid_indices), (1,))]
            mask_positions[b, chosen] = True

    x_t = x_0.clone()
    x_t[mask_positions] = cfg.mask_token_id
    return x_t, mask_positions, t
