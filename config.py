from dataclasses import dataclass

@dataclass
class HybridConfig:
    # Hardware objetivo: Intel Core i3-1115G4 (2 núcleos físicos / 4 hilos lógicos, 16 GB RAM, Linux)
    seed: int = 42
    cpu_threads: int = 4
    device: str = "cpu"

    # Vocabulario a nivel de bytes (0..255) + 4 tokens especiales
    vocab_size: int = 260
    pad_token_id: int = 256
    mask_token_id: int = 257
    bos_token_id: int = 258
    eos_token_id: int = 259

    # Dimensiones: 5 bloques de 16 tokens exactos (L = 80)
    # Atención causal entre bloques y bidireccional dentro del bloque
    max_seq_len: int = 80
    block_size: int = 16             # 5 bloques de 16 tokens
    d_model: int = 192
    n_heads: int = 4                 # Cabezas de consulta Q
    n_kv_heads: int = 2              # Cabezas GQA 2:1 para K y V
    d_ff_expert: int = 384

    # Principio 1: Pesos ternarios W1.58A8 + Empaquetado BITCOSv2
    activation_bits: int = 8
    use_integer_inference: bool = False
    eps: float = 1e-6

    # Principio 2: Difusión discreta enmascarada
    min_mask_rate: float = 0.05
    max_mask_rate: float = 1.0
    diffusion_steps_per_block: int = 4
    consistency_loss_weight: float = 0.15

    # Principio 3: Recursión TRM/TRLM y configuración para Hipótesis H1
    # H1: Entrenamos con R=2 y evaluamos en inferencia con R in {1, 2, 4, 8} sin reentrenar
    num_recursions: int = 2          # Recursiones en entrenamiento (barato y rápido)
    nograd_recursions: int = 0       # Sin truncamiento artificial en R=2 para que todo reciba gradiente
    intermediate_loss_weight: float = 0.25
    halt_threshold: float = 0.85
    ponder_loss_weight: float = 0.02

    # Principio 4: Modularidad Expert-Choice intra-bloque (sin fugas ni contaminación de PAD)
    num_experts: int = 4
    capacity_factor: float = 2.0     # Capacidad media de tokens procesados por experto

    # Entrenamiento con AdamW y planificador BitNet de dos etapas
    batch_size: int = 16
    learning_rate: float = 1.5e-3
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.95)
    max_steps: int = 300
    eval_every: int = 50
    grad_clip: float = 1.0
    checkpoint_path: str = "trdm_v3_checkpoint.pt"
    bitcos_export_path: str = "trdm_v3_bitcos.bin"
