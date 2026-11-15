from dataclasses import dataclass

@dataclass
class HybridConfig:
    # Entorno objetivo: Intel Core i3-1115G4 (2 núcleos / 4 hilos, 16 GB RAM, Linux)
    seed: int = 42
    cpu_threads: int = 4
    device: str = "cpu"

    # Vocabulario por bytes (0..255) + 4 tokens especiales
    vocab_size: int = 260
    pad_token_id: int = 256
    mask_token_id: int = 257
    bos_token_id: int = 258
    eos_token_id: int = 259

    # Dimensiones de secuencia y bloques causales (CDLM + GQA)
    max_seq_len: int = 80
    block_size: int = 16             # 5 bloques de 16 bytes
    d_model: int = 192
    n_heads: int = 4
    n_kv_heads: int = 2
    d_ff_expert: int = 384

    # Principio 1: Capas ternarias BitLinear (W1.58A8)
    activation_bits: int = 8
    use_integer_inference: bool = False
    use_add_sub_inference: bool = False
    eps: float = 1e-6

    # Principio 2: Difusión discreta enmascarada + Consistencia CDLM
    min_mask_rate: float = 0.15
    max_mask_rate: float = 1.0
    diffusion_steps_per_block: int = 4
    consistency_loss_weight: float = 0.10

    # Principio 3: Recursión con codificación sinusoidal continua (sin embeddings huérfanos)
    num_recursions: int = 2          # Profundidad base en entrenamiento (H1 evalúa R in {1,2,4,8})
    randomize_train_recursions: bool = False  # Si True, muestrea R_train en {1, 2, 3}
    nograd_recursions: int = 0       # 0 para que todos los pasos reciban gradiente limpio
    intermediate_loss_weight: float = 0.25
    halt_threshold: float = 0.90
    ponder_loss_weight: float = 0.01

    # Ponderación contra el techo trivial (priorizar bytes de dígitos sobre plantilla y PAD)
    digit_token_weight: float = 6.0  # Peso x6 en posiciones de dígitos ('0'..'9')
    template_token_weight: float = 1.0
    pad_token_weight: float = 0.20   # Peso reducido en PAD post-EOS

    # Principio 4: Modularidad selectiva intra-bloque (Expert-Choice MoE)
    num_experts: int = 4
    capacity_factor: float = 2.0

    # Optimización AdamW (decaimiento desacoplado)
    batch_size: int = 16
    learning_rate: float = 2.0e-3
    weight_decay: float = 0.03
    betas: tuple = (0.9, 0.95)
    max_steps: int = 800             # Presupuesto ampliado para consolidar aritmética
    eval_every: int = 100
    grad_clip: float = 1.0
    checkpoint_path: str = "trdm_v4_checkpoint.pt"
    bitcos_export_path: str = "trdm_v4_bitcos.bin"
