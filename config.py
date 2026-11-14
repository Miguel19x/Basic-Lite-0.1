from dataclasses import dataclass

@dataclass
class HybridConfig:
    # Hardware objetivo: Intel Core i3-1115G4 (2 núcleos / 4 hilos, 16 GB RAM, Linux)
    seed: int = 42
    cpu_threads: int = 4
    device: str = "cpu"

    # Vocabulario por bytes (0..255) + 4 tokens especiales
    vocab_size: int = 260
    pad_token_id: int = 256
    mask_token_id: int = 257
    bos_token_id: int = 258
    eos_token_id: int = 259

    # Dimensiones de red y GQA (iLLaDA) + Bloques Causales (CDLM)
    max_seq_len: int = 80            # 5 bloques de 16 tokens
    block_size: int = 16             # Bloques de 16 tokens: causal entre bloques, bidireccional dentro
    d_model: int = 192
    n_heads: int = 4                 # Cabezas de consulta (Q)
    n_kv_heads: int = 2              # Cabezas compartidas de clave/valor (K, V) -> GQA 2:1
    d_ff_expert: int = 384

    # Principio 1: Pesos ternarios (BitLinear W1.58A8) + Inferencia entera BITCOS
    activation_bits: int = 8
    use_integer_inference: bool = False  # Si True en eval(), opera con códigos {-1,0,+1} e int8
    eps: float = 1e-6

    # Principio 2: Difusión discreta enmascarada + Consistencia CDLM
    min_mask_rate: float = 0.15
    max_mask_rate: float = 1.0
    diffusion_steps_per_block: int = 4   # Pasos de difusión por bloque activo
    consistency_loss_weight: float = 0.15

    # Principio 3: Recursión TRM/TRLM (estados a, y_r, z_r) + Mixture-of-Recursions (MoR)
    num_recursions: int = 4
    nograd_recursions: int = 1
    intermediate_loss_weight: float = 0.25
    halt_threshold: float = 0.85     # Umbral para congelar un token en inferencia
    ponder_loss_weight: float = 0.02 # Penalización suave para fomentar parada temprana cuando ya converge

    # Principio 4: Modularidad selectiva (Expert-Choice MoE)
    num_experts: int = 4
    capacity_factor: float = 2.0

    # Entrenamiento con AdamW (decaimiento de pesos desacoplado)
    batch_size: int = 16
    learning_rate: float = 1.5e-3
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.95)
    max_steps: int = 300
    eval_every: int = 50
    grad_clip: float = 1.0
    checkpoint_path: str = "trdm_v2_checkpoint.pt"
    bitcos_export_path: str = "trdm_v2_bitcos.bin"
