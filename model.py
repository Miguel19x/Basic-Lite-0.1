"""
Módulo de compatibilidad para TRDM-v4.
Re-exporta la arquitectura modularizada desde el paquete 'trdm'.
"""

from trdm.core.types import ModelOutput
from trdm.layers.norm import RMSNorm
from trdm.layers.rotary import RotaryEmbedding
from trdm.layers.embeddings import SinusoidalStepEncoding
from trdm.layers.bitlinear import BitLinear
from trdm.modules.attention import BlockCausalGQATernaryAttention, build_block_causal_mask
from trdm.modules.moe import TernaryExpert, ExpertChoiceMoE
from trdm.modules.recursive_core import SharedDualStateRecursiveCore
from trdm.models.trdm_v4 import TernaryRecursiveDiffusionLM
from trdm.serialization.bitcos import export_bitcos_archive, load_bitcos_archive

__all__ = [
    "ModelOutput",
    "RMSNorm",
    "RotaryEmbedding",
    "SinusoidalStepEncoding",
    "BitLinear",
    "BlockCausalGQATernaryAttention",
    "build_block_causal_mask",
    "TernaryExpert",
    "ExpertChoiceMoE",
    "SharedDualStateRecursiveCore",
    "TernaryRecursiveDiffusionLM",
    "export_bitcos_archive",
    "load_bitcos_archive",
]
