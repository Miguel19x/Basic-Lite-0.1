# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Módulo de acceso para TRDM (Basic-Lite 0.2).
Exporta tanto la arquitectura TRDM-v4 previa como la nueva arquitectura TRDM-v5
diseñada bajo el dictamen del auditor para modelos 30B con memoria y contexto selectivos.
"""

from trdm.core.types import ModelOutput
from trdm.core.weight_manager import WeightMovementManager
from trdm.core.context_manager import HierarchicalContextManager
from trdm.layers.norm import RMSNorm
from trdm.layers.rotary import RotaryEmbedding
from trdm.layers.embeddings import SinusoidalStepEncoding
from trdm.layers.bitlinear import BitLinear
from trdm.layers.byte_patcher import HierarchicalBytePatcher
from trdm.modules.attention import BlockCausalGQATernaryAttention, build_block_causal_mask
from trdm.modules.mla_attention import MultiHeadLatentAttention
from trdm.modules.moe import TernaryExpert, ExpertChoiceMoE
from trdm.modules.paged_moe import MemoryAwarePagedMoE
from trdm.modules.recursive_core import SharedDualStateRecursiveCore
from trdm.layers.abacus import AbacusEmbedding, extract_abacus_ranks
from trdm.models.trdm_v4 import TernaryRecursiveDiffusionLM
from trdm.models.trdm_v5 import TernaryRecursiveDiffusionLM_v5, SharedResidentTernaryCore
from trdm.models.trdm_v6_causal import TRDMv6CausalModel
from trdm.serialization.bitcos import export_bitcos_archive, load_bitcos_archive
from trdm.serialization.paged_bitcos import PagedBitCOSSerializer, PagedBitCOSReader

__all__ = [
    "ModelOutput",
    "WeightMovementManager",
    "HierarchicalContextManager",
    "RMSNorm",
    "RotaryEmbedding",
    "SinusoidalStepEncoding",
    "BitLinear",
    "AbacusEmbedding",
    "extract_abacus_ranks",
    "HierarchicalBytePatcher",
    "BlockCausalGQATernaryAttention",
    "MultiHeadLatentAttention",
    "build_block_causal_mask",
    "TernaryExpert",
    "ExpertChoiceMoE",
    "MemoryAwarePagedMoE",
    "SharedDualStateRecursiveCore",
    "SharedResidentTernaryCore",
    "TernaryRecursiveDiffusionLM",
    "TernaryRecursiveDiffusionLM_v5",
    "TRDMv6CausalModel",
    "export_bitcos_archive",
    "load_bitcos_archive",
    "PagedBitCOSSerializer",
    "PagedBitCOSReader",
]
