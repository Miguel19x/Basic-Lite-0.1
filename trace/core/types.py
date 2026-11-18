# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch


@dataclass
class ModelOutput:
    """
    Salida estructurada para el modelo TernaryRecursiveDiffusionLM.
    Permite desacoplar proyecciones de vocabulario, estados latentes y telemetría.
    """
    final_logits: torch.Tensor
    intermediate_logits: List[torch.Tensor] = field(default_factory=list)
    halt_probs: List[torch.Tensor] = field(default_factory=list)
    kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    carried_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    last_hidden_state: Optional[torch.Tensor] = None

    # Compatibilidad con desempaquetado de tupla clásico:
    # final_logits, intermediate_logits, halt_probs, kv_caches, carried_states = model(...)
    def __iter__(self):
        yield self.final_logits
        yield self.intermediate_logits
        yield self.halt_probs
        yield self.kv_caches
        yield self.carried_states

    def __getitem__(self, idx):
        return tuple(self)[idx]
