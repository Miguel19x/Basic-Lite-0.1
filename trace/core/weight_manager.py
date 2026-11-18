# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

from collections import OrderedDict
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn


class WeightMovementManager:
    """
    Administrador explícito del movimiento de pesos (Weight Page & Cache Manager).
    Implementa:
    1. Presupuesto estricto de memoria RAM para expertos (Caché LRU).
    2. Enrutamiento consciente de memoria:
       S(e) = U(e) - lambda * C_lectura(e) - mu * C_computo(e)
    3. Mecanismo de escape: Si la utilidad de un experto ausente supera el umbral,
       se autoriza su carga desde SSD/almacenamiento secundario.
    4. Anticipación de lectura (Prefetching) para solapar transferencias de I/O.
    5. Telemetría exhaustiva de bytes transferidos por token, tasas de acierto y latencias.
    """

    def __init__(
        self,
        max_resident_experts: int = 2,
        lambda_read_cost: float = 0.35,
        mu_compute_cost: float = 0.05,
        escape_threshold: float = 0.70,
        expert_byte_size: int = 4096 * 8,  # Tamaño en bytes de un experto ternario típico (ej. 32 KB)
        simulated_bandwidth_gb_s: float = 2.0,  # Límite físico: SSD PCIe NVMe de consumo ~2 GB/s
    ):
        self.max_resident_experts = max(1, max_resident_experts)
        self.lambda_read_cost = lambda_read_cost
        self.mu_compute_cost = mu_compute_cost
        self.escape_threshold = escape_threshold
        self.expert_byte_size = expert_byte_size
        self.simulated_bandwidth_gb_s = simulated_bandwidth_gb_s

        # Caché LRU: expert_id -> module / weights
        self.lru_cache: OrderedDict[int, nn.Module] = OrderedDict()
        self.expert_registry: Dict[int, nn.Module] = {}

        # Cola de prefetch
        self.prefetch_queue: Set[int] = set()

        # Contadores de telemetría auditables
        self.total_queries: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.evictions: int = 0
        self.bytes_transferred: int = 0
        self.transfer_time_ms: float = 0.0
        self.escapes_triggered: int = 0

    def register_expert_source(self, expert_id: int, expert_module: nn.Module) -> None:
        """Registra el repositorio de expertos disponible (disco o almacenamiento base)."""
        self.expert_registry[expert_id] = expert_module

    def is_resident(self, expert_id: int) -> bool:
        """Determina si un experto está cargado físicamente en la memoria de trabajo RAM."""
        return expert_id in self.lru_cache

    def get_read_cost(self, expert_id: int) -> float:
        """
        Costo de lectura normalizado en [0, 1].
        0.0 si ya reside en RAM.
        > 0 si requiere I/O desde SSD.
        """
        if self.is_resident(expert_id):
            return 0.0
        # Costo relativo proporcional a la transferencia de bytes
        # Normalizado para la función de selección del router
        return 1.0

    def compute_memory_aware_scores(
        self,
        raw_utilities: torch.Tensor,  # [B, E]
        allow_escape: bool = True,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Calcula las puntuaciones de enrutamiento conscientes del presupuesto de memoria:
          S(e) = U(e) - lambda * C_lectura(e) - mu * C_computo(e)
        donde C_lectura(e) = 0 si e in RAM, y 1 si e requiere lectura de SSD.
        """
        B, num_experts = raw_utilities.shape
        scores = raw_utilities.clone()
        escaped_experts: List[int] = []

        resident_set = set(self.lru_cache.keys())

        for e_idx in range(num_experts):
            if e_idx not in resident_set:
                c_read = 1.0
                c_comp = 0.10
                penalty = self.lambda_read_cost * c_read + self.mu_compute_cost * c_comp
                scores[:, e_idx] = scores[:, e_idx] - penalty

                # Mecanismo de escape: si la utilidad intrínseca U(e) es excepcionalmente alta
                # o el experto es estrictamente necesario, restaurar su prioridad
                max_u = float(raw_utilities[:, e_idx].max().item())
                if allow_escape and max_u >= self.escape_threshold:
                    scores[:, e_idx] = raw_utilities[:, e_idx]
                    escaped_experts.append(e_idx)
                    self.escapes_triggered += 1

        return scores, escaped_experts

    def acquire_expert(self, expert_id: int) -> nn.Module:
        """
        Obtiene un experto para ejecución.
        Si está en caché LRU, actualiza su posición reciente (Cache Hit).
        Si no, lo trae desde almacenamiento (Cache Miss), contabiliza bytes leídos
        y si excede el presupuesto de RAM desaloja el más antiguo.
        """
        self.total_queries += 1

        if expert_id in self.lru_cache:
            self.cache_hits += 1
            self.lru_cache.move_to_end(expert_id)
            return self.lru_cache[expert_id]

        # Cache Miss: Carga física desde almacenamiento
        self.cache_misses += 1
        self.bytes_transferred += self.expert_byte_size
        io_time_sec = self.expert_byte_size / (self.simulated_bandwidth_gb_s * 1e9)
        self.transfer_time_ms += io_time_sec * 1000.0

        expert_module = self.expert_registry.get(expert_id)
        if expert_module is None:
            raise KeyError(f"Experto {expert_id} no registrado en el almacenamiento")

        # Control del presupuesto estricto de RAM
        if len(self.lru_cache) >= self.max_resident_experts:
            # Desalojar el experto menos recientemente usado (LRU)
            evicted_id, _ = self.lru_cache.popitem(last=False)
            self.evictions += 1

        self.lru_cache[expert_id] = expert_module
        return expert_module

    def prefetch_expert(self, expert_id: int) -> None:
        """Encola y anticipa la carga de un experto para el siguiente paso/bloque."""
        if expert_id not in self.lru_cache and expert_id in self.expert_registry:
            self.acquire_expert(expert_id)

    def get_telemetry(self) -> Dict[str, Any]:
        """Reporta estadísticas de movimiento de pesos para auditoría."""
        hit_rate = (
            float(self.cache_hits) / float(max(1, self.total_queries))
        ) * 100.0
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate_pct": round(hit_rate, 2),
            "resident_experts": list(self.lru_cache.keys()),
            "max_resident_budget": self.max_resident_experts,
            "evictions_count": self.evictions,
            "bytes_transferred_kb": round(self.bytes_transferred / 1024.0, 2),
            "simulated_io_latency_ms": round(self.transfer_time_ms, 3),
            "escapes_triggered": self.escapes_triggered,
        }

    def reset_telemetry(self) -> None:
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0
        self.bytes_transferred = 0
        self.transfer_time_ms = 0.0
        self.escapes_triggered = 0
