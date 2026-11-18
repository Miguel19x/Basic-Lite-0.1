# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from trdm.layers.bitlinear import BitLinear


class TernaryExpert(nn.Module):
    """Experto individual con proyecciones ternarias BitLinear y activación ReGLU / Squared-ReLU."""
    def __init__(self, d_model: int, d_ff: int, eps: float = 1e-6):
        super().__init__()
        self.up = BitLinear(d_model, d_ff, eps=eps)
        self.down = BitLinear(d_ff, d_model, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        record_telemetry: bool = False,
        integer_mode: bool = False,
        add_sub_mode: bool = False,
    ) -> torch.Tensor:
        h = F.relu(self.up(x, record_telemetry, integer_mode, add_sub_mode)).square()
        return self.down(h, record_telemetry, integer_mode, add_sub_mode)


class ExpertChoiceMoE(nn.Module):
    """
    Grafo Disperso de Módulos (Expert-Choice MoE intra-bloque):
    - Nodos = Expertos ternarios; Aristas = Matriz de adyacencia dispersa entre recursiones (graph_adj_mask).
    - Omite físicamente la ejecución sobre tokens PAD y plazas con gate <= min_gate_threshold
      mediante index_select / index_add_ sobre tensores compactos 1D, reduciendo FLOPs reales en CPU.
    """
    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.capacity_factor = cfg.capacity_factor
        self.block_size = cfg.block_size
        self.sparse_expert_dispatch = getattr(cfg, "sparse_expert_dispatch", True)
        self.use_sparse_module_graph = getattr(cfg, "use_sparse_module_graph", True)
        self.graph_out_degree = getattr(cfg, "graph_out_degree", 2)
        self.graph_capacity_decay = getattr(cfg, "graph_capacity_decay", 0.75)
        self.min_gate_threshold = getattr(cfg, "min_gate_threshold", 1e-4)

        self.router = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList([
            TernaryExpert(cfg.d_model, cfg.d_ff_expert, eps=cfg.eps)
            for _ in range(cfg.num_experts)
        ])
        self.use_shared_expert = getattr(cfg, "use_shared_expert", True)
        if self.use_shared_expert:
            self.shared_expert = TernaryExpert(cfg.d_model, cfg.d_ff_expert, eps=cfg.eps)

        # Matriz de adyacencia del grafo de módulos [num_experts, num_experts] (no persistente)
        adj = torch.zeros(cfg.num_experts, cfg.num_experts, dtype=torch.float32)
        deg = max(1, min(cfg.num_experts, self.graph_out_degree))
        for i in range(cfg.num_experts):
            for offset in range(deg):
                adj[i, (i + offset) % cfg.num_experts] = 1.0
        self.register_buffer("graph_adj_mask", adj, persistent=False)

        self.last_expert_token_counts: List[int] = [0] * cfg.num_experts
        self.last_possible_slots: int = 0

    def forward(
        self,
        x: torch.Tensor,
        t_noise: torch.Tensor,
        non_pad_mask: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
        r_step: int = 0,
        prev_node_ids: Optional[torch.Tensor] = None,
        record_telemetry: bool = False,
        integer_mode: bool = False,
        add_sub_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        shared_out = 0.0
        if self.use_shared_expert:
            shared_out = self.shared_expert(x, record_telemetry, integer_mode, add_sub_mode)
            if non_pad_mask is not None:
                shared_out = shared_out * non_pad_mask.unsqueeze(-1).float()

        B_blk = min(self.block_size, L)
        num_blocks = L // B_blk
        num_total_blocks = B * num_blocks

        with torch.no_grad():
            mean_t = float(t_noise.detach().mean().item())
            dynamic_factor = self.capacity_factor * (1.25 - 0.50 * mean_t)
            if self.use_sparse_module_graph and r_step > 0:
                dynamic_factor = dynamic_factor * (self.graph_capacity_decay ** r_step)
            k = max(1, min(B_blk, int(round((B_blk * dynamic_factor) / self.num_experts))))

        x_blk = x.view(num_total_blocks, B_blk, D)
        router_logits = self.router(x_blk)
        scores = F.softmax(router_logits, dim=-1)

        # En recursiones r > 0, restringir rutas según las aristas permitidas desde el nodo previo del grafo
        if self.use_sparse_module_graph and r_step > 0 and prev_node_ids is not None:
            allowed_edges = self.graph_adj_mask.index_select(0, prev_node_ids.view(-1)).view(num_total_blocks, B_blk, self.num_experts)
            scores = scores * allowed_edges
            scores = scores / scores.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        if active_mask is not None:
            act_blk = active_mask.view(num_total_blocks, B_blk, 1)
            scores = scores * (0.10 + 0.90 * act_blk)
            scores = scores * (act_blk > 0.05).float()

        if non_pad_mask is not None:
            valid_blk = non_pad_mask.view(num_total_blocks, B_blk, 1).float()
            scores = scores * valid_blk

        # Nodo primario asignado a cada posición para guiar las aristas de la siguiente recursión
        curr_node_ids = scores.argmax(dim=-1).detach()

        scores_t = scores.transpose(1, 2)  # [B*num_blocks, E, B_blk]
        top_gating, top_indices = torch.topk(scores_t, k=k, dim=-1)
        counts = [0] * self.num_experts
        self.last_possible_slots = int(num_total_blocks * k * self.num_experts)

        if not self.sparse_expert_dispatch:
            out_blk = torch.zeros_like(x_blk)
            for e_idx, expert in enumerate(self.experts):
                idx_e = top_indices[:, e_idx, :]
                gate_e = top_gating[:, e_idx, :].unsqueeze(-1)
                gather_idx = idx_e.unsqueeze(-1).expand(-1, -1, D)
                selected = torch.gather(x_blk, dim=1, index=gather_idx)
                expert_out = expert(selected, record_telemetry, integer_mode, add_sub_mode) * gate_e
                out_blk.scatter_add_(dim=1, index=gather_idx, src=expert_out)
                counts[e_idx] = int(num_total_blocks * k)
            if record_telemetry:
                self.last_expert_token_counts = counts
            return (out_blk.view(B, L, D) + shared_out), curr_node_ids

        # Ejecución dispersa agrupada: aplanar y omitir plazas PAD / inactivas
        x_flat = x_blk.reshape(-1, D)
        out_flat = torch.zeros_like(x_flat)
        block_offsets = (torch.arange(num_total_blocks, device=x.device) * B_blk).unsqueeze(1)

        for e_idx, expert in enumerate(self.experts):
            idx_e = top_indices[:, e_idx, :]      # [num_total_blocks, k]
            gate_e = top_gating[:, e_idx, :]      # [num_total_blocks, k]
            flat_idx = (block_offsets + idx_e).reshape(-1)
            flat_gate = gate_e.reshape(-1)

            valid_mask = flat_gate > self.min_gate_threshold
            active_idx = torch.nonzero(valid_mask, as_tuple=True)[0]
            num_active = int(active_idx.numel())
            counts[e_idx] = num_active

            if num_active > 0:
                token_indices = flat_idx.index_select(0, active_idx)
                gates_active = flat_gate.index_select(0, active_idx).unsqueeze(-1)
                selected = x_flat.index_select(0, token_indices)
                expert_out = expert(selected, record_telemetry, integer_mode, add_sub_mode) * gates_active
                out_flat.index_add_(0, token_indices, expert_out)

        if record_telemetry:
            self.last_expert_token_counts = counts
        return (out_flat.view(B, L, D) + shared_out), curr_node_ids
