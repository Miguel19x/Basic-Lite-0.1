# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from trace.layers.norm import RMSNorm
from trace.layers.embeddings import SinusoidalStepEncoding
from trace.layers.bitlinear import BitLinear
from trace.modules.attention import BlockCausalGQATernaryAttention
from trace.modules.moe import ExpertChoiceMoE


class SharedDualStateRecursiveCore(nn.Module):
    """
    Núcleo recurrente TRM/TRLM con codificación sinusoidal continua de profundidad u_r = (r+1)/R
    y escalas residuales compartidas (sin parámetros dependientes de un índice r máximo fijo).
    Mantiene estado dual:
      - z_r: estado latente de razonamiento / cómputo continuo
      - y_r: propuesta de respuesta tentativa refinada en cada iteración
    """
    def __init__(self, cfg):
        super().__init__()
        self.norm_z = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.norm_y = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.norm_moe = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.norm_refine = RMSNorm(cfg.d_model, eps=cfg.eps)

        self.anchor_proj = BitLinear(cfg.d_model, cfg.d_model, eps=cfg.eps)
        self.attn = BlockCausalGQATernaryAttention(cfg)
        self.moe = ExpertChoiceMoE(cfg)
        self.answer_updater = BitLinear(cfg.d_model, cfg.d_model, eps=cfg.eps)

        self.halt_gate = nn.Linear(cfg.d_model, 1)
        nn.init.constant_(self.halt_gate.bias, -1.5)

        # Codificador continuo de progreso recursivo u_r = (r + 1) / R
        self.step_encoder = SinusoidalStepEncoding(cfg.d_model)
        self.noise_mlp = nn.Sequential(
            nn.Linear(2, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        # Escalas residuales compartidas entre todos los pasos r (100% entrenadas siempre)
        self.alpha_z = nn.Parameter(torch.tensor(0.60))
        self.beta_y = nn.Parameter(torch.tensor(0.60))
        self.carry_gate = nn.Parameter(torch.tensor(0.0))

    def precompute_anchor(
        self,
        a: torch.Tensor,
        record_telemetry: bool = False,
        integer_mode: bool = False,
        add_sub_mode: bool = False,
    ) -> torch.Tensor:
        return self.anchor_proj(a, record_telemetry, integer_mode, add_sub_mode)

    def forward(
        self,
        z_r: torch.Tensor,
        y_r: torch.Tensor,
        anchor_feat: torch.Tensor,
        t_noise: torch.Tensor,
        delta_t: torch.Tensor,
        r_step: int,
        total_recursions: int,
        non_pad_mask: Optional[torch.Tensor],
        active_mask: torch.Tensor,
        prev_node_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
        record_telemetry: bool = False,
        integer_mode: bool = False,
        add_sub_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if t_noise.dim() == 1:
            noise_pair = torch.stack([t_noise, delta_t], dim=-1)
            e_t = self.noise_mlp(noise_pair).unsqueeze(1)
        elif t_noise.dim() == 2:
            noise_pair = torch.stack([t_noise, delta_t], dim=-1)
            e_t = self.noise_mlp(noise_pair)
        else:
            e_t = self.noise_mlp(torch.stack([t_noise, delta_t], dim=-1))

        # Progreso normalizado en (0, 1] y amortiguación de paso 2 / (R + 1)
        u_norm = float(r_step + 1) / float(max(1, total_recursions))
        step_scale = 2.0 / float(total_recursions + 1)
        e_r = self.step_encoder(u_norm, z_r.device)

        joint_state = self.norm_z(z_r) + self.norm_y(y_r) + anchor_feat + e_t + e_r
        attn_out, new_kv_cache = self.attn(
            joint_state,
            kv_cache=kv_cache,
            start_pos=start_pos,
            record_telemetry=record_telemetry,
            integer_mode=integer_mode,
            add_sub_mode=add_sub_mode,
        )
        eff_alpha = self.alpha_z * step_scale
        eff_beta = self.beta_y * step_scale

        z_mid = z_r + active_mask * eff_alpha * attn_out
        moe_out, curr_node_ids = self.moe(
            self.norm_moe(z_mid),
            t_noise=t_noise,
            non_pad_mask=non_pad_mask,
            active_mask=active_mask,
            r_step=r_step,
            prev_node_ids=prev_node_ids,
            record_telemetry=record_telemetry,
            integer_mode=integer_mode,
            add_sub_mode=add_sub_mode,
        )
        z_next = z_mid + active_mask * eff_alpha * moe_out

        fusion_yz = self.norm_refine(y_r + z_next)
        delta_y = self.answer_updater(fusion_yz, record_telemetry, integer_mode, add_sub_mode)
        y_next = y_r + active_mask * eff_beta * F.silu(delta_y)

        halt_prob = torch.sigmoid(self.halt_gate(self.norm_refine(z_next + y_next)))
        return z_next, y_next, halt_prob, new_kv_cache, curr_node_ids
