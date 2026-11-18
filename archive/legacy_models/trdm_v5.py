# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from trdm.core.context_manager import HierarchicalContextManager
from trdm.core.types import ModelOutput
from trdm.core.weight_manager import WeightMovementManager
from trdm.layers.bitlinear import BitLinear
from trdm.layers.byte_patcher import HierarchicalBytePatcher
from trdm.layers.embeddings import SinusoidalStepEncoding
from trdm.layers.norm import RMSNorm
from trdm.modules.mla_attention import MultiHeadLatentAttention
from trdm.modules.paged_moe import MemoryAwarePagedMoE
from trdm.serialization.paged_bitcos import PagedBitCOSSerializer


class SharedResidentTernaryCore(nn.Module):
    """
    Núcleo ternario compartido residente de TRDM-v5:
    - Multi-Head Latent Attention (MLA) para compresión de caché KV activa.
    - MoE con paginación de expertos y enrutamiento consciente de memoria RAM/SSD.
    - Dinámica dual de razonamiento (z_r) y respuesta (y_r) con escala residual compartida.
    """

    def __init__(self, cfg, weight_manager: Optional[WeightMovementManager] = None):
        super().__init__()
        self.cfg = cfg
        self.norm_z = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.norm_y = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.norm_moe = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.norm_refine = RMSNorm(cfg.d_model, eps=cfg.eps)

        self.anchor_proj = BitLinear(cfg.d_model, cfg.d_model, eps=cfg.eps)
        self.attn = MultiHeadLatentAttention(cfg)
        self.paged_moe = MemoryAwarePagedMoE(cfg, weight_manager=weight_manager)
        self.answer_updater = BitLinear(cfg.d_model, cfg.d_model, eps=cfg.eps)

        self.halt_gate = nn.Linear(cfg.d_model, 1)
        nn.init.constant_(self.halt_gate.bias, -1.5)

        self.step_encoder = SinusoidalStepEncoding(cfg.d_model)
        self.noise_mlp = nn.Sequential(
            nn.Linear(2, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.alpha_z = nn.Parameter(torch.tensor(0.60))
        self.beta_y = nn.Parameter(torch.tensor(0.60))
        self.carry_gate = nn.Parameter(torch.tensor(0.0))

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
        latent_kv_cache: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        record_telemetry: bool = False,
        integer_mode: bool = False,
        add_sub_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if t_noise.dim() == 1:
            noise_pair = torch.stack([t_noise, delta_t], dim=-1)
            e_t = self.noise_mlp(noise_pair).unsqueeze(1)
        elif t_noise.dim() == 2:
            L_in = t_noise.shape[1]
            L_core = z_r.shape[1]
            if L_in != L_core and L_in % L_core == 0:
                step_down = L_in // L_core
                t_down = t_noise[:, ::step_down]
                dt_down = delta_t[:, ::step_down] if delta_t.dim() == 2 else delta_t
            else:
                t_down, dt_down = t_noise, delta_t
            noise_pair = torch.stack([t_down, dt_down], dim=-1)
            e_t = self.noise_mlp(noise_pair)
        else:
            e_t = self.noise_mlp(torch.stack([t_noise, delta_t], dim=-1))

        u_norm = float(r_step + 1) / float(max(1, total_recursions))
        step_scale = 2.0 / float(total_recursions + 1)
        e_r = self.step_encoder(u_norm, z_r.device)

        joint_state = self.norm_z(z_r) + self.norm_y(y_r) + anchor_feat + e_t + e_r

        # Multi-Head Latent Attention (MLA)
        attn_out, updated_latent_kv = self.attn(
            joint_state,
            latent_kv_cache=latent_kv_cache,
            start_pos=start_pos,
            record_telemetry=record_telemetry,
            integer_mode=integer_mode,
            add_sub_mode=add_sub_mode,
        )
        eff_alpha = self.alpha_z * step_scale
        eff_beta = self.beta_y * step_scale

        z_mid = z_r + active_mask * eff_alpha * attn_out

        # Memory-Aware Paged MoE
        moe_out, curr_node_ids = self.paged_moe(
            self.norm_moe(z_mid),
            t_noise=t_noise,
            non_pad_mask=non_pad_mask,
            active_mask=active_mask,
            r_step=r_step,
            record_telemetry=record_telemetry,
            integer_mode=integer_mode,
            add_sub_mode=add_sub_mode,
        )
        z_next = z_mid + active_mask * eff_alpha * moe_out

        fusion_yz = self.norm_refine(y_r + z_next)
        delta_y = self.answer_updater(fusion_yz, record_telemetry, integer_mode, add_sub_mode)
        y_next = y_r + active_mask * eff_beta * F.silu(delta_y)

        halt_prob = torch.sigmoid(self.halt_gate(self.norm_refine(z_next + y_next)))
        return z_next, y_next, halt_prob, updated_latent_kv, curr_node_ids


class TernaryRecursiveDiffusionLM_v5(nn.Module):
    """
    Arquitectura TRDM-v5 (Basic-Lite 0.2) conforme al dictamen de auditoría:
    1. Codificación local ligera por parches de bytes (HierarchicalBytePatcher).
    2. Núcleo ternario residente con recursión adaptativa (Early Exit).
    3. Enrutamiento consciente de memoria y expertos paginados RAM/SSD (WeightMovementManager).
    4. Caché de claves y valores comprimida con Multi-Head Latent Attention (MLA).
    5. Administrador explícito de información contextual jerárquica en 3 niveles (HierarchicalContextManager).
    6. Decodificación local adaptativa por difusión.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Administrador explícito del movimiento de pesos
        self.weight_manager = WeightMovementManager(
            max_resident_experts=getattr(cfg, "max_resident_experts", 2),
            lambda_read_cost=getattr(cfg, "lambda_read_cost", 0.35),
            mu_compute_cost=getattr(cfg, "mu_compute_cost", 0.05),
            escape_threshold=getattr(cfg, "memory_escape_threshold", 0.70),
        )

        # Administrador explícito de información contextual jerárquica
        self.context_manager = HierarchicalContextManager(
            active_window_size=getattr(cfg, "active_context_tokens", 64),
            compressed_segment_size=getattr(cfg, "compressed_segment_tokens", 32),
            db_path=getattr(cfg, "archive_storage_path", ":memory:"),
            embedding_dim=cfg.d_model,
        )

        # Compresor y decodificador jerárquico de bytes
        self.use_byte_patching = getattr(cfg, "use_byte_patching", True)
        self.patch_size = getattr(cfg, "byte_patch_size", 4)
        if self.use_byte_patching:
            self.byte_patcher = HierarchicalBytePatcher(
                vocab_size=cfg.vocab_size,
                d_model=cfg.d_model,
                patch_size=self.patch_size,
                encoder_dim=getattr(cfg, "patch_encoder_dim", 96),
                eps=cfg.eps,
            )
        else:
            self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Núcleo ternario residente compartido
        self.resident_core = SharedResidentTernaryCore(cfg, weight_manager=self.weight_manager)
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.eps)

        if not self.use_byte_patching:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            self.lm_head.weight = self.token_emb.weight

        self.last_recursions_executed: int = 0
        self.last_deltas: List[float] = []

    def prepare_for_eval(self) -> None:
        """Congela pesos efectivos cuantizados al entrar en modo evaluación."""
        self.eval()
        for m in self.modules():
            if isinstance(m, BitLinear):
                m.refresh_eval_cache()

    def forward(
        self,
        x_t: torch.Tensor,
        t_noise: torch.Tensor,
        delta_t: Optional[torch.Tensor] = None,
        prev_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_recursions: Optional[int] = None,
        latent_kv_caches_per_r: Optional[List[torch.Tensor]] = None,
        start_pos: int = 0,
        record_telemetry: bool = False,
        supervised_mask: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        B, L = x_t.shape
        R = num_recursions if num_recursions is not None else self.cfg.num_recursions
        int_mode = (not self.training) and self.cfg.use_integer_inference
        add_sub = (not self.training) and self.cfg.use_add_sub_inference

        if delta_t is None:
            delta_t = torch.full_like(t_noise, 1.0 / max(1, self.cfg.diffusion_steps_per_block))

        non_pad_mask = (x_t != self.cfg.pad_token_id)

        # 1. Codificación Local Ligera (Byte Patching)
        if self.use_byte_patching:
            a, pad_needed = self.byte_patcher.encode_patches(x_t)
            L_eff = a.shape[1]
            non_pad_mask_eff = non_pad_mask[:, :: self.patch_size]
            if non_pad_mask_eff.shape[1] < L_eff:
                pad_mask_extra = torch.zeros(
                    B, L_eff - non_pad_mask_eff.shape[1], dtype=torch.bool, device=x_t.device
                )
                non_pad_mask_eff = torch.cat([non_pad_mask_eff, pad_mask_extra], dim=1)
        else:
            a = self.token_emb(x_t)
            L_eff = L
            non_pad_mask_eff = non_pad_mask
            pad_needed = 0

        # Estados latentes z e y
        if prev_states is not None:
            prev_z, prev_y = prev_states
            g = torch.sigmoid(self.resident_core.carry_gate)
            z_r = (1.0 - g) * a + g * prev_z.detach()
            y_r = g * prev_y.detach()
        else:
            z_r = a.clone()
            y_r = torch.zeros_like(a)

        anchor_feat = self.resident_core.anchor_proj(
            a, record_telemetry=record_telemetry, integer_mode=int_mode, add_sub_mode=add_sub
        )

        cum_halt = torch.zeros(B, L_eff, 1, device=x_t.device)
        active_mask = torch.ones(B, L_eff, 1, device=x_t.device)

        intermediate_logits: List[torch.Tensor] = []
        halt_probs_list: List[torch.Tensor] = []
        new_latent_kvs: List[torch.Tensor] = []
        deltas: List[float] = []

        actual_recursions = 0
        adaptive_tol = getattr(self.cfg, "adaptive_recursion_tol", 0.05)

        for r in range(R):
            actual_recursions += 1
            cache_r = (
                latent_kv_caches_per_r[r]
                if (latent_kv_caches_per_r is not None and r < len(latent_kv_caches_per_r))
                else None
            )

            z_next, y_next, h_prob, updated_kv, _ = self.resident_core(
                z_r,
                y_r,
                anchor_feat,
                t_noise,
                delta_t,
                r,
                R,
                non_pad_mask_eff,
                active_mask,
                latent_kv_cache=cache_r,
                start_pos=start_pos,
                record_telemetry=(record_telemetry and r == R - 1),
                integer_mode=int_mode,
                add_sub_mode=add_sub,
            )

            new_latent_kvs.append(updated_kv)
            halt_probs_list.append(h_prob)

            delta = 1.0
            with torch.no_grad():
                ref_norm = y_r.norm() if r > 0 else a.norm()
                delta = float(((y_next - y_r).norm() / (ref_norm + 1e-4)).item())
                deltas.append(delta)

            z_r, y_r = z_next, y_next
            cum_halt = (cum_halt + h_prob * active_mask).clamp(max=1.0)

            # Proyección hacia logits
            if self.training or r == R - 1:
                normed_y = self.final_norm(y_r)
                if self.use_byte_patching:
                    logits_r = self.byte_patcher.decode_patches(normed_y, original_len=L)
                else:
                    logits_r = self.lm_head(normed_y)
                intermediate_logits.append(logits_r)

            if self.training:
                active_mask = (1.0 - cum_halt).clamp(min=0.0)
            else:
                active_mask = (cum_halt < self.cfg.halt_threshold).float()
                # Cómputo Adaptativo: Salida Temprana (Early Exit)
                if getattr(self.cfg, "adaptive_recursion", True) and r > 0 and delta < adaptive_tol:
                    while len(new_latent_kvs) < R:
                        new_latent_kvs.append(new_latent_kvs[-1])
                    break
                if active_mask.max() < 0.01:
                    while len(new_latent_kvs) < R:
                        new_latent_kvs.append(new_latent_kvs[-1])
                    break

        self.last_recursions_executed = actual_recursions
        self.last_deltas = deltas

        if not intermediate_logits:
            normed_y = self.final_norm(y_r)
            if self.use_byte_patching:
                logits_r = self.byte_patcher.decode_patches(normed_y, original_len=L)
            else:
                logits_r = self.lm_head(normed_y)
            intermediate_logits.append(logits_r)

        final_logits = intermediate_logits[-1]
        carried_states = (z_r.detach(), y_r.detach())

        return ModelOutput(
            final_logits=final_logits,
            intermediate_logits=intermediate_logits,
            halt_probs=halt_probs_list,
            kv_caches=new_latent_kvs,
            carried_states=carried_states,
            last_hidden_state=y_r,
        )

    def configure_adamw_groups(self) -> List[Dict]:
        """Configuración de decaimiento desacoplado de pesos (no-decay en normas, escalas y BitLinear)."""
        bitlinear_param_ids = {
            id(p) for m in self.modules() if isinstance(m, BitLinear) for p in m.parameters()
        }
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if (
                param.ndim < 2
                or id(param) in bitlinear_param_ids
                or "emb" in name
                or "norm" in name
                or "alpha" in name
                or "beta" in name
                or "gate" in name
                or "patcher" in name
            ):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "weight_decay": self.cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def collect_ternary_telemetry(self) -> Dict[str, float]:
        stats_list = [m.last_stats for m in self.modules() if isinstance(m, BitLinear) and m.last_stats]
        if not stats_list:
            return {}
        avg_scale = sum(s.get("scale_w", 0.0) for s in stats_list) / len(stats_list)
        avg_sparsity = sum(s.get("zero_frac", s.get("weight_sparsity", 0.0)) for s in stats_list) / len(stats_list)
        avg_bpw = sum(s.get("bitcos_bpw", 1.58) for s in stats_list) / len(stats_list)
        moe_active = 1.0
        if hasattr(self.resident_core, "paged_moe"):
            moe_stats = self.resident_core.paged_moe.get_last_telemetry()
            moe_active = moe_stats.get("active_slot_ratio", 1.0)
        return {
            "scale_w": avg_scale,
            "weight_sparsity": avg_sparsity,
            "bitcos_bpw": avg_bpw,
            "moe_active_slot_ratio": moe_active,
        }

    def export_paged_bitcos(self, path: str, page_size: int = 4096, format_type: str = "compact") -> Dict[str, Any]:
        """Exporta el modelo a formato Paged BitCOS v1 alineado a páginas."""
        return PagedBitCOSSerializer.export_paged_model(self, path, page_size, format_type)

    def get_full_telemetry(self) -> Dict[str, Any]:
        """Recopila telemetría de movimiento de pesos, memoria contextual y cómputo adaptativo."""
        wm_stats = self.weight_manager.get_telemetry()
        cm_stats = self.context_manager.get_telemetry()
        return {
            "weight_movement": wm_stats,
            "hierarchical_context": cm_stats,
            "adaptive_recursions_last": self.last_recursions_executed,
            "latent_deltas": self.last_deltas,
        }
