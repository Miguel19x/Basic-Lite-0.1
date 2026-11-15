import math
import struct
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import HybridConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SinusoidalStepEncoding(nn.Module):
    """
    Codificación sinusoidal continua y sin parámetros huérfanos para el progreso
    recursivo normalizado u_r = (r + 1) / R en (0, 1].
    Permite extrapolar a cualquier R_infer >= 1 usando únicamente pesos 100% entrenados.
    """
    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(torch.arange(half, dtype=torch.float32) * (-math.log(100.0) / max(1, half - 1)))
        self.register_buffer("freqs", freqs, persistent=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, u_norm: float, device: torch.device) -> torch.Tensor:
        u = torch.tensor([u_norm * math.pi], dtype=torch.float32, device=device)
        angles = u.unsqueeze(-1) * self.freqs.unsqueeze(0)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return self.proj(emb).view(1, 1, -1)


class BitLinear(nn.Module):
    """
    Capa BitLinear W1.58A8 con caché FP32 en eval() para benchmarks justos,
    descomposición de suma/resta pura y exportación BITCOSv2.
    """
    def __init__(self, in_features: int, out_features: int, eps: float = 1e-6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.sub_ln = RMSNorm(in_features, eps=eps)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.register_buffer("int8_codes", torch.zeros(out_features, in_features, dtype=torch.int8), persistent=False)
        self.register_buffer("packed_scale", torch.tensor(1.0, dtype=torch.float32), persistent=False)
        self.register_buffer("cached_w_eff", torch.empty(0), persistent=False)
        self.register_buffer("cached_pos_mask", torch.empty(0), persistent=False)
        self.register_buffer("cached_neg_mask", torch.empty(0), persistent=False)
        self.register_buffer("prev_codes", torch.zeros(out_features, in_features, dtype=torch.int8), persistent=False)
        self.use_loaded_bitcos: bool = False
        self.last_stats: Dict[str, float] = {}

    def quantize_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_w = self.weight.abs().mean().clamp(min=self.eps)
        q_w = torch.clamp(torch.round(self.weight / scale_w), -1.0, 1.0)
        w_eff = self.weight + (scale_w * q_w - self.weight).detach()
        return w_eff, q_w.detach(), scale_w.detach()

    @torch.no_grad()
    def refresh_eval_cache(self) -> None:
        """Precalcula W_eff y máscaras de signo una sola vez al pasar a eval()."""
        w_eff, q_w, scale_w = self.quantize_weights()
        self.cached_w_eff = w_eff.detach().clone()
        self.int8_codes.copy_(q_w.to(torch.int8))
        self.packed_scale.fill_(float(scale_w.item()))
        self.cached_pos_mask = (self.int8_codes == 1).to(torch.float32)
        self.cached_neg_mask = (self.int8_codes == -1).to(torch.float32)

    def forward(
        self,
        x: torch.Tensor,
        record_telemetry: bool = False,
        integer_mode: bool = False,
        add_sub_mode: bool = False,
    ) -> torch.Tensor:
        x_norm = self.sub_ln(x)
        q_max = 127.0
        scale_x = x_norm.abs().amax(dim=-1, keepdim=True).clamp(min=self.eps)

        if not self.training:
            q_x = torch.clamp(torch.round(x_norm * (q_max / scale_x)), -q_max, q_max)
            if add_sub_mode:
                if self.cached_pos_mask.numel() == 0:
                    self.refresh_eval_cache()
                acc = F.linear(q_x, self.cached_pos_mask) - F.linear(q_x, self.cached_neg_mask)
                return acc * ((scale_x / q_max) * self.packed_scale)

            if integer_mode or self.use_loaded_bitcos:
                q_x_int = q_x.to(torch.int32)
                w_codes_int = self.int8_codes.to(torch.int32)
                acc_int32 = torch.matmul(q_x_int, w_codes_int.t())
                return acc_int32.to(x.dtype) * ((scale_x / q_max) * self.packed_scale)

            if self.cached_w_eff.numel() == 0:
                self.refresh_eval_cache()
            x_eff = q_x * (scale_x / q_max)
            return F.linear(x_eff, self.cached_w_eff, bias=None)

        # Ruta de entrenamiento QAT con STE (invalida caché de evaluación)
        if self.cached_w_eff.numel() > 0:
            self.cached_w_eff = torch.empty(0, device=x.device)

        q_x = torch.clamp(torch.round(x_norm * (q_max / scale_x)), -q_max, q_max)
        x_eff = x_norm + (q_x * (scale_x / q_max) - x_norm).detach()
        w_eff, q_w, scale_w = self.quantize_weights()

        if record_telemetry:
            codes_i8 = q_w.to(torch.int8)
            total = float(q_w.numel())
            zeros = float((codes_i8 == 0).sum().item()) / total
            pos = float((codes_i8 == 1).sum().item()) / total
            neg = float((codes_i8 == -1).sum().item()) / total
            flips = float((codes_i8 != self.prev_codes).sum().item()) / total
            sat_ratio = float((q_x.abs() == q_max).float().mean().item())
            self.prev_codes.copy_(codes_i8)
            self.last_stats = {
                "zero_frac": zeros,
                "pos_frac": pos,
                "neg_frac": neg,
                "bitcos_bpw": 2.0 - zeros,
                "scale_w": float(scale_w.item()),
                "act_saturation": sat_ratio,
                "code_flip_rate": flips,
            }

        return F.linear(x_eff, w_eff, bias=None)

    @torch.no_grad()
    def export_bitcos_bytes(self) -> bytes:
        _, q_w, scale_w = self.quantize_weights()
        flat = q_w.flatten().to(torch.int8).cpu().numpy()
        presence_bits = (flat != 0).astype(np.uint8)
        nonzero_vals = flat[flat != 0]
        sign_bits = (nonzero_vals > 0).astype(np.uint8)

        packed_presence = np.packbits(presence_bits).tobytes()
        packed_signs = np.packbits(sign_bits).tobytes()
        header = struct.pack(
            "<IIfII",
            self.out_features,
            self.in_features,
            float(scale_w.item()),
            len(packed_presence),
            len(packed_signs),
        )
        return header + packed_presence + packed_signs

    @torch.no_grad()
    def load_from_bitcos_bytes(self, blob: bytes, offset: int = 0) -> int:
        header_size = struct.calcsize("<IIfII")
        out_f, in_f, scale_val, len_pres, len_signs = struct.unpack(
            "<IIfII", blob[offset : offset + header_size]
        )
        offset += header_size
        pres_bytes = blob[offset : offset + len_pres]
        offset += len_pres
        sign_bytes = blob[offset : offset + len_signs]
        offset += len_signs

        total_weights = out_f * in_f
        presence = np.unpackbits(np.frombuffer(pres_bytes, dtype=np.uint8))[:total_weights]
        num_nonzero = int(presence.sum())
        signs = np.unpackbits(np.frombuffer(sign_bytes, dtype=np.uint8))[:num_nonzero]

        reconstructed = np.zeros(total_weights, dtype=np.int8)
        reconstructed[presence == 1] = np.where(signs == 1, 1, -1).astype(np.int8)

        self.int8_codes.copy_(torch.from_numpy(reconstructed.reshape(out_f, in_f)))
        self.packed_scale.fill_(scale_val)
        self.cached_w_eff = self.int8_codes.to(torch.float32) * scale_val
        self.cached_pos_mask = (self.int8_codes == 1).to(torch.float32)
        self.cached_neg_mask = (self.int8_codes == -1).to(torch.float32)
        self.use_loaded_bitcos = True
        return offset


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 128):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        seq_len = x.size(2)
        cos = self.cos[:, :, start_pos : start_pos + seq_len, :]
        sin = self.sin[:, :, start_pos : start_pos + seq_len, :]
        return (x * cos) + (self._rotate_half(x) * sin)


def build_block_causal_mask(seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device)
    block_id = idx // block_size
    allowed = block_id.unsqueeze(0) <= block_id.unsqueeze(1)
    return allowed.unsqueeze(0).unsqueeze(0)


class BlockCausalGQATernaryAttention(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.num_queries_per_kv = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.block_size = cfg.block_size

        self.q_proj = BitLinear(cfg.d_model, cfg.d_model, eps=cfg.eps)
        self.k_proj = BitLinear(cfg.d_model, self.kv_dim, eps=cfg.eps)
        self.v_proj = BitLinear(cfg.d_model, self.kv_dim, eps=cfg.eps)
        self.o_proj = BitLinear(cfg.d_model, cfg.d_model, eps=cfg.eps)
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_queries_per_kv == 1:
            return x
        B, H_kv, L, D_h = x.shape
        return (
            x[:, :, None, :, :]
            .expand(B, H_kv, self.num_queries_per_kv, L, D_h)
            .reshape(B, self.n_heads, L, D_h)
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
        record_telemetry: bool = False,
        integer_mode: bool = False,
        add_sub_mode: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L, D = x.shape
        q = self.q_proj(x, record_telemetry, integer_mode, add_sub_mode).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x, record_telemetry, integer_mode, add_sub_mode).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x, record_telemetry, integer_mode, add_sub_mode).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope.apply_rope(q, start_pos=start_pos)
        k = self.rope.apply_rope(k, start_pos=start_pos)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k_full = torch.cat([k_prev, k], dim=2)
            v_full = torch.cat([v_prev, v], dim=2)
        else:
            k_full, v_full = k, v

        k_exp = self._repeat_kv(k_full)
        v_exp = self._repeat_kv(v_full)

        if kv_cache is not None:
            attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)
        else:
            attn_mask = build_block_causal_mask(L, self.block_size, x.device)
            attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=attn_mask)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(attn_out, record_telemetry, integer_mode, add_sub_mode)
        return out, (k_full, v_full)


class TernaryExpert(nn.Module):
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
    Enrutamiento Expert-Choice estrictamente intra-bloque (sin fuga causal entre bloques)
    y con enmascaramiento de tokens PAD.
    """
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.capacity_factor = cfg.capacity_factor
        self.block_size = cfg.block_size
        self.router = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList([
            TernaryExpert(cfg.d_model, cfg.d_ff_expert, eps=cfg.eps)
            for _ in range(cfg.num_experts)
        ])
        self.last_expert_token_counts: List[int] = [0] * cfg.num_experts

    def forward(
        self,
        x: torch.Tensor,
        t_noise: torch.Tensor,
        non_pad_mask: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
        record_telemetry: bool = False,
        integer_mode: bool = False,
        add_sub_mode: bool = False,
    ) -> torch.Tensor:
        B, L, D = x.shape
        B_blk = min(self.block_size, L)
        num_blocks = L // B_blk

        mean_t = float(t_noise.mean().item())
        dynamic_factor = self.capacity_factor * (1.25 - 0.50 * mean_t)
        k = max(1, min(B_blk, int(round((B_blk * dynamic_factor) / self.num_experts))))

        x_blk = x.view(B * num_blocks, B_blk, D)
        router_logits = self.router(x_blk)
        scores = F.softmax(router_logits, dim=-1)

        if active_mask is not None:
            act_blk = active_mask.view(B * num_blocks, B_blk, 1)
            scores = scores * (0.10 + 0.90 * act_blk)

        if non_pad_mask is not None:
            valid_blk = non_pad_mask.view(B * num_blocks, B_blk, 1).float()
            scores = scores * valid_blk

        scores_t = scores.transpose(1, 2)  # [B*num_blocks, E, B_blk]
        top_gating, top_indices = torch.topk(scores_t, k=k, dim=-1)

        out_blk = torch.zeros_like(x_blk)
        counts = [0] * self.num_experts

        for e_idx, expert in enumerate(self.experts):
            idx_e = top_indices[:, e_idx, :]
            gate_e = top_gating[:, e_idx, :].unsqueeze(-1)
            gather_idx = idx_e.unsqueeze(-1).expand(-1, -1, D)
            selected = torch.gather(x_blk, dim=1, index=gather_idx)
            expert_out = expert(selected, record_telemetry, integer_mode, add_sub_mode) * gate_e
            out_blk.scatter_add_(dim=1, index=gather_idx, src=expert_out)
            counts[e_idx] = int(B * num_blocks * k)

        if record_telemetry:
            self.last_expert_token_counts = counts
        return out_blk.view(B, L, D)


class SharedDualStateRecursiveCore(nn.Module):
    """
    Núcleo recurrente TRM/TRLM con codificación sinusoidal continua de profundidad u_r = (r+1)/R
    y escalas residuales compartidas (sin parámetros dependientes de un índice r máximo fijo).
    """
    def __init__(self, cfg: HybridConfig):
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
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
        record_telemetry: bool = False,
        integer_mode: bool = False,
        add_sub_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        noise_pair = torch.stack([t_noise, delta_t], dim=-1)
        e_t = self.noise_mlp(noise_pair).unsqueeze(1)

        # Progreso normalizado en (0, 1] y amortiguación de paso 2 / (R + 1) para estabilidad en R > R_train
        u_norm = float(r_step + 1) / float(max(1, total_recursions))
        step_scale = 2.0 / float(max(2, total_recursions))
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
        moe_out = self.moe(
            self.norm_moe(z_mid),
            t_noise=t_noise,
            non_pad_mask=non_pad_mask,
            active_mask=active_mask,
            record_telemetry=record_telemetry,
            integer_mode=integer_mode,
            add_sub_mode=add_sub_mode,
        )
        z_next = z_mid + active_mask * eff_alpha * moe_out

        fusion_yz = self.norm_refine(y_r + z_next)
        delta_y = self.answer_updater(fusion_yz, record_telemetry, integer_mode, add_sub_mode)
        y_next = y_r + active_mask * eff_beta * F.silu(delta_y)

        halt_prob = torch.sigmoid(self.halt_gate(self.norm_refine(z_next + y_next)))
        return z_next, y_next, halt_prob, new_kv_cache


class TernaryRecursiveDiffusionLM(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.shared_core = SharedDualStateRecursiveCore(cfg)
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.last_recursion_deltas: List[float] = []
        self.last_active_token_ratios: List[float] = []

    def prepare_for_eval(self) -> None:
        """Congela los pesos efectivos cuantizados en caché al entrar en modo evaluación."""
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
        kv_caches_per_r: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0,
        record_telemetry: bool = False,
    ):
        B, L = x_t.shape
        R = num_recursions if num_recursions is not None else self.cfg.num_recursions
        int_mode = (not self.training) and self.cfg.use_integer_inference
        add_sub = (not self.training) and self.cfg.use_add_sub_inference

        if delta_t is None:
            delta_t = torch.full_like(t_noise, 1.0 / max(1, self.cfg.diffusion_steps_per_block))

        non_pad_mask = (x_t != self.cfg.pad_token_id)
        a = self.token_emb(x_t)

        if prev_states is not None:
            prev_z, prev_y = prev_states
            g = torch.sigmoid(self.shared_core.carry_gate)
            z_r = (1.0 - g) * a + g * prev_z.detach()
            y_r = g * prev_y.detach()
        else:
            z_r = a.clone()
            y_r = torch.zeros_like(a)

        anchor_feat = self.shared_core.precompute_anchor(
            a, record_telemetry=record_telemetry, integer_mode=int_mode, add_sub_mode=add_sub
        )

        cum_halt = torch.zeros(B, L, 1, device=x_t.device)
        active_mask = torch.ones(B, L, 1, device=x_t.device)

        intermediate_logits: List[torch.Tensor] = []
        halt_probs_list: List[torch.Tensor] = []
        new_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        deltas: List[float] = []
        active_ratios: List[float] = []

        for r in range(R):
            cache_r = kv_caches_per_r[r] if (kv_caches_per_r is not None and r < len(kv_caches_per_r)) else None
            z_next, y_next, h_prob, updated_kv = self.shared_core(
                z_r, y_r, anchor_feat, t_noise, delta_t, r, R, non_pad_mask, active_mask,
                kv_cache=cache_r,
                start_pos=start_pos,
                record_telemetry=(record_telemetry and r == R - 1),
                integer_mode=int_mode,
                add_sub_mode=add_sub,
            )

            new_kv_caches.append(updated_kv)
            halt_probs_list.append(h_prob)

            if record_telemetry:
                with torch.no_grad():
                    delta = (y_next - y_r).norm() / (y_r.norm() + 1e-4)
                    deltas.append(float(delta.item()))
                    active_ratios.append(float((active_mask > 0.1).float().mean().item()))

            z_r, y_r = z_next, y_next
            cum_halt = (cum_halt + h_prob * active_mask).clamp(max=1.0)
            active_mask = 1.0 - cum_halt * 0.5 if self.training else (cum_halt < self.cfg.halt_threshold).float()

            logits_r = self.lm_head(self.final_norm(y_r))
            intermediate_logits.append(logits_r)

        if record_telemetry:
            self.last_recursion_deltas = deltas
            self.last_active_token_ratios = active_ratios

        final_logits = intermediate_logits[-1]
        carried_states = (z_r.detach(), y_r.detach())
        return final_logits, intermediate_logits, halt_probs_list, new_kv_caches, carried_states

    def configure_adamw_groups(self) -> List[Dict]:
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "emb" in name or "norm" in name or "alpha" in name or "beta" in name or "gate" in name:
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
        keys = stats_list[0].keys()
        return {k: sum(d[k] for d in stats_list) / len(stats_list) for k in keys}

    def export_bitcos_archive(self, path: str) -> int:
        chunks = []
        for name, module in self.named_modules():
            if isinstance(module, BitLinear):
                name_bytes = name.encode("utf-8")
                payload = module.export_bitcos_bytes()
                chunks.append(struct.pack("<H", len(name_bytes)) + name_bytes + payload)
        full_blob = b"BITCOSv2" + b"".join(chunks)
        with open(path, "wb") as f:
            f.write(full_blob)
        return len(full_blob)

    def load_bitcos_archive(self, path: str) -> int:
        with open(path, "rb") as f:
            blob = f.read()
        assert blob[:8] == b"BITCOSv2", "Formato BITCOS inválido"
        offset = 8
        modules_dict = dict(self.named_modules())
        loaded_layers = 0
        while offset < len(blob):
            (name_len,) = struct.unpack("<H", blob[offset : offset + 2])
            offset += 2
            layer_name = blob[offset : offset + name_len].decode("utf-8")
            offset += name_len
            mod = modules_dict[layer_name]
            assert isinstance(mod, BitLinear)
            offset = mod.load_from_bitcos_bytes(blob, offset)
            loaded_layers += 1
        return loaded_layers
