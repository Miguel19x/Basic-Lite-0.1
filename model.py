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
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight


class BitLinear(nn.Module):
    """
    Capa BitLinear W1.58A8:
    1. Entrenamiento QAT con STE sobre pesos latentes FP32.
    2. Exportación e importación binaria BITCOSv2 (2 - z bits/peso).
    3. Inferencia entera matricial y por descomposición de signos (+1, -1).
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
        self.register_buffer("prev_codes", torch.zeros(out_features, in_features, dtype=torch.int8), persistent=False)
        self.use_loaded_bitcos: bool = False
        self.last_stats: Dict[str, float] = {}

    def quantize_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_w = self.weight.abs().mean().clamp(min=self.eps)
        q_w = torch.clamp(torch.round(self.weight / scale_w), -1.0, 1.0)
        w_eff = self.weight + (scale_w * q_w - self.weight).detach()
        return w_eff, q_w.detach(), scale_w.detach()

    def forward_add_sub_only(self, x_norm: torch.Tensor, scale_x: torch.Tensor) -> torch.Tensor:
        """Suma/resta pura descomponiendo los códigos ternarios en máscaras booleanas."""
        q_max = 127.0
        q_x = torch.clamp(torch.round(x_norm * (q_max / scale_x)), -q_max, q_max)
        codes = self.int8_codes if self.use_loaded_bitcos else self.quantize_weights()[1].to(torch.int8)
        s_w = self.packed_scale if self.use_loaded_bitcos else self.quantize_weights()[2]

        pos_mask = (codes == 1).to(q_x.dtype)
        neg_mask = (codes == -1).to(q_x.dtype)
        acc = F.linear(q_x, pos_mask) - F.linear(q_x, neg_mask)
        return acc * ((scale_x / q_max) * s_w)

    def forward_kernel_add_sub(self, x_norm: torch.Tensor, scale_x: torch.Tensor) -> torch.Tensor:
        """Suma/resta pura ejecutada por kernel C con AVX2/OpenMP sin multiplicaciones."""
        from ternary_cpu_kernel import ternary_addsub_matmul
        q_max = 127.0
        q_x_i8 = torch.clamp(torch.round(x_norm * (q_max / scale_x)), -q_max, q_max).to(torch.int8)
        codes = self.int8_codes if self.use_loaded_bitcos else self.quantize_weights()[1].to(torch.int8)
        s_w = self.packed_scale if self.use_loaded_bitcos else self.quantize_weights()[2]
        acc_int32 = ternary_addsub_matmul(q_x_i8, codes)
        rescale = (scale_x / q_max) * s_w
        return acc_int32.to(x_norm.dtype) * rescale

    def forward(
        self,
        x: torch.Tensor,
        record_telemetry: bool = False,
        integer_mode: bool = False,
        add_sub_only: bool = False,
        use_kernel: bool = False,
    ) -> torch.Tensor:
        x_norm = self.sub_ln(x)
        q_max = 127.0
        scale_x = x_norm.abs().amax(dim=-1, keepdim=True).clamp(min=self.eps)

        if use_kernel and not self.training:
            return self.forward_kernel_add_sub(x_norm, scale_x)

        if add_sub_only and not self.training:
            return self.forward_add_sub_only(x_norm, scale_x)

        if (integer_mode or self.use_loaded_bitcos) and not self.training:
            q_x_int = torch.clamp(torch.round(x_norm * (q_max / scale_x)), -q_max, q_max).to(torch.int32)
            if self.use_loaded_bitcos:
                w_codes_int = self.int8_codes.to(torch.int32)
                s_w = self.packed_scale
            else:
                _, q_w, s_w = self.quantize_weights()
                w_codes_int = q_w.to(torch.int32)

            acc_int32 = torch.matmul(q_x_int, w_codes_int.t())
            rescale = (scale_x / q_max) * s_w
            return acc_int32.to(x.dtype) * rescale

        # QAT con STE para entrenamiento
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
        self.use_loaded_bitcos = True
        return offset


class RotaryEmbedding(nn.Module):
    """Posiciones rotatorias puras (RoPE) sin incrustación posicional absoluta redundante."""
    def __init__(self, dim: int, max_seq_len: int = 256):
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
    allowed = block_id.unsqueeze(0) <= block_id.unsqueeze(1)  # [L, L]
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
        self.rope = RotaryEmbedding(self.head_dim, 256)

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
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L, D = x.shape
        q = self.q_proj(x, record_telemetry, integer_mode).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x, record_telemetry, integer_mode).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x, record_telemetry, integer_mode).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

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
        out = self.o_proj(attn_out, record_telemetry, integer_mode)
        return out, (k_full, v_full)


class TernaryExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, eps: float = 1e-6):
        super().__init__()
        self.up = BitLinear(d_model, d_ff, eps=eps)
        self.down = BitLinear(d_ff, d_model, eps=eps)

    def forward(self, x: torch.Tensor, record_telemetry: bool = False, integer_mode: bool = False) -> torch.Tensor:
        h = F.relu(self.up(x, record_telemetry, integer_mode)).square()
        return self.down(h, record_telemetry, integer_mode)


class ExpertChoiceMoE(nn.Module):
    """
    Enrutamiento Expert-Choice estrictamente intra-bloque:
    1. Las decisiones de asignación top-k se computan localmente dentro de cada bloque de tamaño B_blk.
       Esto garantiza 100% la causalidad por bloques y la validez perpetua de la caché KV.
    2. Los tokens de relleno PAD se excluyen del cómputo para que jamás roben cupo a los expertos.
    3. La capacidad escala adaptativamente con el nivel de ruido t (C_cap(t)).
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
        active_mask: Optional[torch.Tensor] = None,
        is_pad_mask: Optional[torch.Tensor] = None,
        record_telemetry: bool = False,
        integer_mode: bool = False,
    ) -> torch.Tensor:
        B, L, D = x.shape
        B_blk = self.block_size

        # Si L > B_blk, dividimos en sub-bloques para enrutar estrictamente intra-bloque
        num_blocks = math.ceil(L / B_blk)
        x_blocks = []
        pad_blocks = []
        active_blocks = []

        for b_idx in range(num_blocks):
            s_idx = b_idx * B_blk
            e_idx = min((b_idx + 1) * B_blk, L)
            x_blocks.append(x[:, s_idx:e_idx, :])
            if is_pad_mask is not None:
                pad_blocks.append(is_pad_mask[:, s_idx:e_idx])
            else:
                pad_blocks.append(None)
            if active_mask is not None:
                active_blocks.append(active_mask[:, s_idx:e_idx, :])
            else:
                active_blocks.append(None)

        out_blocks = []
        total_counts = [0] * self.num_experts

        mean_t = float(t_noise.mean().item())
        dynamic_factor = self.capacity_factor * (1.35 - 0.70 * mean_t)

        for b_idx in range(num_blocks):
            xb = x_blocks[b_idx]
            L_b = xb.size(1)
            k_b = max(1, min(L_b, int(round((L_b * dynamic_factor) / self.num_experts))))

            router_logits = self.router(xb)
            scores = F.softmax(router_logits, dim=-1)

            # Excluir tokens PAD para que no consuman cupos de los expertos
            if pad_blocks[b_idx] is not None:
                non_pad_weight = (~pad_blocks[b_idx]).float().unsqueeze(-1)
                scores = scores * non_pad_weight

            # Modular por tokens aún activos en la recursión
            if active_blocks[b_idx] is not None:
                scores = scores * (0.05 + 0.95 * active_blocks[b_idx])

            scores = scores.transpose(1, 2)  # [B, E, L_b]
            top_gating, top_indices = torch.topk(scores, k=k_b, dim=-1)

            out_b = torch.zeros_like(xb)
            for e_idx, expert in enumerate(self.experts):
                idx_e = top_indices[:, e_idx, :]
                gate_e = top_gating[:, e_idx, :].unsqueeze(-1)
                gather_idx = idx_e.unsqueeze(-1).expand(-1, -1, D)
                selected = torch.gather(xb, dim=1, index=gather_idx)

                expert_out = expert(selected, record_telemetry, integer_mode) * gate_e
                out_b.scatter_add_(dim=1, index=gather_idx, src=expert_out)
                total_counts[e_idx] += int(B * k_b)

            out_blocks.append(out_b)

        if record_telemetry:
            self.last_expert_token_counts = total_counts

        return torch.cat(out_blocks, dim=1)


class SharedDualStateRecursiveCore(nn.Module):
    """
    Núcleo recurrente TRM/TRLM:
    - z_r: razonamiento latente profundo -> z_{r+1} = f_theta(a, y_r, z_r)
    - y_r: propuesta de respuesta tentativa -> y_{r+1} = g_theta(y_r, z_{r+1})
    - carry_gate: compuerta aprendible para transferir memoria entre rondas externas
    - noise_mlp: condicionamiento conjunto por (t, delta_t)
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

        self.recursion_emb = nn.Embedding(cfg.num_recursions + 8, cfg.d_model)
        self.noise_mlp = nn.Sequential(
            nn.Linear(2, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.carry_gate = nn.Parameter(torch.tensor(0.5))
        self.alpha_z = nn.Parameter(torch.full((cfg.num_recursions + 8,), 0.5))
        self.beta_y = nn.Parameter(torch.full((cfg.num_recursions + 8,), 0.5))

    def precompute_anchor(
        self, a: torch.Tensor, record_telemetry: bool = False, integer_mode: bool = False
    ) -> torch.Tensor:
        return self.anchor_proj(a, record_telemetry, integer_mode)

    def forward(
        self,
        z_r: torch.Tensor,
        y_r: torch.Tensor,
        anchor_feat: torch.Tensor,
        t_noise: torch.Tensor,
        delta_t: torch.Tensor,
        r_step: int,
        active_mask: torch.Tensor,
        is_pad_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
        record_telemetry: bool = False,
        integer_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        t_in = t_noise.view(-1, 1)
        dt_in = delta_t.view(-1, 1)
        if t_in.size(0) != z_r.size(0):
            t_in = t_in.expand(z_r.size(0), -1)
        if dt_in.size(0) != z_r.size(0):
            dt_in = dt_in.expand(z_r.size(0), -1)
        t_dt_feat = torch.cat([t_in, dt_in], dim=-1)
        e_t = self.noise_mlp(t_dt_feat).unsqueeze(1)

        r_idx = torch.tensor(r_step, device=z_r.device)
        e_r = self.recursion_emb(r_idx).view(1, 1, -1)

        # 1. Razonamiento latente z_{r+1}
        joint_state = self.norm_z(z_r) + self.norm_y(y_r) + anchor_feat + e_t + e_r
        attn_out, new_kv_cache = self.attn(
            joint_state,
            kv_cache=kv_cache,
            start_pos=start_pos,
            record_telemetry=record_telemetry,
            integer_mode=integer_mode,
        )
        z_mid = z_r + active_mask * self.alpha_z[r_step] * attn_out
        moe_out = self.moe(
            self.norm_moe(z_mid),
            t_noise=t_noise,
            active_mask=active_mask,
            is_pad_mask=is_pad_mask,
            record_telemetry=record_telemetry,
            integer_mode=integer_mode,
        )
        z_next = z_mid + active_mask * self.alpha_z[r_step] * moe_out

        # 2. Propuesta de respuesta y_{r+1}
        fusion_yz = self.norm_refine(y_r + z_next)
        delta_y = self.answer_updater(fusion_yz, record_telemetry, integer_mode)
        y_next = y_r + active_mask * self.beta_y[r_step] * F.silu(delta_y)

        # 3. Parada por token
        halt_prob = torch.sigmoid(self.halt_gate(self.norm_refine(z_next + y_next)))
        return z_next, y_next, halt_prob, new_kv_cache


class TernaryRecursiveDiffusionLM(nn.Module):
    """
    Arquitectura TRDM-v3 Saneada:
    - Pure RoPE (sin incrustación posicional absoluta redundante).
    - MoE intra-bloque estricto (causalidad perfecta y caché KV invariante).
    - Transporte latente (z_R, y_R) desconectado vía carry_gate.
    - Condicionamiento conjunto (t, delta_t) y parada adaptativa MoR.
    """
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
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]:
        B, L = x_t.shape
        R = num_recursions if num_recursions is not None else self.cfg.num_recursions
        integer_mode = (not self.training) and self.cfg.use_integer_inference

        if delta_t is None:
            delta_t = torch.full_like(t_noise, 1.0 / max(1, self.cfg.diffusion_steps_per_block))

        # Posiciones puras con RoPE (eliminada la incrustación absoluta redundante)
        a = self.token_emb(x_t)

        # Identificar tokens PAD para excluirlos de los cupos de expertos MoE
        is_pad_mask = (x_t == self.cfg.pad_token_id)

        # Transporte latente entre rondas de difusión (TRM / TRLM)
        if prev_states is not None:
            prev_z, prev_y = prev_states
            g = torch.sigmoid(self.shared_core.carry_gate)
            z_r = (1.0 - g) * a + g * prev_z.detach()
            y_r = g * prev_y.detach()
        else:
            z_r = a.clone()
            y_r = torch.zeros_like(a)

        anchor_feat = self.shared_core.precompute_anchor(
            a, record_telemetry=record_telemetry, integer_mode=integer_mode
        )

        cum_halt = torch.zeros(B, L, 1, device=x_t.device)
        active_mask = torch.ones(B, L, 1, device=x_t.device)

        intermediate_logits: List[torch.Tensor] = []
        halt_probs_list: List[torch.Tensor] = []
        new_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        deltas: List[float] = []
        active_ratios: List[float] = []

        for r in range(R):
            if kv_caches_per_r is not None and len(kv_caches_per_r) > 0:
                cache_r = kv_caches_per_r[r] if r < len(kv_caches_per_r) else kv_caches_per_r[-1]
            else:
                cache_r = None
            use_nograd = self.training and (r < self.cfg.nograd_recursions)

            if use_nograd:
                with torch.no_grad():
                    z_next, y_next, h_prob, updated_kv = self.shared_core(
                        z_r, y_r, anchor_feat, t_noise, delta_t, r, active_mask,
                        is_pad_mask=is_pad_mask, kv_cache=cache_r, start_pos=start_pos,
                        record_telemetry=False, integer_mode=integer_mode
                    )
                z_next, y_next, h_prob = z_next.detach(), y_next.detach(), h_prob.detach()
            else:
                z_next, y_next, h_prob, updated_kv = self.shared_core(
                    z_r, y_r, anchor_feat, t_noise, delta_t, r, active_mask,
                    is_pad_mask=is_pad_mask,
                    kv_cache=cache_r,
                    start_pos=start_pos,
                    record_telemetry=(record_telemetry and r == R - 1),
                    integer_mode=integer_mode,
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
            if self.training:
                active_mask = 1.0 - cum_halt * 0.7
            else:
                active_mask = (cum_halt < self.cfg.halt_threshold).float()
                if active_mask.sum() == 0 and not record_telemetry:
                    logits_r = self.lm_head(self.final_norm(y_r))
                    intermediate_logits.append(logits_r)
                    break

            if self.training or record_telemetry or r == R - 1:
                logits_r = self.lm_head(self.final_norm(y_r))
                intermediate_logits.append(logits_r)

        if record_telemetry:
            self.last_recursion_deltas = deltas
            self.last_active_token_ratios = active_ratios

        if len(new_kv_caches) < R and len(new_kv_caches) > 0:
            while len(new_kv_caches) < R:
                new_kv_caches.append(new_kv_caches[-1])

        final_logits = intermediate_logits[-1]
        carried_states = (z_r.detach(), y_r.detach())
        return final_logits, intermediate_logits, halt_probs_list, new_kv_caches, carried_states

    def configure_adamw_groups(self) -> List[Dict]:
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "emb" in name or "norm" in name or "alpha" in name or "beta" in name or "halt" in name or "carry" in name:
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
        assert blob[:8] == b"BITCOSv2", "Formato de archivo BITCOS inválido"
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
        self.cfg.use_integer_inference = True
        return loaded_layers
