from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from trdm.core.types import ModelOutput
from trdm.layers.norm import RMSNorm
from trdm.layers.bitlinear import BitLinear
from trdm.modules.recursive_core import SharedDualStateRecursiveCore
from trdm.serialization.bitcos import export_bitcos_archive, load_bitcos_archive


class TernaryRecursiveDiffusionLM(nn.Module):
    """
    Arquitectura TRDM-v4 (Ternary Recursive Discrete Diffusion Language Model):
    - Cuantización W1.58A8 con capas BitLinear y soporte nativo de aceleración C.
    - Difusión discreta causal por bloques (CDLM) con caché KV inmutable.
    - Núcleo recurrente con codificación continua sinusoidal y estados duales (z, y).
    - Enrutamiento Expert-Choice MoE intra-bloque con Grafo Disperso de Módulos.
    - Parada adaptativa por token (MoR) y proyección perezosa de logits (Lazy Logits).
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.shared_core = SharedDualStateRecursiveCore(cfg)
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.last_recursion_deltas: List[float] = []
        self.last_active_token_ratios: List[float] = []

        # Compilación opcional de expertos ternarios
        if getattr(self.cfg, "use_torch_compile", False) and hasattr(torch, "compile"):
            try:
                for idx in range(len(self.shared_core.moe.experts)):
                    self.shared_core.moe.experts[idx] = torch.compile(self.shared_core.moe.experts[idx])
            except Exception:
                pass

    def _apply_logit_capping(self, logits: torch.Tensor) -> torch.Tensor:
        """Acotación hiperbólica de logits (cap 30.0) estilo Gemma 2 para estabilizar difusión."""
        if getattr(self.cfg, "use_logit_softcapping", True):
            cap = getattr(self.cfg, "logit_cap_value", 30.0)
            return cap * torch.tanh(logits / cap)
        return logits

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
        supervised_mask: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        B, L = x_t.shape
        R = num_recursions if num_recursions is not None else self.cfg.num_recursions
        int_mode = (not self.training) and self.cfg.use_integer_inference
        add_sub = (not self.training) and self.cfg.use_add_sub_inference
        lazy_vocab = getattr(self.cfg, "lazy_vocab_projection", True)

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
        prev_node_ids: Optional[torch.Tensor] = None

        intermediate_logits: List[torch.Tensor] = []
        halt_probs_list: List[torch.Tensor] = []
        new_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        deltas: List[float] = []
        active_ratios: List[float] = []

        for r in range(R):
            cache_r = kv_caches_per_r[r] if (kv_caches_per_r is not None and r < len(kv_caches_per_r)) else None
            z_next, y_next, h_prob, updated_kv, prev_node_ids = self.shared_core(
                z_r, y_r, anchor_feat, t_noise, delta_t, r, R, non_pad_mask, active_mask,
                prev_node_ids=prev_node_ids,
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

            # Salida al vocabulario perezosa (Lazy Logits):
            # 1) Si hay máscara supervisada, proyectar SOLO los estados supervisados [N_sup, V].
            # 2) En inferencia libre, proyectar solo en r == R - 1 si lazy_vocab está activo.
            if supervised_mask is not None and lazy_vocab:
                need_inter = self.training and (
                    r == R - 1 or self.cfg.intermediate_loss_weight > 0 or self.cfg.consistency_loss_weight > 0
                )
                if need_inter or r == R - 1:
                    y_sup = self.final_norm(y_r)[supervised_mask]
                    logits_r = self._apply_logit_capping(self.lm_head(y_sup))
                    intermediate_logits.append(logits_r)
            elif (not self.training) and lazy_vocab and r < R - 1:
                pass
            else:
                logits_r = self._apply_logit_capping(self.lm_head(self.final_norm(y_r)))
                intermediate_logits.append(logits_r)

            if self.training:
                active_mask = (1.0 - cum_halt).clamp(min=0.0)
            else:
                active_mask = (cum_halt < self.cfg.halt_threshold).float()
                if active_mask.max() < 0.01:
                    # Early exit: todos los tokens alcanzaron umbral de parada
                    while len(new_kv_caches) < R:
                        new_kv_caches.append(new_kv_caches[-1])
                    break

        # Si en inferencia solo proyectamos en la última recursión pero se rompió antes
        if not intermediate_logits:
            logits_r = self._apply_logit_capping(self.lm_head(self.final_norm(y_r)))
            intermediate_logits.append(logits_r)

        if (not self.training) and lazy_vocab and supervised_mask is None and len(intermediate_logits) == 1 and R > 1:
            intermediate_logits = intermediate_logits * R

        if record_telemetry:
            self.last_recursion_deltas = deltas
            self.last_active_token_ratios = active_ratios

        final_logits = intermediate_logits[-1]
        carried_states = (z_r.detach(), y_r.detach())

        return ModelOutput(
            final_logits=final_logits,
            intermediate_logits=intermediate_logits,
            halt_probs=halt_probs_list,
            kv_caches=new_kv_caches,
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
        res = {}
        if stats_list:
            keys = stats_list[0].keys()
            res = {k: sum(d[k] for d in stats_list) / len(stats_list) for k in keys}
        moe_mod = self.shared_core.moe
        if moe_mod.last_possible_slots > 0:
            exec_slots = sum(moe_mod.last_expert_token_counts)
            res["moe_active_slot_ratio"] = float(exec_slots) / float(moe_mod.last_possible_slots)
        return res

    def export_bitcos_archive(self, path: str) -> int:
        return export_bitcos_archive(self, path)

    def load_bitcos_archive(self, path: str) -> int:
        return load_bitcos_archive(self, path)
