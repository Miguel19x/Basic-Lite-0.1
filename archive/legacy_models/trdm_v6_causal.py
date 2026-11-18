# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

import random
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from trdm.core.types import ModelOutput
from trdm.layers.abacus import AbacusEmbedding
from trdm.layers.bitlinear import BitLinear
from trdm.layers.embeddings import SinusoidalStepEncoding
from trdm.layers.norm import RMSNorm
from trdm.models.trdm_v5 import SharedResidentTernaryCore


class TRDMv6CausalModel(nn.Module):
    """
    TRDM-v6: Arquitectura Causal de 2 Etapas con Abacus Embeddings (Fase 2).
    
    Innovaciones Arquitectónicas:
    1. Abacus Positional Embeddings (McLeish et al., NeurIPS 2024):
       Codifica el rango decimal de cada dígito (unidades=0, decenas=1, centenas=2)
       otorgando conciencia de valor posicional a los cabezales de atención.
    2. Factorización Causal en 2 Pasadas (Two-Stage Causal Field):
       - Etapa 1: Predicción de subtotales (multiplicación base Pw, Pms).
       - Anclaje Causal: Los subtotales se inyectan como hechos consumados del contexto.
       - Etapa 2: Predicción de totales finales (adición multivariable watt, ms).
    3. Mitigación de Sesgo de Exposición:
       Entrenamiento con Scheduled Sampling para robustez ante errores de la Etapa 1.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 1. Capas de Entrada
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.abacus_emb = AbacusEmbedding(cfg.d_model, max_ranks=16, delimiter_rank=15)
        self.time_emb = SinusoidalStepEncoding(cfg.d_model)

        # 2. Núcleo Ternario Recursivo Residente (Backbone TRDM-v5 con BitLinear W1.58A8)
        self.resident_core = SharedResidentTernaryCore(cfg)

        # 3. Normalización y Cabezal de Lenguaje
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.lm_head = BitLinear(cfg.d_model, cfg.vocab_size, eps=cfg.eps)

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

    def compute_input_embeddings(self, x_t: torch.Tensor) -> torch.Tensor:
        """Suma de incrustación de token más incrustación de ábaco decimal."""
        tok_feat = self.token_emb(x_t)
        abacus_feat = self.abacus_emb(x_t)
        return tok_feat + abacus_feat

    def forward_backbone(
        self,
        x_t: torch.Tensor,
        t_noise: torch.Tensor,
        num_recursions: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Ejecuta el paso hacia adelante a través del núcleo recursivo ternario.
        Retorna los logits de salida para toda la secuencia: (B, L, vocab_size).
        """
        B, L = x_t.shape
        R = num_recursions if num_recursions is not None else self.cfg.num_recursions
        delta_t = torch.full_like(t_noise, 1.0 / max(1, self.cfg.diffusion_steps_per_block))
        non_pad_mask = (x_t != self.cfg.pad_token_id)

        # Incrustación combinada con Abacus
        a = self.compute_input_embeddings(x_t)
        z_r = a.clone()
        y_r = torch.zeros_like(a)

        anchor_feat = self.resident_core.anchor_proj(
            a, record_telemetry=False, integer_mode=False, add_sub_mode=False
        )

        active_mask = torch.ones(B, L, 1, device=x_t.device)

        for r in range(R):
            z_next, y_next, _, _, _ = self.resident_core(
                z_r,
                y_r,
                anchor_feat,
                t_noise,
                delta_t,
                r,
                R,
                non_pad_mask,
                active_mask,
                latent_kv_cache=None,
                start_pos=0,
                record_telemetry=False,
                integer_mode=False,
                add_sub_mode=False,
            )
            z_r, y_r = z_next, y_next

        normed_y = self.final_norm(y_r)
        logits = self.lm_head(normed_y)
        return logits

    def forward_two_stage(
        self,
        x_0: torch.Tensor,
        prompt_lens: List[int],
        t_noise: torch.Tensor,
        scheduled_sampling_p: float = 0.85,
    ) -> Dict[str, torch.Tensor]:
        """
        Paso hacia adelante de 2 etapas para entrenamiento conjunto.
        
        Etapa 1: Enmascara subtotales y totales. Predice subtotales (pw, pms).
        Etapa 2: Ancla subtotales predichos (o ground truth con prob p). Predice totales (watt, ms).
        """
        B, L = x_0.shape
        device = x_0.device

        # Construir máscara de Etapa 1: Subtotales y Totales enmascarados
        x_stage1 = x_0.clone()
        stage1_digit_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        stage2_digit_mask = torch.zeros(B, L, dtype=torch.bool, device=device)

        pw_spans = []
        pms_spans = []
        watt_spans = []
        ms_spans = []

        for b in range(B):
            plen = prompt_lens[b]
            # Offsets exactos de data_v2 con scratchpad
            pw_range = (plen + 4, plen + 7)
            pms_range = (plen + 13, plen + 16)
            watt_range = (plen + 23, plen + 26)
            ms_range = (plen + 31, plen + 34)

            pw_spans.append(pw_range)
            pms_spans.append(pms_range)
            watt_spans.append(watt_range)
            ms_spans.append(ms_range)

            # En Etapa 1, todos los dígitos de target están enmascarados
            x_stage1[b, pw_range[0]:pw_range[1]] = self.cfg.mask_token_id
            x_stage1[b, pms_range[0]:pms_range[1]] = self.cfg.mask_token_id
            x_stage1[b, watt_range[0]:watt_range[1]] = self.cfg.mask_token_id
            x_stage1[b, ms_range[0]:ms_range[1]] = self.cfg.mask_token_id

            # Máscaras de supervisión
            stage1_digit_mask[b, pw_range[0]:pw_range[1]] = True
            stage1_digit_mask[b, pms_range[0]:pms_range[1]] = True

            stage2_digit_mask[b, watt_range[0]:watt_range[1]] = True
            stage2_digit_mask[b, ms_range[0]:ms_range[1]] = True

        # --- EJECUCIÓN ETAPA 1 (Multiplicaciones Subtotales) ---
        logits_stage1 = self.forward_backbone(x_stage1, t_noise)

        # Pérdida de Etapa 1
        loss_stage1 = F.cross_entropy(
            logits_stage1[stage1_digit_mask],
            x_0[stage1_digit_mask],
        )

        # --- ANCLAJE CAUSAL PARA ETAPA 2 ---
        x_stage2 = x_stage1.clone()
        with torch.no_grad():
            preds_stage1 = logits_stage1.argmax(dim=-1)

        for b in range(B):
            pw_r = pw_spans[b]
            pms_r = pms_spans[b]

            # Scheduled Sampling: Usar Ground Truth con prob p, o predicción propia con prob (1-p)
            use_gt = (random.random() < scheduled_sampling_p) if self.training else False
            if use_gt:
                x_stage2[b, pw_r[0]:pw_r[1]] = x_0[b, pw_r[0]:pw_r[1]]
                x_stage2[b, pms_r[0]:pms_r[1]] = x_0[b, pms_r[0]:pms_r[1]]
            else:
                x_stage2[b, pw_r[0]:pw_r[1]] = preds_stage1[b, pw_r[0]:pw_r[1]]
                x_stage2[b, pms_r[0]:pms_r[1]] = preds_stage1[b, pms_r[0]:pms_r[1]]

        # Desacoplamiento de gradientes para evitar retropropagación destructiva a través del anclaje
        x_stage2 = x_stage2.detach()

        # --- EJECUCIÓN ETAPA 2 (Sumas Finales Condicionadas) ---
        logits_stage2 = self.forward_backbone(x_stage2, t_noise)

        # Pérdida de Etapa 2
        loss_stage2 = F.cross_entropy(
            logits_stage2[stage2_digit_mask],
            x_0[stage2_digit_mask],
        )

        total_loss = loss_stage1 + loss_stage2

        return {
            "loss": total_loss,
            "loss_stage1": loss_stage1,
            "loss_stage2": loss_stage2,
            "logits_stage1": logits_stage1,
            "logits_stage2": logits_stage2,
            "stage1_mask": stage1_digit_mask,
            "stage2_mask": stage2_digit_mask,
        }

    @torch.no_grad()
    def generate_causal_two_step(
        self,
        prompt_bytes: bytes,
        override_scratchpad_tokens: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Inferencia Causal Determinista en 2 Pasadas (Zero Exposure Bias):
        1. Emite subtotales Pw, Pms (o utiliza override_scratchpad_tokens para auditoría causal).
        2. Fija Pw, Pms en la secuencia de entrada como hechos consumados del contexto.
        3. Emite totales finales watt, ms condicionados a dichos subtotales.
        """
        self.eval()
        device = next(self.parameters()).device

        # Construir plantilla de ancho fijo con scratchpad
        # pw: ???\npms: ???\nwatt: ???\nms: ???\n
        template_target = b"pw: ???\npms: ???\nwatt: ???\nms: ???\n"
        full_bytes = [self.cfg.bos_token_id] + list(prompt_bytes) + list(template_target) + [self.cfg.eos_token_id]
        prompt_len = 1 + len(prompt_bytes)

        # Padding a max_seq_len
        pad_len = max(0, self.cfg.max_seq_len - len(full_bytes))
        full_seq = full_bytes[:self.cfg.max_seq_len] + [self.cfg.pad_token_id] * pad_len
        x = torch.tensor([full_seq], dtype=torch.long, device=device)

        # Posiciones de los campos
        pw_start, pw_end = prompt_len + 4, prompt_len + 7
        pms_start, pms_end = prompt_len + 13, prompt_len + 16
        watt_start, watt_end = prompt_len + 23, prompt_len + 26
        ms_start, ms_end = prompt_len + 31, prompt_len + 34

        # Enmascarar todos los dígitos en Etapa 1
        x[0, pw_start:pw_end] = self.cfg.mask_token_id
        x[0, pms_start:pms_end] = self.cfg.mask_token_id
        x[0, watt_start:watt_end] = self.cfg.mask_token_id
        x[0, ms_start:ms_end] = self.cfg.mask_token_id

        t_noise = torch.ones(1, device=device)  # Inferencia directa 1D (B,)

        # --- PASADA 1: Predecir Subtotales ---
        logits_1 = self.forward_backbone(x, t_noise)
        pred_pw_tokens = logits_1[0, pw_start:pw_end].argmax(dim=-1)
        pred_pms_tokens = logits_1[0, pms_start:pms_end].argmax(dim=-1)

        # --- ANCLAJE CAUSAL (Inyección en contexto para Etapa 2) ---
        if override_scratchpad_tokens is not None:
            actual_pw_tokens, actual_pms_tokens = override_scratchpad_tokens
            x[0, pw_start:pw_end] = actual_pw_tokens
            x[0, pms_start:pms_end] = actual_pms_tokens
        else:
            actual_pw_tokens, actual_pms_tokens = pred_pw_tokens, pred_pms_tokens
            x[0, pw_start:pw_end] = pred_pw_tokens
            x[0, pms_start:pms_end] = pred_pms_tokens

        # --- PASADA 2: Predecir Totales Condicionados ---
        logits_2 = self.forward_backbone(x, t_noise)
        pred_watt_tokens = logits_2[0, watt_start:watt_end].argmax(dim=-1)
        pred_ms_tokens = logits_2[0, ms_start:ms_end].argmax(dim=-1)

        # Decodificación final de caracteres ASCII
        pw_str = "".join(chr(int(t)) for t in actual_pw_tokens)
        pms_str = "".join(chr(int(t)) for t in actual_pms_tokens)
        pred_pw_raw = "".join(chr(int(t)) for t in pred_pw_tokens)
        pred_pms_raw = "".join(chr(int(t)) for t in pred_pms_tokens)
        watt_str = "".join(chr(int(t)) for t in pred_watt_tokens)
        ms_str = "".join(chr(int(t)) for t in pred_ms_tokens)

        full_output_str = f"pw: {pw_str}\npms: {pms_str}\nwatt: {watt_str}\nms: {ms_str}\n"

        meta = {
            "pw": pw_str,
            "pms": pms_str,
            "pred_pw_raw": pred_pw_raw,
            "pred_pms_raw": pred_pms_raw,
            "watt": watt_str,
            "ms": ms_str,
            "pw_int": int(pw_str) if pw_str.isdigit() else None,
            "pms_int": int(pms_str) if pms_str.isdigit() else None,
            "pred_pw_raw_int": int(pred_pw_raw) if pred_pw_raw.isdigit() else None,
            "pred_pms_raw_int": int(pred_pms_raw) if pred_pms_raw.isdigit() else None,
            "watt_int": int(watt_str) if watt_str.isdigit() else None,
            "ms_int": int(ms_str) if ms_str.isdigit() else None,
        }

        return full_output_str, meta
