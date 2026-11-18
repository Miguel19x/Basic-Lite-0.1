# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

from typing import Optional, Tuple
import torch


def apply_blockwise_diffusion_corruption(
    x_0: torch.Tensor,
    can_mask: torch.Tensor,
    cfg,
    fixed_t: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aplica corrupción por bloques con soporte para bloques limpios (t=0.0).
    Devuelve (x_t, mask_positions, t_per_token).
    t_per_token tiene forma (B, L) y representa la tasa de ruido exacta de cada token/bloque.
    """
    B, L = x_0.shape
    device = x_0.device
    B_blk = cfg.block_size
    num_blocks = (L + B_blk - 1) // B_blk

    if fixed_t is not None:
        t_block = torch.full((B, num_blocks), fixed_t, device=device)
    else:
        t_block = torch.empty((B, num_blocks), device=device).uniform_(
            cfg.min_mask_rate, cfg.max_mask_rate
        )
        # Bloques completamente enmascarados (t=1.0) con probabilidad 0.30 para entrenar generación desde cero
        full_mask_selector = torch.rand((B, num_blocks), device=device) < 0.30
        t_block = torch.where(full_mask_selector, torch.ones_like(t_block), t_block)

        # Bloques limpios reciben t=0.0
        zero_selector = torch.rand((B, num_blocks), device=device) < 0.15
        t_block = torch.where(zero_selector, torch.zeros_like(t_block), t_block)

    t_per_token = t_block.repeat_interleave(B_blk, dim=1)[:, :L]
    # Posiciones que no se pueden enmascarar (prompt prefijo) tienen t=0.0
    t_per_token = torch.where(can_mask, t_per_token, torch.zeros_like(t_per_token))

    rand_u = torch.rand((B, L), device=device)
    mask_positions = (rand_u < t_per_token) & can_mask

    for b in range(B):
        if not mask_positions[b].any() and can_mask[b].any():
            valid_idx = torch.nonzero(can_mask[b], as_tuple=False).squeeze(-1)
            chosen = valid_idx[torch.randint(0, len(valid_idx), (1,), device=device)]
            mask_positions[b, chosen] = True
            t_per_token[b, chosen] = max(cfg.min_mask_rate, float(t_per_token[b, chosen].item()))

    x_t = x_0.clone()
    x_t[mask_positions] = cfg.mask_token_id
    return x_t, mask_positions, t_per_token
