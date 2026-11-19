# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from trace.layers.norm import RMSNorm


class HierarchicalBytePatcher(nn.Module):
    """
    Compresor y Descompresor Jerárquico de Bytes (MEGABYTE / BLT):
    - Mantiene vocab_size = 260 (bytes puros 0..255 + tokens especiales).
    - Agrupa P bytes consecutivos en un parche latente en R^{d_model} mediante un codificador local ligero.
    - Reduce la longitud de secuencia efectiva del tronco pesado por un factor de P
      (ej. P=4 reduce la complejidad de atención por un factor de P^2 = 16x).
    - Un decodificador local ligero reconstruye o proyecta los logits para los P bytes del parche.
    """

    def __init__(
        self,
        vocab_size: int = 260,
        d_model: int = 192,
        patch_size: int = 4,
        encoder_dim: int = 96,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim

        # Incrustación local de bytes individuales
        self.byte_embedding = nn.Embedding(vocab_size, encoder_dim)

        # Codificador local ligero: Conv1D causal o Linear agrupando P bytes -> d_model
        self.local_encoder = nn.Sequential(
            nn.Linear(patch_size * encoder_dim, d_model),
            RMSNorm(d_model, eps=eps),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Decodificador local ligero: Proyecta d_model -> P * vocab_size
        self.local_decoder = nn.Sequential(
            RMSNorm(d_model, eps=eps),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, patch_size * vocab_size),
        )

    def encode_patches(self, byte_tokens: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Agrupa bytes [B, L] en parches latentes [B, L // P, d_model].
        Si L no es múltiplo de P, añade padding en el extremo.
        """
        B, L = byte_tokens.shape
        pad_needed = (self.patch_size - (L % self.patch_size)) % self.patch_size
        if pad_needed > 0:
            pad_tensor = torch.full((B, pad_needed), 256, dtype=byte_tokens.dtype, device=byte_tokens.device)
            byte_tokens = torch.cat([byte_tokens, pad_tensor], dim=1)
            L = byte_tokens.shape[1]

        num_patches = L // self.patch_size

        # [B, L, encoder_dim]
        byte_emb = self.byte_embedding(byte_tokens)

        # [B, num_patches, patch_size * encoder_dim]
        grouped = byte_emb.view(B, num_patches, self.patch_size * self.encoder_dim)

        # [B, num_patches, d_model]
        latent_patches = self.local_encoder(grouped)
        return latent_patches, pad_needed

    def decode_patches(self, patch_latents: torch.Tensor, original_len: Optional[int] = None) -> torch.Tensor:
        """
        Descomprime parches latentes [B, num_patches, d_model]
        en logits a nivel de byte [B, L, vocab_size].
        """
        B, num_patches, D = patch_latents.shape
        raw_logits = self.local_decoder(patch_latents)  # [B, num_patches, P * vocab_size]
        byte_logits = raw_logits.view(B, num_patches * self.patch_size, self.vocab_size)

        if original_len is not None and original_len < byte_logits.shape[1]:
            byte_logits = byte_logits[:, :original_len, :]

        return byte_logits
