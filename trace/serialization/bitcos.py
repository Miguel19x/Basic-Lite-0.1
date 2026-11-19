# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

import struct
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn


def pack_bitlinear_layer(
    out_features: int,
    in_features: int,
    scale_w: float,
    q_w_int8: torch.Tensor,
) -> bytes:
    """
    Serializa una matriz ternaria {-1, 0, +1} en formato BITCOSv2 comprimido:
    - presence_bits: 1 bit por peso indicando si es distinto de cero.
    - sign_bits: 1 bit por peso no nulo indicando signo (1: positivo, 0: negativo).
    """
    flat = q_w_int8.flatten().to(torch.int8).cpu().numpy()
    presence_bits = (flat != 0).astype(np.uint8)
    nonzero_vals = flat[flat != 0]
    sign_bits = (nonzero_vals > 0).astype(np.uint8)

    packed_presence = np.packbits(presence_bits).tobytes()
    packed_signs = np.packbits(sign_bits).tobytes()

    header = struct.pack(
        "<IIfII",
        out_features,
        in_features,
        float(scale_w),
        len(packed_presence),
        len(packed_signs),
    )
    return header + packed_presence + packed_signs


def unpack_bitlinear_layer(
    blob: bytes,
    offset: int = 0,
) -> Tuple[int, int, float, torch.Tensor, int]:
    """
    Reconstruye los códigos int8 {-1, 0, +1} y la escala a partir de un flujo BITCOSv2.
    Retorna: (out_features, in_features, scale_val, codes_tensor, nuevo_offset)
    """
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
    codes_tensor = torch.from_numpy(reconstructed.reshape(out_f, in_f))
    return out_f, in_f, scale_val, codes_tensor, offset


def export_bitcos_archive(model: nn.Module, path: str) -> int:
    """Exporta todas las capas ternarias de un modelo al archivo binario BITCOSv2."""
    from trace.layers.bitlinear import BitLinear

    chunks = []
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            name_bytes = name.encode("utf-8")
            payload = module.export_bitcos_bytes()
            chunks.append(struct.pack("<H", len(name_bytes)) + name_bytes + payload)
    full_blob = b"BITCOSv2" + b"".join(chunks)
    with open(path, "wb") as f:
        f.write(full_blob)
    return len(full_blob)


def load_bitcos_archive(model: nn.Module, path: str) -> int:
    """Carga pesos ternarios desde un archivo BITCOSv2 hacia las capas BitLinear del modelo."""
    from trace.layers.bitlinear import BitLinear

    with open(path, "rb") as f:
        blob = f.read()
    assert blob[:8] == b"BITCOSv2", "Formato BITCOS inválido"
    offset = 8
    modules_dict = dict(model.named_modules())
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
