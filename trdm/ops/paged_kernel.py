import struct
import time
from typing import Dict, Tuple
import numpy as np
import torch


def unpack_page_to_dense(page_bytes: bytes) -> Tuple[int, int, int, float, torch.Tensor]:
    """
    Decodifica una página binaria Paged BitCOS a un tensor denso int8 {-1, 0, +1}.
    Retorna: (row_start, rows, cols, scale_w, codes_tensor)
    """
    header_fmt = "<IIHHfII"
    header_size = struct.calcsize(header_fmt)
    row_start, rows, cols, fmt_code, scale_w, len_p1, len_p2 = struct.unpack(
        header_fmt, page_bytes[:header_size]
    )
    offset = header_size

    total_weights = rows * cols
    if fmt_code == 0:  # compact: presencia + signos
        pres_bytes = page_bytes[offset : offset + len_p1]
        offset += len_p1
        sign_bytes = page_bytes[offset : offset + len_p2]

        presence = np.unpackbits(np.frombuffer(pres_bytes, dtype=np.uint8))[:total_weights]
        num_nonzero = int(presence.sum())
        signs = np.unpackbits(np.frombuffer(sign_bytes, dtype=np.uint8))[:num_nonzero]

        reconstructed = np.zeros(total_weights, dtype=np.int8)
        reconstructed[presence == 1] = np.where(signs == 1, 1, -1).astype(np.int8)
    else:  # fast: 2 bits directos por peso
        packed_bytes = page_bytes[offset : offset + len_p1]
        packed_arr = np.frombuffer(packed_bytes, dtype=np.uint8)
        # Extraer los 4 valores de cada byte
        c0 = (packed_arr & 0x03).astype(np.int8) - 1
        c1 = ((packed_arr >> 2) & 0x03).astype(np.int8) - 1
        c2 = ((packed_arr >> 4) & 0x03).astype(np.int8) - 1
        c3 = ((packed_arr >> 6) & 0x03).astype(np.int8) - 1
        reconstructed = np.empty(len(packed_arr) * 4, dtype=np.int8)
        reconstructed[0::4] = c0
        reconstructed[1::4] = c1
        reconstructed[2::4] = c2
        reconstructed[3::4] = c3
        reconstructed = reconstructed[:total_weights]

    codes_tensor = torch.from_numpy(reconstructed.reshape(rows, cols))
    return row_start, rows, cols, scale_w, codes_tensor


def paged_direct_block_matmul(
    q_x_int8: torch.Tensor,
    page_bytes: bytes,
) -> Tuple[int, int, float, torch.Tensor]:
    """
    Ejecuta la multiplicación de la página empaquetada contra activaciones INT8
    sin reconstruir todo el tensor en FP32.
    Proceso:
      Activación q_x [B, in_features] (INT8)
      ↓
      Desempaquetado local inmediato por filas / bloques de página
      ↓
      Acumulador INT32
    Retorna: (row_start, rows, scale_w, acc_int32 [B, rows])
    """
    row_start, rows, cols, scale_w, codes = unpack_page_to_dense(page_bytes)
    # Matmul aditiva/entera directa sobre el bloque de página
    if q_x_int8.dim() == 1:
        q_x_int8 = q_x_int8.unsqueeze(0)

    # Multiplicación por bloque de página confinado a 'rows' filas
    # En CPU, operar en bloques pequeños cabe en caché L1/L2
    acc_int32 = torch.matmul(q_x_int8.to(torch.int32), codes.to(torch.int32).t())
    return row_start, rows, scale_w, acc_int32


def benchmark_packed_vs_dense(
    batch_size: int = 1,
    in_features: int = 192,
    out_features: int = 384,
    page_size: int = 4096,
    num_runs: int = 50,
) -> Dict[str, Any]:
    """
    Compara de forma estricta:
    1. Reconstrucción completa en FP32 + GEMM convencional
    2. Ejecución directa por bloques de páginas empaquetadas
    3. Formato Compact vs. Formato Fast
    Responde a la pregunta 1 del auditor:
    «¿El resultado coincide y cuánto cuesta ejecutarla sin expansión completa?»
    """
    from trdm.serialization.paged_bitcos import PagedBitCOSSerializer

    # Generar pesos ternarios sintéticos {-1, 0, +1}
    torch.manual_seed(42)
    raw_weights = torch.randint(-1, 2, (out_features, in_features), dtype=torch.int8)
    scale_w = 0.05
    dense_w_fp32 = raw_weights.to(torch.float32) * scale_w

    # Generar activaciones de prueba
    x_fp32 = torch.randn(batch_size, in_features)
    scale_x = float(x_fp32.abs().max().item()) + 1e-6
    q_x_int8 = torch.clamp(torch.round(x_fp32 * (127.0 / scale_x)), -127, 127).to(torch.int8)

    # Serializar páginas en ambos formatos
    pages_compact = []
    pages_fast = []
    rows_per_page = max(1, (page_size - 64) // max(1, int(in_features * 0.25) + 4))

    for r_start in range(0, out_features, rows_per_page):
        r_end = min(out_features, r_start + rows_per_page)
        chunk = raw_weights[r_start:r_end, :].numpy()

        p_c = PagedBitCOSSerializer.pack_ternary_page(
            r_start, out_features, in_features, scale_w, chunk, page_size, "compact"
        )
        p_f = PagedBitCOSSerializer.pack_ternary_page(
            r_start, out_features, in_features, scale_w, chunk, page_size, "fast"
        )
        pages_compact.append(p_c)
        pages_fast.append(p_f)

    # 1. Baseline: Reconstrucción completa en FP32 + Matmul Convencional
    t0 = time.perf_counter()
    for _ in range(num_runs):
        # Descomprimir tensor entero en RAM
        full_recon = torch.empty(out_features, in_features, dtype=torch.float32)
        for p in pages_compact:
            r_s, r_cnt, _, sc, codes = unpack_page_to_dense(p)
            full_recon[r_s : r_s + r_cnt, :] = codes.to(torch.float32) * sc
        out_fp32 = torch.matmul(x_fp32, full_recon.t())
    lat_fp32_ms = ((time.perf_counter() - t0) / num_runs) * 1000.0

    # 2. Ejecución Paginada Directa (Compact)
    t0 = time.perf_counter()
    for _ in range(num_runs):
        out_paged = torch.empty(batch_size, out_features, dtype=torch.float32)
        for p in pages_compact:
            r_s, r_cnt, sc, acc_i32 = paged_direct_block_matmul(q_x_int8, p)
            out_paged[:, r_s : r_s + r_cnt] = acc_i32.to(torch.float32) * ((scale_x / 127.0) * sc)
    lat_paged_compact_ms = ((time.perf_counter() - t0) / num_runs) * 1000.0

    # 3. Ejecución Paginada Directa (Fast)
    t0 = time.perf_counter()
    for _ in range(num_runs):
        out_paged_fast = torch.empty(batch_size, out_features, dtype=torch.float32)
        for p in pages_fast:
            r_s, r_cnt, sc, acc_i32 = paged_direct_block_matmul(q_x_int8, p)
            out_paged_fast[:, r_s : r_s + r_cnt] = acc_i32.to(torch.float32) * ((scale_x / 127.0) * sc)
    lat_paged_fast_ms = ((time.perf_counter() - t0) / num_runs) * 1000.0

    # Verificación de coincidencia numérica
    # Nota: la ruta entera tiene una leve diferencia de redondeo de cuantización INT8
    # Comprobamos coincidencia de enteros exacta entre los dos desempaquetados
    max_diff_recon = float((out_paged - out_paged_fast).abs().max().item())
    corr_fp32 = float(torch.corrcoef(torch.stack([out_fp32.flatten(), out_paged.flatten()]))[0, 1].item())

    total_bytes_compact = sum(len(p) for p in pages_compact)
    total_bytes_fast = sum(len(p) for p in pages_fast)
    dense_bytes_fp32 = out_features * in_features * 4

    return {
        "dimensions": f"{out_features}x{in_features}",
        "dense_fp32_bytes": dense_bytes_fp32,
        "paged_compact_bytes": total_bytes_compact,
        "paged_fast_bytes": total_bytes_fast,
        "compression_compact_x": round(dense_bytes_fp32 / total_bytes_compact, 2),
        "compression_fast_x": round(dense_bytes_fp32 / total_bytes_fast, 2),
        "lat_fp32_reconstruction_ms": round(lat_fp32_ms, 3),
        "lat_paged_compact_ms": round(lat_paged_compact_ms, 3),
        "lat_paged_fast_ms": round(lat_paged_fast_ms, 3),
        "speedup_fast_vs_reconstruction": round(lat_fp32_ms / max(1e-6, lat_paged_fast_ms), 2),
        "numerical_exact_agreement": max_diff_recon < 1e-4,
        "correlation_with_fp32": round(corr_fp32, 6),
    }
