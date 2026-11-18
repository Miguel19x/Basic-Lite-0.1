# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

import re
from collections import Counter

NODOS = [("sensor", 10, 4), ("gateway", 28, 9), ("switch", 55, 3), ("router", 90, 14)]
CARGAS = [("idle", 0.4, 1.0), ("baja", 0.7, 1.2), ("media", 1.0, 1.5),
          ("alta", 1.4, 2.0), ("pico", 1.8, 2.7), ("saturada", 2.2, 3.5)]
REDUNDANCIAS = [("ninguna", 0, 0), ("simple", 8, 2), ("doble", 18, 5), ("mesh", 30, 9)]
PERFILES = [("eco", -5, 3), ("base", 0, 0), ("turbo", 15, -2), ("edge", 6, -1), ("batch", 10, 4)]

# 24 tripletas válidas (nodo, carga, perfil) -> 24 * 4 redundancias = 96 Held-Out exactas (20.0%)
HELD_OUT_TRIPLETS = {
    ("sensor", "idle", "turbo"), ("sensor", "baja", "edge"), ("sensor", "media", "batch"),
    ("sensor", "alta", "eco"), ("sensor", "pico", "base"), ("sensor", "saturada", "turbo"),
    ("gateway", "idle", "edge"), ("gateway", "baja", "batch"), ("gateway", "media", "eco"),
    ("gateway", "alta", "base"), ("gateway", "pico", "turbo"), ("gateway", "saturada", "edge"),
    ("switch", "idle", "batch"), ("switch", "baja", "eco"), ("switch", "media", "base"),
    ("switch", "alta", "turbo"), ("switch", "pico", "edge"), ("switch", "saturada", "batch"),
    ("router", "idle", "eco"), ("router", "baja", "base"), ("router", "media", "turbo"),
    ("router", "alta", "edge"), ("router", "pico", "batch"), ("router", "saturada", "eco"),
}

def audit():
    max_seq_len = 80
    pad_id, eos_id = 256, 259
    train_samples, val_samples = [], []

    for n_name, b_w, b_ms in NODOS:
        for c_name, m_w, m_ms in CARGAS:
            for r_name, a_w, a_ms in REDUNDANCIAS:
                for p_name, o_w, o_ms in PERFILES:
                    watt = max(5, int(round(b_w * m_w + a_w + o_w)))
                    ms = max(1, int(round(b_ms * m_ms + a_ms + o_ms)))
                    prompt = f"nodo: {n_name}\ncarga: {c_name}\nred: {r_name}\nperfil: {p_name}\n".encode()
                    target = f"watt: {watt}\nms: {ms}\n".encode()
                    item = (prompt, target, watt, ms)
                    if (n_name, c_name, p_name) in HELD_OUT_TRIPLETS:
                        val_samples.append(item)
                    else:
                        train_samples.append(item)

    assert len(train_samples) == 384 and len(val_samples) == 96
    total_pos, template_pos, eos_pos, pad_pos, digit_pos = 0, 0, 0, 0, 0
    digit_counter_by_offset = Counter()
    digit_hits_majority = 0

    # Contabilizar distribución de bytes en la ventana supervisada de validación
    for idx_pos in range(32):
        col_digits = [
            tgt[idx_pos] for _, tgt, _, _ in train_samples
            if idx_pos < len(tgt) and 48 <= tgt[idx_pos] <= 57
        ]
        if col_digits:
            digit_counter_by_offset[idx_pos] = Counter(col_digits).most_common(1)[0][0]

    for prompt, target, _, _ in val_samples:
        resp_len = max_seq_len - (1 + len(prompt))
        total_pos += resp_len
        for i, b in enumerate(target):
            if 48 <= b <= 57:
                digit_pos += 1
                if digit_counter_by_offset.get(i) == b:
                    digit_hits_majority += 1
            else:
                template_pos += 1
        eos_pos += 1
        pad_pos += resp_len - (len(target) + 1)

    trivial_template_ceil = (template_pos + eos_pos + pad_pos) / total_pos
    trivial_majority_ceil = (template_pos + eos_pos + pad_pos + digit_hits_majority) / total_pos

    print("=== Auditoría Estadística del Dataset (Python Puro) ===")
    print(f"Combinaciones Totales : {len(train_samples) + len(val_samples)} (Train: {len(train_samples)} [80.0%] | Held-Out: {len(val_samples)} [20.0%])")
    print(f"Posiciones supervisadas por muestra (media): {total_pos / len(val_samples):.1f} bytes")
    print(f"  - Plantilla fija ('watt: ', 'ms: ', '\\n') : {100 * template_pos / total_pos:.1f}%")
    print(f"  - Token [EOS]                             : {100 * eos_pos / total_pos:.1f}%")
    print(f"  - Tokens [PAD]                            : {100 * pad_pos / total_pos:.1f}%")
    print(f"  - Dígitos reales a calcular ('0'..'9')    : {100 * digit_pos / total_pos:.1f}%")
    print(f"Techo Trivial (acertar solo plantilla+PAD)  : {100 * trivial_template_ceil:.2f}%")
    print(f"Techo Mayoría (plantilla + dígito frecuente): {100 * trivial_majority_ceil:.2f}%")

if __name__ == "__main__":
    audit()
