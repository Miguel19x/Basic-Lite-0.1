#!/usr/bin/env python3
# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Prueba de Residencia e Instrumentación de E/S - Basic-Lite 0.2 (TRDM-v5)
Eje 2: Validación funcional y medición de rendimiento de expertos ternarios paginados.

Configuración auditada:
- 32 expertos sintetizados empaquetados en un archivo binario de 16 MiB (512 KiB / experto).
- Presupuesto estricto de pesos empaquetados residentes en aplicación: 2 slots (1 MiB).
- Instrumentación de E/S mediante /proc/self/io (rchar y read_bytes) con aislamiento de overhead.
- Tres regímenes explícitos:
    1. Acierto residente (en slot LRU de aplicación).
    2. Fallo de aplicación, caché del sistema caliente (atendido por Linux Page Cache).
    3. Fallo con lectura de almacenamiento observada (incremento en read_bytes).
- Desglose de latencias por etapa: lectura, preparación, mult1, mult2, total.
"""

import os
import sys
import time
import json
import struct
import statistics
from collections import OrderedDict
from typing import Dict, List, Tuple, Any

import numpy as np
import torch

from trdm.ops.paged_c_kernel import paged_lowrank_matmul_c


def get_proc_io_stats() -> Dict[str, int]:
    """Lee y parsea /proc/self/io con mínimo overhead."""
    stats = {}
    with open("/proc/self/io", "r") as f:
        for line in f:
            parts = line.strip().split(": ")
            if len(parts) == 2:
                stats[parts[0]] = int(parts[1])
    return stats


def measure_io_stats_overhead(iterations: int = 20) -> int:
    """Mide con precisión el costo en rchar de invocar get_proc_io_stats()."""
    diffs = []
    for _ in range(iterations):
        s0 = get_proc_io_stats()
        s1 = get_proc_io_stats()
        diffs.append(s1["rchar"] - s0["rchar"])
    # La mediana del overhead de la propia llamada
    return int(statistics.median(diffs))


def pack_ternary_matrix_2bit(w_ternary: torch.Tensor) -> bytes:
    """
    Empaqueta matriz ternaria {-1, 0, +1} a 2 bits por peso (4 pesos por byte).
    Mapeo: -1 -> 0b00 (0), 0 -> 0b01 (1), +1 -> 0b10 (2).
    """
    rows, cols = w_ternary.shape
    assert cols % 4 == 0, "cols debe ser divisible entre 4"
    w_shifted = (w_ternary.to(torch.int8) + 1).to(torch.uint8)
    w_np = w_shifted.numpy()
    
    bytes_list = []
    cols_div_4 = cols // 4
    for r in range(rows):
        row_bytes = bytearray(cols_div_4)
        r_vals = w_np[r]
        for k in range(0, cols, 4):
            b = (int(r_vals[k]) & 3) | ((int(r_vals[k+1]) & 3) << 2) | ((int(r_vals[k+2]) & 3) << 4) | ((int(r_vals[k+3]) & 3) << 6)
            row_bytes[k >> 2] = b
        bytes_list.append(bytes(row_bytes))
    return b"".join(bytes_list)


def create_paged_experts_file(bin_path: str, num_experts: int = 32) -> int:
    """
    Genera el archivo binario con 32 expertos de bajo rango ternarios (4096 x 256).
    Cada experto:
      A: 256 rows x 4096 cols -> 262.144 bytes (256 KiB)
      B: 4096 rows x 256 cols -> 262.144 bytes (256 KiB)
      Total empaquetado por experto = 524.288 bytes (512 KiB)
    """
    expert_size = 524288
    total_bytes = num_experts * expert_size

    print(f"[Preparación] Generando archivo de {num_experts} expertos: {bin_path} ({total_bytes / (1024*1024):.1f} MiB)...")
    with open(bin_path, "wb") as f:
        for e in range(num_experts):
            torch.manual_seed(1000 + e)
            A_raw = torch.randint(-1, 2, (256, 4096), dtype=torch.int8)
            torch.manual_seed(2000 + e)
            B_raw = torch.randint(-1, 2, (4096, 256), dtype=torch.int8)

            A_packed = pack_ternary_matrix_2bit(A_raw)
            B_packed = pack_ternary_matrix_2bit(B_raw)
            assert len(A_packed) == 262144
            assert len(B_packed) == 262144
            
            f.write(A_packed)
            f.write(B_packed)
        f.flush()
        os.fsync(f.fileno())

    file_size = os.path.getsize(bin_path)
    assert file_size == total_bytes, f"Tamaño incorrecto: {file_size} != {total_bytes}"
    print(f"[Preparación] Archivo {bin_path} creado y sincronizado exitosamente ({file_size} bytes).")
    return file_size


class ReferenceRAMEngine:
    """Motor de referencia: mantiene los 32 expertos permanentemente en RAM."""
    def __init__(self, bin_path: str, num_experts: int = 32):
        self.num_experts = num_experts
        self.expert_size = 524288
        self.experts = {}
        with open(bin_path, "rb") as f:
            for e in range(num_experts):
                raw = f.read(self.expert_size)
                self.experts[e] = raw

    def forward(self, expert_id: int, x_int8: torch.Tensor) -> torch.Tensor:
        raw = self.experts[expert_id]
        packed_A = raw[:262144]
        packed_B = raw[262144:]
        h = paged_lowrank_matmul_c(x_int8, packed_A, 256, 4096)
        h_int8 = torch.clamp(h >> 5, -128, 127).to(torch.int8)
        y = paged_lowrank_matmul_c(h_int8, packed_B, 4096, 256)
        return y


class PagedResidentEngine:
    """
    Motor de Ejecución con Residencia Paginada y Presupuesto Estricto de 2 Slots:
    - Política LRU en espacio de aplicación.
    - Lecturas explícitas mediante pread().
    - Medición de I/O por /proc/self/io discriminando rchar y read_bytes.
    - Desglose de latencias por etapa en nanosegundos.
    """
    def __init__(self, bin_path: str, max_slots: int = 2, io_overhead: int = 95):
        self.bin_path = bin_path
        self.max_slots = max_slots
        self.expert_size = 524288
        self.io_overhead = io_overhead
        self.fd = os.open(bin_path, os.O_RDONLY)

        # Slots de aplicación (LRU)
        self.slots: OrderedDict[int, bytes] = OrderedDict()

        # Telemetría
        self.total_queries = 0
        self.resident_hits = 0
        self.faults_page_cache_hot = 0
        self.faults_storage_observed = 0
        self.evictions = 0

        self.bytes_solicitados_pread = 0
        self.bytes_devueltos_pread = 0
        self.rchar_net = 0
        self.read_bytes_storage = 0

        self.call_history: List[Dict[str, Any]] = []

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def forward(self, expert_id: int, x_int8: torch.Tensor, force_storage_miss: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        self.total_queries += 1
        is_hit = expert_id in self.slots

        # La instrumentación del experimento se ejecuta estrictamente FUERA del cronómetro de la llamada
        if force_storage_miss and not is_hit:
            os.posix_fadvise(self.fd, expert_id * self.expert_size, self.expert_size, os.POSIX_FADV_DONTNEED)

        if not is_hit:
            io_0 = get_proc_io_stats()

        # === INICIO CRONÓMETRO DE LLAMADA AL MOTOR ===
        t_call_start = time.perf_counter_ns()

        t_read_us = 0.0
        bytes_devueltos_call = 0
        regime = ""

        if is_hit:
            regime = "Acierto residente"
            self.resident_hits += 1
            self.slots.move_to_end(expert_id)
            expert_data = self.slots[expert_id]
        else:
            # Fallo de aplicación: lectura explícita con pread(2)
            t_read_start = time.perf_counter_ns()
            expert_data = os.pread(self.fd, self.expert_size, expert_id * self.expert_size)
            t_read_end = time.perf_counter_ns()
            t_read_us = (t_read_end - t_read_start) / 1000.0
            bytes_devueltos_call = len(expert_data)

            # Gestión de slots LRU en memoria de aplicación
            if len(self.slots) >= self.max_slots:
                evicted_id, _ = self.slots.popitem(last=False)
                self.evictions += 1

            self.slots[expert_id] = expert_data
            # Invariante duro de memoria
            assert len(self.slots) <= self.max_slots, f"Invariante violado: {len(self.slots)} > {self.max_slots}"

        # Ejecución aritmética del experto
        # 1. Preparación de punteros
        t_prep_start = time.perf_counter_ns()
        packed_A = expert_data[:262144]
        packed_B = expert_data[262144:]
        t_prep_end = time.perf_counter_ns()
        t_prep_us = (t_prep_end - t_prep_start) / 1000.0

        # 2. Primera multiplicación x @ A (rows=256, cols=4096)
        t_m1_start = time.perf_counter_ns()
        h = paged_lowrank_matmul_c(x_int8, packed_A, 256, 4096)
        t_m1_end = time.perf_counter_ns()
        t_m1_us = (t_m1_end - t_m1_start) / 1000.0

        # Cuantización intermedia a INT8
        h_int8 = torch.clamp(h >> 5, -128, 127).to(torch.int8)

        # 3. Segunda multiplicación h @ B (rows=4096, cols=256)
        t_m2_start = time.perf_counter_ns()
        y = paged_lowrank_matmul_c(h_int8, packed_B, 4096, 256)
        t_m2_end = time.perf_counter_ns()
        t_m2_us = (t_m2_end - t_m2_start) / 1000.0

        # === FIN CRONÓMETRO DE LLAMADA AL MOTOR ===
        t_call_end = time.perf_counter_ns()
        t_total_us = (t_call_end - t_call_start) / 1000.0

        # Medición de E/S y clasificación de régimen estrictamente DESPUÉS de detener el cronómetro
        delta_rchar = 0
        delta_read_bytes = 0
        if not is_hit:
            io_1 = get_proc_io_stats()
            raw_rchar_diff = io_1["rchar"] - io_0["rchar"]
            delta_rchar = max(0, raw_rchar_diff - self.io_overhead)
            delta_read_bytes = max(0, io_1["read_bytes"] - io_0["read_bytes"])

            self.bytes_solicitados_pread += self.expert_size
            self.bytes_devueltos_pread += bytes_devueltos_call
            self.rchar_net += delta_rchar
            self.read_bytes_storage += delta_read_bytes

            if delta_read_bytes > 0:
                regime = "Fallo con lectura de almacenamiento observada"
                self.faults_storage_observed += 1
            else:
                regime = "Fallo de aplicación, caché del sistema caliente"
                self.faults_page_cache_hot += 1

        # Tiempo residual: gestión LRU, búsqueda de hash, asignación de slot y envoltorio
        t_residual_us = max(0.0, t_total_us - (t_read_us + t_prep_us + t_m1_us + t_m2_us))

        record = {
            "query_idx": self.total_queries,
            "expert_id": expert_id,
            "regime": regime,
            "t_read_us": t_read_us,
            "t_prep_us": t_prep_us,
            "t_mult1_us": t_m1_us,
            "t_mult2_us": t_m2_us,
            "t_residual_us": t_residual_us,
            "t_total_us": t_total_us,
            "bytes_solicitados": self.expert_size if not is_hit else 0,
            "bytes_devueltos": bytes_devueltos_call,
            "delta_rchar": delta_rchar,
            "delta_read_bytes": delta_read_bytes,
            "active_resident_slots": len(self.slots),
        }
        self.call_history.append(record)
        return y, record


def compute_statistics_summary(values: List[float]) -> Dict[str, float]:
    """Calcula mediana convencional (promedio de centrales para N par), IQR, min, max, media y std."""
    if not values:
        return {"n": 0, "mediana": 0.0, "iqr": 0.0, "min": 0.0, "max": 0.0, "media": 0.0, "std": 0.0}
    
    n = len(values)
    med = float(statistics.median(values))
    sorted_vals = sorted(values)
    min_v = float(sorted_vals[0])
    max_v = float(sorted_vals[-1])
    media = float(statistics.mean(values))
    std = float(statistics.stdev(values)) if n > 1 else 0.0
    
    # Cuartiles Q1 y Q3
    q1 = float(np.percentile(sorted_vals, 25))
    q3 = float(np.percentile(sorted_vals, 75))
    iqr = q3 - q1

    return {
        "n": n,
        "mediana": round(med, 2),
        "iqr": round(iqr, 2),
        "min": round(min_v, 2),
        "max": round(max_v, 2),
        "media": round(media, 2),
        "std": round(std, 2),
        "p95": round(float(np.percentile(sorted_vals, 95)), 2),
        "p99": round(float(np.percentile(sorted_vals, 99)), 2),
    }


def main():
    torch.set_num_threads(2)
    bin_path = "paged_experts_32.bin"
    output_json_path = "prueba_residencia_results.json"

    print("================================================================================")
    print(" PROTOCOLO DE AUDITORÍA: PRUEBA DE RESIDENCIA E INSTRUMENTACIÓN DE E/S (EJE 2)")
    print(" Basic-Lite 0.2 (TRDM-v5): 32 Expertos en Disco, Presupuesto Estricto de 2 Slots")
    print("================================================================================\n")

    # Medir overhead de lectura de /proc/self/io
    io_overhead = measure_io_stats_overhead(iterations=30)
    print(f"[Calibración] Overhead medido de lectura en /proc/self/io: {io_overhead} bytes rchar.")

    # Crear o verificar archivo de 32 expertos
    create_paged_experts_file(bin_path, num_experts=32)

    # -------------------------------------------------------------------------
    # FASE 1: VALIDACIÓN FUNCIONAL CONTRA REFERENCIA RESIDENTE
    # -------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print(" FASE 1: VALIDACIÓN FUNCIONAL (Equivalencia Numérica y Política de Slots)")
    print("--------------------------------------------------------------------------------")

    ref_engine = ReferenceRAMEngine(bin_path, num_experts=32)
    paged_engine = PagedResidentEngine(bin_path, max_slots=2, io_overhead=io_overhead)

    # Secuencia determinista de 50 accesos que incluye alternancia, bucles y accesos a nuevos expertos
    seq_func = [0, 1, 0, 1, 2, 0, 2, 3, 4, 3, 4, 5, 5, 0, 1, 2, 3, 4, 5, 6, 7, 6, 7, 8, 9, 8, 10, 11, 10, 11]
    seq_func += [12, 13, 14, 15, 12, 13, 0, 1, 0, 1, 31, 30, 31, 30, 15, 14, 2, 3, 0, 1]
    assert len(seq_func) == 50

    max_diff_overall = 0
    slot_occupancy_violations = 0

    for step, e_idx in enumerate(seq_func):
        torch.manual_seed(5000 + step)
        x_int8 = torch.randint(-128, 127, (4096,), dtype=torch.int8)

        y_ref = ref_engine.forward(e_idx, x_int8)
        y_paged, rec = paged_engine.forward(e_idx, x_int8)

        diff = (y_ref - y_paged).abs().max().item()
        if diff > max_diff_overall:
            max_diff_overall = diff

        if len(paged_engine.slots) > 2:
            slot_occupancy_violations += 1

    print(f"  Total de llamadas de verificación:       {len(seq_func)}")
    print(f"  Diferencia numérica máxima (Ref vs Pag): {max_diff_overall} (Esperado: 0)")
    print(f"  Violaciones de capacidad (slots > 2):    {slot_occupancy_violations} (Esperado: 0)")
    print(f"  Aciertos residentes (Hit LRU):           {paged_engine.resident_hits} ({paged_engine.resident_hits/len(seq_func)*100:.1f}%)")
    print(f"  Fallos de aplicación (Miss LRU):         {paged_engine.faults_page_cache_hot + paged_engine.faults_storage_observed}")
    print(f"  Desalojos ejecutados:                    {paged_engine.evictions}")
    print(f"  Ocupación final en memoria de slots:     {len(paged_engine.slots)} / {paged_engine.max_slots}")

    paged_engine.close()

    assert max_diff_overall == 0, f"Fallo de equivalencia numérica: max_diff = {max_diff_overall}"
    assert slot_occupancy_violations == 0, "Fallo en invariante de slots de memoria"

    # -------------------------------------------------------------------------
    # FASE 2: MEDICIÓN DE RENDIMIENTO (100 LLAMADAS DETERMINISTAS DE ENRUTAMIENTO)
    # -------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print(" FASE 2: MEDICIÓN DE RENDIMIENTO (100 Llamadas Deterministas y 3 Regímenes)")
    print("--------------------------------------------------------------------------------")

    perf_engine = PagedResidentEngine(bin_path, max_slots=2, io_overhead=io_overhead)

    # Diseñar secuencia de 100 llamadas deterministas:
    # - 10 llamadas iniciales con fadvise para medir "Fallo con lectura de almacenamiento observada"
    # - 45 llamadas con "Fallo de aplicación, caché del sistema caliente" (barridos sobre expertos no residentes)
    # - 45 llamadas con "Acierto residente" (ráfagas de reuso de los 2 expertos cargados)
    np.random.seed(42)
    seq_100 = []
    force_miss_flags = []

    # Bloque 1: 10 accesos en frío con fadvise
    for e in range(10):
        seq_100.append(e)
        force_miss_flags.append(True)

    # Bloque 2: Ráfagas de aciertos residentes y fallos en caché caliente
    # Repetir pares de expertos (ej. 3 llamadas seguidas al mismo par) y luego saltar a un nuevo par
    for pair_idx in range(15):
        e_a = (pair_idx * 2 + 10) % 32
        e_b = (pair_idx * 2 + 11) % 32
        # Llamar e_a (fallo caliente), e_b (fallo caliente), luego e_a (hit), e_b (hit), e_a (hit), e_b (hit)
        sub_seq = [e_a, e_b, e_a, e_b, e_a, e_b]
        for item in sub_seq:
            if len(seq_100) < 100:
                seq_100.append(item)
                force_miss_flags.append(False)

    assert len(seq_100) == 100
    assert len(force_miss_flags) == 100

    for idx, (e_idx, force_miss) in enumerate(zip(seq_100, force_miss_flags)):
        torch.manual_seed(7000 + idx)
        x_int8 = torch.randint(-128, 127, (4096,), dtype=torch.int8)
        _, rec = perf_engine.forward(e_idx, x_int8, force_storage_miss=force_miss)

    # -------------------------------------------------------------------------
    # TANDA EXTENDIDA: 200 LLAMADAS ADICIONALES PARA CARACTERIZAR COLAS
    # -------------------------------------------------------------------------
    print("\n[Tanda Extendida] Ejecutando 200 llamadas adicionales para colas P95/P99...")
    for idx in range(200):
        # Distribución de acceso Zipfiana centrada en 8 expertos populares para simular enrutamiento real
        popular_pool = [0, 1, 2, 3, 4, 5, 6, 7]
        if np.random.rand() < 0.75:
            e_idx = int(np.random.choice(popular_pool))
        else:
            e_idx = int(np.random.randint(8, 32))

        torch.manual_seed(9000 + idx)
        x_int8 = torch.randint(-128, 127, (4096,), dtype=torch.int8)
        perf_engine.forward(e_idx, x_int8, force_storage_miss=False)

    perf_engine.close()

    # -------------------------------------------------------------------------
    # ANÁLISIS ESTADÍSTICO DE RESULTADOS POR RÉGIMEN (SOBRE LAS PRIMERAS 100 LLAMADAS)
    # -------------------------------------------------------------------------
    records_100 = perf_engine.call_history[:100]

    by_regime = {
        "Acierto residente": [],
        "Fallo de aplicación, caché del sistema caliente": [],
        "Fallo con lectura de almacenamiento observada": [],
    }

    for r in records_100:
        by_regime[r["regime"]].append(r)

    print("\n================================================================================")
    print(" RESUMEN DE LAS 100 LLAMADAS DETERMINISTAS (CLASIFICACIÓN DE RÉGIMEN)")
    print("================================================================================")
    tot_solicitados = sum(r["bytes_solicitados"] for r in records_100)
    tot_devueltos = sum(r["bytes_devueltos"] for r in records_100)
    tot_rchar = sum(r["delta_rchar"] for r in records_100)
    tot_storage = sum(r["delta_read_bytes"] for r in records_100)

    print(f"Total consultas:                           {len(records_100)}")
    print(f"Presupuesto pesos empaquetados residentes: 2 slots (1.0 MiB)")
    print(f"Total Bytes Solicitados a pread:           {tot_solicitados:,} bytes ({tot_solicitados / (1024*1024):.2f} MiB)")
    print(f"Total Bytes Devueltos por pread:           {tot_devueltos:,} bytes ({tot_devueltos / (1024*1024):.2f} MiB)")
    print(f"Total rchar devuelto (neto pread + proc):  {tot_rchar:,} bytes (diferencia instrumentación: {tot_rchar - tot_devueltos} bytes)")
    print(f"Total read_bytes (disco físico NVMe):      {tot_storage:,} bytes ({tot_storage / (1024*1024):.2f} MiB)\n")

    regime_summaries = {}

    for reg_name, reg_records in by_regime.items():
        n = len(reg_records)
        pct = (n / len(records_100)) * 100.0

        t_reads = [r["t_read_us"] for r in reg_records]
        t_preps = [r["t_prep_us"] for r in reg_records]
        t_m1s = [r["t_mult1_us"] for r in reg_records]
        t_m2s = [r["t_mult2_us"] for r in reg_records]
        t_residuals = [r["t_residual_us"] for r in reg_records]
        t_totals = [r["t_total_us"] for r in reg_records]

        bytes_sol = sum(r["bytes_solicitados"] for r in reg_records)
        bytes_dev = sum(r["bytes_devueltos"] for r in reg_records)
        rchar_sum = sum(r["delta_rchar"] for r in reg_records)
        read_bytes_sum = sum(r["delta_read_bytes"] for r in reg_records)

        reg_stats = {
            "conteo": n,
            "porcentaje": round(pct, 1),
            "bytes_solicitados_pread": bytes_sol,
            "bytes_devueltos_pread": bytes_dev,
            "rchar_neto": rchar_sum,
            "read_bytes_storage": read_bytes_sum,
            "t_lectura_us": compute_statistics_summary(t_reads),
            "t_prep_us": compute_statistics_summary(t_preps),
            "t_mult1_us": compute_statistics_summary(t_m1s),
            "t_mult2_us": compute_statistics_summary(t_m2s),
            "t_residual_us": compute_statistics_summary(t_residuals),
            "t_total_us": compute_statistics_summary(t_totals),
        }
        regime_summaries[reg_name] = reg_stats

        print(f"---> Régimen: {reg_name}")
        print(f"  Conteos: {n}/100 ({pct:.1f}%) | Bytes Pread: {bytes_dev:,} | rchar: {rchar_sum:,} | read_bytes: {read_bytes_sum:,}")
        tot_stat = reg_stats["t_total_us"]
        read_stat = reg_stats["t_lectura_us"]
        prep_stat = reg_stats["t_prep_us"]
        m1_stat = reg_stats["t_mult1_us"]
        m2_stat = reg_stats["t_mult2_us"]
        res_stat = reg_stats["t_residual_us"]
        print(f"  Mediana del tiempo total medido por llamada : {tot_stat['mediana']} us | IQR={tot_stat['iqr']} us | Rango=[{tot_stat['min']}, {tot_stat['max']}] us | Media={tot_stat['media']} +- {tot_stat['std']} us")
        print(f"  Desglose de Medianas por Etapa (us)         : Lectura={read_stat['mediana']} | Prep={prep_stat['mediana']} | Mult1(4096->256)={m1_stat['mediana']} | Mult2(256->4096)={m2_stat['mediana']} | Residual(LRU/gestor)={res_stat['mediana']}")
        print()

    # Resumen de la tanda completa de 300 llamadas
    extended_totals = [r["t_total_us"] for r in perf_engine.call_history]
    extended_stats = compute_statistics_summary(extended_totals)

    final_payload = {
        "metadata": {
            "date": "2026-11-16",
            "model": "TRDM-v5",
            "total_experts": 32,
            "expert_packed_bytes": 524288,
            "file_size_bytes": os.path.getsize(bin_path),
            "resident_slots_limit": 2,
            "resident_packed_weight_budget_bytes": 1048576,
            "io_mechanism": "explicit pread(2) with /proc/self/io accounting",
            "io_overhead_subtracted_rchar": io_overhead,
        },
        "functional_validation": {
            "total_checks": 50,
            "max_abs_diff_against_ram_reference": max_diff_overall,
            "slot_capacity_violations": slot_occupancy_violations,
            "resident_hits": paged_engine.resident_hits,
            "app_faults": paged_engine.faults_page_cache_hot + paged_engine.faults_storage_observed,
            "evictions": paged_engine.evictions,
        },
        "performance_100_calls": regime_summaries,
        "performance_300_calls_extended": {
            "total_calls": len(perf_engine.call_history),
            "resident_hits": perf_engine.resident_hits,
            "faults_page_cache_hot": perf_engine.faults_page_cache_hot,
            "faults_storage_observed": perf_engine.faults_storage_observed,
            "evictions": perf_engine.evictions,
            "latencies_overall_us": extended_stats,
        },
    }

    with open(output_json_path, "w") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\n[Completado] Reporte de auditoría de residencia guardado en {output_json_path}")


if __name__ == "__main__":
    main()
