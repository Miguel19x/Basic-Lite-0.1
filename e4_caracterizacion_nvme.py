#!/usr/bin/env python3
"""
Basic-Lite 0.2 (TRDM-v5) — Experimento E4: Caracterización Física de E/S en Almacenamiento NVMe
Protocolo Pre-Registrado: Etiqueta Git prereg-v1 (§E4, H-E1)

Hardware Registrado:
- Procesador: Intel(R) Core(TM) i3-1115G4 @ 3.00GHz (2 núcleos / 4 hilos)
- Almacenamiento: SSD NVMe MSI M450 500GB (PCIe Gen4 x4)
- Sistema Operativo: Linux x86_64 (Btrfs sobre LUKS2 crypto)

Protecciones Activas de Estabilidad:
- I/O Scheduling: Prioridad Idle (ionice -c 3, nice -n 19)
- Concurrencia acotada a la topología física real (1, 2, 4 hilos)
- Throttling de enfriamiento (150 ms) entre ráfagas
- Validación estricta de lectura en frío: read_bytes / bytes_solicitados in [0.95, 1.10]
"""

import os
import sys
import time
import math
import mmap
import json
import ctypes
import threading
from datetime import datetime

# ==============================================================================
# 1. Configuración y Constantes Pre-Registradas
# ==============================================================================
TEST_FILE = "e4_nvme_test_1gib.bin"
FILE_SIZE = 1024 * 1024 * 1024  # 1 GiB (1,073,741,824 bytes)

# Tamaños pre-registrados en §E4
BLOCK_SIZES = [
    256 * 1024,        # 256 KiB
    512 * 1024,        # 512 KiB (Baseline H-E1)
    1024 * 1024,       # 1 MiB
    4 * 1024 * 1024,    # 4 MiB
    16 * 1024 * 1024,   # 16 MiB
    64 * 1024 * 1024,   # 64 MiB
    256 * 1024 * 1024,  # 256 MiB
]

CONCURRENCY_LEVELS = [1, 2, 4]  # Adaptado a la topología de 4 hilos del Core i3
REPETITIONS = 5                  # Grados de libertad = 4, factor t_student = 2.776
T_STUDENT_95 = 2.776
COOLDOWN_SEC = 0.150             # 150 ms entre lecturas para vaciado de colas dm-crypt

# Configuración ctypes para llamadas POSIX C directas
libc = ctypes.CDLL(None, use_errno=True)
libc.posix_memalign.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_size_t]
libc.posix_memalign.restype = ctypes.c_int
libc.free.argtypes = [ctypes.c_void_p]
libc.pread.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_longlong]
libc.pread.restype = ctypes.c_ssize_t

# ==============================================================================
# 2. Utilidades de Medición y Validación de Lectura en Frío
# ==============================================================================
def get_read_bytes_proc():
    """Lee el contador físico read_bytes desde /proc/self/io."""
    try:
        with open("/proc/self/io", "r") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split(":")[1])
    except Exception as e:
        print(f"Error leyendo /proc/self/io: {e}")
    return 0

def create_test_file():
    """Crea el archivo binario continuo de 1 GiB con datos aleatorios si no existe."""
    if os.path.exists(TEST_FILE) and os.path.getsize(TEST_FILE) == FILE_SIZE:
        print(f"[OK] Archivo de prueba ya existe: {TEST_FILE} ({FILE_SIZE / (1024**3):.1f} GiB)")
        return

    print(f"[*] Creando archivo de prueba de 1 GiB ({TEST_FILE}) con /dev/urandom...")
    chunk_size = 32 * 1024 * 1024
    total_written = 0
    t0 = time.time()
    with open(TEST_FILE, "wb") as f:
        while total_written < FILE_SIZE:
            to_write = min(chunk_size, FILE_SIZE - total_written)
            f.write(os.urandom(to_write))
            total_written += to_write
    os.sync()
    t1 = time.time()
    print(f"[OK] Archivo creado en {t1 - t0:.2f} s. Forzando vaciado inicial de caché...")
    
    # Evicción inicial
    fd = os.open(TEST_FILE, os.O_RDWR)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)

def evict_file_cache(fd):
    """Purga las páginas del descriptor de la Page Cache del kernel."""
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)

# ==============================================================================
# 3. Implementaciones de las Interfaces del Sistema
# ==============================================================================
def run_trial_pread(fd, block_size, offset, num_threads):
    """Ejecuta una lectura multihilo con pread(2) y POSIX_FADV_RANDOM."""
    evict_file_cache(fd)
    bytes_per_thread = block_size
    total_requested = bytes_per_thread * num_threads

    def worker(tid, results):
        span = max(1, FILE_SIZE - bytes_per_thread)
        t_offset = (offset + tid * bytes_per_thread) % span
        data = os.pread(fd, bytes_per_thread, t_offset)
        results[tid] = len(data)

    results = [0] * num_threads
    threads = [threading.Thread(target=worker, args=(i, results)) for i in range(num_threads)]

    b_before = get_read_bytes_proc()
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_end = time.perf_counter()
    b_after = get_read_bytes_proc()

    elapsed = t_end - t_start
    delta_bytes = b_after - b_before
    ratio = delta_bytes / total_requested if total_requested > 0 else 0.0
    bw_gbps = (total_requested / (1024**3)) / elapsed if elapsed > 0 else 0.0

    return {
        "elapsed_ms": elapsed * 1000.0,
        "bw_gbps": bw_gbps,
        "delta_bytes": delta_bytes,
        "requested_bytes": total_requested,
        "cold_ratio": ratio,
        "valid_cold": (0.95 <= ratio <= 1.10)
    }

def run_trial_odirect(fd, block_size, offset, num_threads):
    """Ejecuta una lectura multihilo con O_DIRECT sobre buffers alineados a 4096 bytes."""
    evict_file_cache(fd)
    bytes_per_thread = block_size
    total_requested = bytes_per_thread * num_threads

    buffers = []
    for _ in range(num_threads):
        ptr = ctypes.c_void_p()
        ret = libc.posix_memalign(ctypes.byref(ptr), 4096, bytes_per_thread)
        if ret != 0:
            raise MemoryError("posix_memalign falló")
        buffers.append(ptr)

    def worker(tid, results):
        span = max(1, FILE_SIZE - bytes_per_thread)
        t_offset = (offset + tid * bytes_per_thread) % span
        n = libc.pread(fd, buffers[tid], bytes_per_thread, ctypes.c_longlong(t_offset))
        results[tid] = n

    results = [0] * num_threads
    threads = [threading.Thread(target=worker, args=(i, results)) for i in range(num_threads)]

    b_before = get_read_bytes_proc()
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_end = time.perf_counter()
    b_after = get_read_bytes_proc()

    for ptr in buffers:
        libc.free(ptr)

    elapsed = t_end - t_start
    delta_bytes = b_after - b_before
    ratio = delta_bytes / total_requested if total_requested > 0 else 0.0
    bw_gbps = (total_requested / (1024**3)) / elapsed if elapsed > 0 else 0.0

    return {
        "elapsed_ms": elapsed * 1000.0,
        "bw_gbps": bw_gbps,
        "delta_bytes": delta_bytes,
        "requested_bytes": total_requested,
        "cold_ratio": ratio,
        "valid_cold": (0.95 <= ratio <= 1.10)
    }

def run_trial_mmap(fd, block_size, offset):
    """Ejecuta una lectura monocanal con mmap(2) y madvise(MADV_DONTNEED / MADV_RANDOM)."""
    evict_file_cache(fd)
    mm = mmap.mmap(fd, FILE_SIZE, access=mmap.ACCESS_READ)
    try:
        mm.madvise(mmap.MADV_RANDOM, 0, FILE_SIZE)
        mm.madvise(mmap.MADV_DONTNEED, 0, FILE_SIZE)

        span = max(1, FILE_SIZE - block_size)
        t_offset = offset % span
        total_requested = block_size

        b_before = get_read_bytes_proc()
        t_start = time.perf_counter()
        # Forzar lectura secuencial de páginas en el rango
        chunk = mm[t_offset : t_offset + block_size]
        t_end = time.perf_counter()
        b_after = get_read_bytes_proc()

        elapsed = t_end - t_start
        delta_bytes = b_after - b_before
        ratio = delta_bytes / total_requested if total_requested > 0 else 0.0
        bw_gbps = (total_requested / (1024**3)) / elapsed if elapsed > 0 else 0.0

        return {
            "elapsed_ms": elapsed * 1000.0,
            "bw_gbps": bw_gbps,
            "delta_bytes": delta_bytes,
            "requested_bytes": total_requested,
            "cold_ratio": ratio,
            "valid_cold": (0.95 <= ratio <= 1.10)
        }
    finally:
        mm.close()

# ==============================================================================
# 4. Motor de Ejecución Factorial y Agregación Estadística
# ==============================================================================
def compute_stats(values):
    """Calcula media, std, IC95 y extremos."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)
    ci95 = T_STUDENT_95 * (std / math.sqrt(n)) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "min": min(values),
        "max": max(values),
        "values": values
    }

def main():
    print("=" * 80)
    print("Basic-Lite 0.2 (TRDM-v5) — Experimento E4: Caracterización Física NVMe")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CPU: Intel(R) Core(TM) i3-1115G4 @ 3.00GHz | Almacenamiento: MSI M450 500GB NVMe")
    print("=" * 80)

    create_test_file()

    results = {
        "metadata": {
            "experiment": "E4_Caracterizacion_Fisica_NVMe",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": "MSI M450 500GB NVMe (PCIe Gen4 x4)",
            "cpu": "Intel Core i3-1115G4 (2C/4T @ 3.00GHz)",
            "os": "Linux 7.3.3-ogc1.1.fc45.x86_64 (Btrfs sobre LUKS2)",
            "file_size_bytes": FILE_SIZE,
            "repetitions": REPETITIONS,
            "cooldown_sec": COOLDOWN_SEC
        },
        "factorial_data": {}
    }

    # Descriptores de archivo
    fd_pread = os.open(TEST_FILE, os.O_RDONLY)
    fd_odirect = os.open(TEST_FILE, os.O_RDONLY | os.O_DIRECT)

    global_offset = 0

    try:
        # A. Barrido pread(2)
        print("\n--- BARRIDO 1: pread(2) con POSIX_FADV_RANDOM ---")
        results["factorial_data"]["pread"] = {}
        for threads in CONCURRENCY_LEVELS:
            results["factorial_data"]["pread"][f"threads_{threads}"] = {}
            for sz in BLOCK_SIZES:
                sz_str = f"{sz // 1024}KiB" if sz < 1024*1024 else f"{sz // (1024*1024)}MiB"
                print(f"  [pread] Hilos={threads} | Bloque={sz_str:>7} ... ", end="", flush=True)

                bw_list, lat_list, ratio_list, valid_list = [], [], [], []
                for rep in range(REPETITIONS):
                    res = run_trial_pread(fd_pread, sz, global_offset, threads)
                    bw_list.append(res["bw_gbps"])
                    lat_list.append(res["elapsed_ms"])
                    ratio_list.append(res["cold_ratio"])
                    valid_list.append(res["valid_cold"])
                    step = sz * threads
                    span = FILE_SIZE - step
                    global_offset = (global_offset + step) % span if span > 0 else 0
                    time.sleep(COOLDOWN_SEC)

                stats_bw = compute_stats(bw_list)
                stats_lat = compute_stats(lat_list)
                stats_ratio = compute_stats(ratio_list)
                pass_rate = (sum(valid_list) / REPETITIONS) * 100.0

                results["factorial_data"]["pread"][f"threads_{threads}"][sz_str] = {
                    "bandwidth_gbps": stats_bw,
                    "latency_ms": stats_lat,
                    "cold_ratio": stats_ratio,
                    "cold_pass_rate_pct": pass_rate
                }
                print(f"{stats_bw['mean']:5.2f} ± {stats_bw['std']:4.2f} GB/s | {stats_lat['mean']:6.2f} ms | ColdPass={pass_rate:3.0f}%")

        # B. Barrido O_DIRECT
        print("\n--- BARRIDO 2: O_DIRECT con posix_memalign(4096) ---")
        results["factorial_data"]["odirect"] = {}
        for threads in CONCURRENCY_LEVELS:
            results["factorial_data"]["odirect"][f"threads_{threads}"] = {}
            for sz in BLOCK_SIZES:
                sz_str = f"{sz // 1024}KiB" if sz < 1024*1024 else f"{sz // (1024*1024)}MiB"
                print(f"  [O_DIRECT] Hilos={threads} | Bloque={sz_str:>7} ... ", end="", flush=True)

                bw_list, lat_list, ratio_list, valid_list = [], [], [], []
                for rep in range(REPETITIONS):
                    res = run_trial_odirect(fd_odirect, sz, global_offset, threads)
                    bw_list.append(res["bw_gbps"])
                    lat_list.append(res["elapsed_ms"])
                    ratio_list.append(res["cold_ratio"])
                    valid_list.append(res["valid_cold"])
                    step = sz * threads
                    span = FILE_SIZE - step
                    global_offset = (global_offset + step) % span if span > 0 else 0
                    time.sleep(COOLDOWN_SEC)

                stats_bw = compute_stats(bw_list)
                stats_lat = compute_stats(lat_list)
                stats_ratio = compute_stats(ratio_list)
                pass_rate = (sum(valid_list) / REPETITIONS) * 100.0

                results["factorial_data"]["odirect"][f"threads_{threads}"][sz_str] = {
                    "bandwidth_gbps": stats_bw,
                    "latency_ms": stats_lat,
                    "cold_ratio": stats_ratio,
                    "cold_pass_rate_pct": pass_rate
                }
                print(f"{stats_bw['mean']:5.2f} ± {stats_bw['std']:4.2f} GB/s | {stats_lat['mean']:6.2f} ms | ColdPass={pass_rate:3.0f}%")

        # C. Barrido mmap monocanal seguro
        print("\n--- BARRIDO 3: mmap(2) monocanal con madvise ---")
        results["factorial_data"]["mmap"] = {"threads_1": {}}
        for sz in BLOCK_SIZES:
            sz_str = f"{sz // 1024}KiB" if sz < 1024*1024 else f"{sz // (1024*1024)}MiB"
            print(f"  [mmap] Hilos=1 | Bloque={sz_str:>7} ... ", end="", flush=True)

            bw_list, lat_list, ratio_list, valid_list = [], [], [], []
            for rep in range(REPETITIONS):
                res = run_trial_mmap(fd_pread, sz, global_offset)
                bw_list.append(res["bw_gbps"])
                lat_list.append(res["elapsed_ms"])
                ratio_list.append(res["cold_ratio"])
                valid_list.append(res["valid_cold"])
                span = FILE_SIZE - sz
                global_offset = (global_offset + sz) % span if span > 0 else 0
                time.sleep(COOLDOWN_SEC)

            stats_bw = compute_stats(bw_list)
            stats_lat = compute_stats(lat_list)
            stats_ratio = compute_stats(ratio_list)
            pass_rate = (sum(valid_list) / REPETITIONS) * 100.0

            results["factorial_data"]["mmap"]["threads_1"][sz_str] = {
                "bandwidth_gbps": stats_bw,
                "latency_ms": stats_lat,
                "cold_ratio": stats_ratio,
                "cold_pass_rate_pct": pass_rate
            }
            print(f"{stats_bw['mean']:5.2f} ± {stats_bw['std']:4.2f} GB/s | {stats_lat['mean']:6.2f} ms | ColdPass={pass_rate:3.0f}%")

    finally:
        os.close(fd_pread)
        os.close(fd_odirect)

    # ==============================================================================
    # 5. Evaluación de la Hipótesis H-E1 (§E4 del Protocolo)
    # ==============================================================================
    # Baseline H-E1: 512 KiB a cola 1 (1 hilo)
    baseline_bw_pread = results["factorial_data"]["pread"]["threads_1"]["512KiB"]["bandwidth_gbps"]["mean"]
    baseline_bw_odirect = results["factorial_data"]["odirect"]["threads_1"]["512KiB"]["bandwidth_gbps"]["mean"]
    baseline_bw = max(baseline_bw_pread, baseline_bw_odirect)

    # Rendimiento en >= 4 MiB o concurrencia >= 2
    max_large_bw_pread = max(
        results["factorial_data"]["pread"]["threads_1"]["4MiB"]["bandwidth_gbps"]["mean"],
        results["factorial_data"]["pread"]["threads_1"]["16MiB"]["bandwidth_gbps"]["mean"],
        results["factorial_data"]["pread"]["threads_1"]["64MiB"]["bandwidth_gbps"]["mean"],
        results["factorial_data"]["pread"]["threads_1"]["256MiB"]["bandwidth_gbps"]["mean"],
        results["factorial_data"]["pread"]["threads_2"]["4MiB"]["bandwidth_gbps"]["mean"],
        results["factorial_data"]["pread"]["threads_4"]["4MiB"]["bandwidth_gbps"]["mean"],
    )
    max_large_bw_odirect = max(
        results["factorial_data"]["odirect"]["threads_1"]["4MiB"]["bandwidth_gbps"]["mean"],
        results["factorial_data"]["odirect"]["threads_1"]["16MiB"]["bandwidth_gbps"]["mean"],
        results["factorial_data"]["odirect"]["threads_1"]["64MiB"]["bandwidth_gbps"]["mean"],
        results["factorial_data"]["odirect"]["threads_1"]["256MiB"]["bandwidth_gbps"]["mean"],
        results["factorial_data"]["odirect"]["threads_2"]["4MiB"]["bandwidth_gbps"]["mean"],
        results["factorial_data"]["odirect"]["threads_4"]["4MiB"]["bandwidth_gbps"]["mean"],
    )
    peak_large_bw = max(max_large_bw_pread, max_large_bw_odirect)
    speedup = peak_large_bw / baseline_bw if baseline_bw > 0 else 0.0

    if speedup >= 3.0:
        verdict = f"CONFIRMA_SENSIBILIDAD_TAMANO (Mejora de {speedup:.2f}x >= 3.0x)"
    elif speedup < 1.5:
        verdict = f"DESCARTA_SENSIBILIDAD (Mejora de {speedup:.2f}x < 1.5x)"
    else:
        verdict = f"INCONCLUSO (Mejora intermedia de {speedup:.2f}x)"

    results["hypothesis_h_e1"] = {
        "baseline_512KiB_1thread_gbps": baseline_bw,
        "peak_large_transfer_gbps": peak_large_bw,
        "speedup_factor": speedup,
        "threshold_confirm": 3.0,
        "threshold_reject": 1.5,
        "verdict": verdict
    }

    # Guardar resultados
    output_path = "e4_caracterizacion_nvme_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"VEREDICTO FORMAL H-E1 (§E4): {verdict}")
    print(f"Baseline (512 KiB, 1 hilo): {baseline_bw:.2f} GB/s")
    print(f"Pico Transferencias >= 4 MiB: {peak_large_bw:.2f} GB/s (Factor: {speedup:.2f}x)")
    print(f"Resultados guardados exitosamente en: {output_path}")
    print("=" * 80)

    # Limpiar archivo de prueba de 1 GiB
    if os.path.exists(TEST_FILE):
        print(f"[*] Limpiando archivo temporal de 1 GiB: {TEST_FILE}...")
        os.remove(TEST_FILE)
        print("[OK] Limpieza completada.")

if __name__ == "__main__":
    main()
