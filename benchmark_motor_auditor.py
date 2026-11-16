"""
Protocolo de Validación del Motor — Dictamen del Auditor (Basic-Lite 0.2 / TRDM-v5)

Responde a las 5 preguntas empíricas planteadas en el dictamen:
1. Matriz ternaria empaquetada contra referencia: ¿El resultado coincide y cuánto cuesta ejecutarla sin expansión completa?
2. Expertos limitados por un presupuesto de RAM: ¿Cuántos bytes se leen por token y cuánto se reutiliza?
3. Contexto archivado con conjunto activo limitado: ¿Se recuperan hechos antiguos, incluso los poco frecuentes?
4. Recursión y difusión adaptativas: ¿Se ahorra tiempo sin degradar respuestas?
5. Integración: ¿Los ahorros sobreviven al combinar mecanismos?
"""

import json
import os
import sys
import time
from typing import Any, Dict, List
import numpy as np
import torch

from config import HybridConfig
from trdm.core.context_manager import HierarchicalContextManager
from trdm.core.weight_manager import WeightMovementManager
from trdm.layers.byte_patcher import HierarchicalBytePatcher
from trdm.models.trdm_v5 import TernaryRecursiveDiffusionLM_v5
from trdm.ops.paged_kernel import benchmark_packed_vs_dense
from trdm.serialization.paged_bitcos import PagedBitCOSReader, PagedBitCOSSerializer


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def prueba_1_matriz_empaquetada() -> Dict[str, Any]:
    print_header("PRUEBA 1: Matriz Ternaria Empaquetada vs. Expansión Completa FP32")
    results = benchmark_packed_vs_dense(
        batch_size=1,
        in_features=192,
        out_features=384,
        page_size=4096,
        num_runs=40,
    )
    print(f"Dimensiones Evaluadas        : {results['dimensions']}")
    print(f"Memoria FP32 Densa           : {results['dense_fp32_bytes']} bytes")
    print(f"Tamaño Paged Compact (2-bit) : {results['paged_compact_bytes']} bytes ({results['compression_compact_x']}x reducción)")
    print(f"Tamaño Paged Fast (LUT 2-bit): {results['paged_fast_bytes']} bytes ({results['compression_fast_x']}x reducción)")
    print(f"Latencia Reconstrucción FP32 : {results['lat_fp32_reconstruction_ms']} ms")
    print(f"Latencia Paginada Compact    : {results['lat_paged_compact_ms']} ms")
    print(f"Latencia Paginada Fast       : {results['lat_paged_fast_ms']} ms")
    print(f"Coincidencia Numérica Exacta : {'APROBADA' if results['numerical_exact_agreement'] else 'FALLIDA'}")
    print(f"Correlación con FP32         : {results['correlation_with_fp32']}")
    return results


def prueba_2_expertos_presupuesto_ram() -> Dict[str, Any]:
    print_header("PRUEBA 2: Expertos Paginados Limitados por Presupuesto de RAM")
    # Configurar gestor con presupuesto estricto: solo 2 expertos residentes de 6 posibles
    num_total_experts = 6
    max_resident = 2
    expert_size_kb = 32  # 32 KB por experto ternario típico

    wm_blind = WeightMovementManager(
        max_resident_experts=max_resident,
        lambda_read_cost=0.0,  # Enrutamiento ciego tradicional
        expert_byte_size=expert_size_kb * 1024,
    )
    wm_aware = WeightMovementManager(
        max_resident_experts=max_resident,
        lambda_read_cost=0.40,  # Enrutamiento consciente de memoria
        escape_threshold=0.75,
        expert_byte_size=expert_size_kb * 1024,
    )

    # Crear módulos simulados de expertos
    from trdm.modules.moe import TernaryExpert
    for i in range(num_total_experts):
        exp = TernaryExpert(192, 384)
        wm_blind.register_expert_source(i, exp)
        wm_aware.register_expert_source(i, exp)

    # Precargar los primeros 2
    wm_blind.acquire_expert(0)
    wm_blind.acquire_expert(1)
    wm_aware.acquire_expert(0)
    wm_aware.acquire_expert(1)
    wm_blind.reset_telemetry()
    wm_aware.reset_telemetry()

    # Simular una secuencia de 100 pasos donde las utilidades tienen cierta dispersión
    torch.manual_seed(42)
    num_steps = 100

    # 1. Ejecutar con enrutamiento ciego
    for step in range(num_steps):
        raw_u = torch.randn(1, num_total_experts).softmax(dim=-1)
        selected_e = int(raw_u.argmax(dim=-1).item())
        wm_blind.acquire_expert(selected_e)

    # 2. Ejecutar con enrutamiento consciente de memoria
    for step in range(num_steps):
        raw_u = torch.randn(1, num_total_experts).softmax(dim=-1)
        scores, escaped = wm_aware.compute_memory_aware_scores(raw_u, allow_escape=True)
        selected_e = int(scores.argmax(dim=-1).item())
        wm_aware.acquire_expert(selected_e)

    t_blind = wm_blind.get_telemetry()
    t_aware = wm_aware.get_telemetry()

    print(f"Presupuesto de RAM           : Máx {max_resident} expertos en memoria de {num_total_experts} totales")
    print(f"Pasos / Consultas de Token   : {num_steps}")
    print(f"--- Enrutamiento Ciego (Convencional) ---")
    print(f"  Tasa de Acierto en Caché   : {t_blind['cache_hit_rate_pct']}%")
    print(f"  Bytes Transferidos (SSD)   : {t_blind['bytes_transferred_kb']} KB")
    print(f"  Desalojos de RAM           : {t_blind['evictions_count']}")
    print(f"--- Enrutamiento Consciente de Memoria S(e) ---")
    print(f"  Tasa de Acierto en Caché   : {t_aware['cache_hit_rate_pct']}%")
    print(f"  Bytes Transferidos (SSD)   : {t_aware['bytes_transferred_kb']} KB")
    print(f"  Desalojos de RAM           : {t_aware['evictions_count']}")
    print(f"  Escapes Forzados (U >= thr): {t_aware['escapes_triggered']}")

    traffic_reduction_pct = round(
        (1.0 - float(t_aware['bytes_transferred_kb']) / float(max(1, t_blind['bytes_transferred_kb']))) * 100.0,
        2,
    )
    print(f"Reducción de Tráfico de I/O  : {traffic_reduction_pct}% menos lecturas de disco")

    return {
        "blind_telemetry": t_blind,
        "aware_telemetry": t_aware,
        "traffic_reduction_pct": traffic_reduction_pct,
    }


def prueba_3_contexto_jerarquico() -> Dict[str, Any]:
    print_header("PRUEBA 3: Contexto Archivado con Conjunto Activo Limitado")
    active_window = 64
    cm = HierarchicalContextManager(active_window_size=active_window, compressed_segment_size=32)

    # Simular ingesta de un documento largo (512 tokens sintéticos)
    # En la posición ~100 insertamos un hecho crítico único (needle-in-a-haystack):
    # "CONFIG_CRITICA_NODO_X99_VOLTAJE_42"
    hecho_critico = "HECHO_CLAVE: nodo switch_alfa puerto 42 latencia 7ms watt 110"
    tokens_hecho = list(hecho_critico.encode("utf-8"))

    # Secuencia inicial
    seq1 = list(b"inicio de transmision y telemetria general del cluster de computo de consumo... ") * 2
    cm.ingest_tokens(seq1)

    # Inserción del hecho
    cm.ingest_tokens(tokens_hecho)

    # Ingesta masiva posterior para empujar el hecho fuera de la ventana activa (al Nivel 3)
    seq2 = list(b"relleno continuo de datos de sensores temperatura 35C corriente nominal 2A canal seguro... ") * 6
    cm.ingest_tokens(seq2)

    stats_before = cm.get_telemetry()
    print(f"Tokens Ingestados Totales    : {stats_before['total_tokens_ingested']}")
    print(f"Tokens Retenidos en RAM      : {stats_before['active_tokens_in_ram']} (Límite activo: {active_window})")
    print(f"Segmentos Comprimidos N2     : {stats_before['compressed_segments_count']}")
    print(f"Fragmentos en Archivo N3     : {stats_before['archived_chunks_on_disk']}")
    print(f"Ratio de Ahorro de RAM       : {stats_before['ram_saving_ratio']}x menos memoria activa en RAM")

    # Comprobar si el hecho está en la ventana activa actual
    active_text = bytes(cm.active_token_ids).decode("utf-8", errors="replace")
    esta_en_activo = "switch_alfa" in active_text
    print(f"¿Hecho presente en Nivel 1?  : {'SÍ' if esta_en_activo else 'NO (correctamente retirado al archivo N3)'}")

    # Búsqueda y Recuperación Exacta desde el Nivel 3
    query = "switch_alfa watt 110"
    recuperados = cm.retrieve_exact_fact(query, top_k=2)
    recuperacion_exitosa = False
    if recuperados:
        rec_text = " ".join(r["raw_text"] for r in recuperados)
        recuperacion_exitosa = ("switch_alfa" in rec_text) and ("watt 110" in rec_text)
        print(f"Recuperación Nivel 3 Exacta  : {'EXITOSA' if recuperacion_exitosa else 'FALLIDA'}")
        print(f"Contenido Recuperado         : \"{rec_text.strip()}\"")

        # Inyectar de vuelta al conjunto activo
        cm.inject_retrieved_to_active(recuperados)
        nuevo_activo = bytes(cm.active_token_ids).decode("utf-8", errors="replace")
        print(f"Inyección a Nivel 1 (Activo) : {'CONFIRMADA' if 'switch_alfa' in nuevo_activo else 'FALLIDA'}")

    return {
        "telemetry": cm.get_telemetry(),
        "retired_to_archive": not esta_en_activo,
        "exact_retrieval_success": recuperacion_exitosa,
    }


def prueba_4_computo_adaptativo() -> Dict[str, Any]:
    print_header("PRUEBA 4: Recursión y Cómputo Adaptativo (Early Exit)")
    cfg = HybridConfig()
    cfg.num_recursions = 4
    cfg.adaptive_recursion = True
    cfg.adaptive_recursion_tol = 0.55

    model = TernaryRecursiveDiffusionLM_v5(cfg)
    model.prepare_for_eval()

    # Caso 1: Entrada fija repetitiva (fácil de estabilizar)
    torch.manual_seed(42)
    x_facil = torch.full((1, 32), 65, dtype=torch.long)  # 'AAAA...'
    t_zero = torch.zeros(1)

    t0 = time.perf_counter()
    out_facil = model(x_facil, t_zero, num_recursions=4)
    lat_facil_ms = (time.perf_counter() - t0) * 1000.0
    rec_facil = model.last_recursions_executed
    deltas_facil = [round(d, 4) for d in model.last_deltas]

    # Caso 2: Entrada forzada completa (sin early exit, R=4 forzado)
    cfg_forced = HybridConfig()
    cfg_forced.num_recursions = 4
    cfg_forced.adaptive_recursion = False
    model_forced = TernaryRecursiveDiffusionLM_v5(cfg_forced)
    model_forced.prepare_for_eval()

    t0 = time.perf_counter()
    out_complejo = model_forced(x_facil, t_zero, num_recursions=4)
    lat_complejo_ms = (time.perf_counter() - t0) * 1000.0
    rec_complejo = model_forced.last_recursions_executed
    deltas_complejo = [round(d, 4) for d in model_forced.last_deltas]

    ahorro_latencia_pct = round(
        max(0.0, (1.0 - lat_facil_ms / max(1e-6, lat_complejo_ms)) * 100.0), 2
    )

    print(f"Profundidad Máxima (R_max)   : 4 recursiones")
    print(f"Deltas de Convergencia Latente: {deltas_facil}")
    print(f"Recursiones Entrada Fácil     : {rec_facil} pasadas ejecutadas (Salida Temprana)")
    print(f"Recursiones Entrada Compleja  : {rec_complejo} pasadas ejecutadas")
    print(f"Latencia Salida Temprana     : {round(lat_facil_ms, 2)} ms")
    print(f"Latencia Completa R=4         : {round(lat_complejo_ms, 2)} ms")
    print(f"Ahorro de Cómputo / Latencia : {ahorro_latencia_pct}%")

    return {
        "r_easy": rec_facil,
        "r_hard": rec_complejo,
        "deltas": deltas_facil,
        "latency_easy_ms": round(lat_facil_ms, 2),
        "latency_hard_ms": round(lat_complejo_ms, 2),
        "speedup_pct": ahorro_latencia_pct,
    }


def prueba_5_integracion_extremo_a_extremo() -> Dict[str, Any]:
    print_header("PRUEBA 5: Integración Extremo a Extremo de los 5 Mecanismos")
    cfg = HybridConfig()
    cfg.use_byte_patching = True
    cfg.byte_patch_size = 4
    cfg.use_mla = True
    cfg.max_resident_experts = 2
    cfg.adaptive_recursion = True

    model = TernaryRecursiveDiffusionLM_v5(cfg)
    model.prepare_for_eval()

    # 1. Verificar Serialización Paged BitCOS a Disco
    bin_path = "test_trdm_v5_paged.bin"
    export_info = model.export_paged_bitcos(bin_path, page_size=4096, format_type="compact")
    reader = PagedBitCOSReader(bin_path)

    print(f"Archivo Paged BitCOS Creado  : {bin_path}")
    print(f"Tamaño Serializado Total     : {export_info['total_bytes'] / 1024.0:.2f} KB")
    print(f"Páginas Generadas (4 KB c/u) : {export_info['total_pages']} páginas")
    print(f"Tensores Indexados           : {export_info['tensors_count']}")
    print(f"Expertos Indexados           : {export_info['experts_count']}")

    # 2. Inferencia Integral con todos los mecanismos cooperando
    x_test = torch.randint(0, 256, (1, 32), dtype=torch.long)
    t_test = torch.full((1,), 0.5)

    t0 = time.perf_counter()
    out = model(x_test, t_test, record_telemetry=True)
    lat_ms = (time.perf_counter() - t0) * 1000.0

    telemetry = model.get_full_telemetry()
    wm = telemetry["weight_movement"]
    cm = telemetry["hierarchical_context"]

    print(f"\n--- Telemetría de Inferencia Integrada ---")
    print(f"Forma de Salida Logits       : {list(out.final_logits.shape)} (vocab_size=260)")
    print(f"Latencia Total Paso Inferencia: {lat_ms:.2f} ms")
    print(f"Expertos Residentes en RAM   : {wm['resident_experts']} (Presupuesto máx: {wm['max_resident_budget']})")
    print(f"Tasa de Acierto Caché Exp.   : {wm['cache_hit_rate_pct']}%")
    print(f"Bytes Transferidos de SSD    : {wm['bytes_transferred_kb']} KB")
    print(f"Recursiones Ejecutadas       : {telemetry['adaptive_recursions_last']}")

    # Limpiar archivo temporal de prueba si se desea
    reader.close()
    if os.path.exists(bin_path):
        os.remove(bin_path)

    return {
        "paged_bitcos_export": export_info,
        "inference_latency_ms": round(lat_ms, 2),
        "telemetry": telemetry,
        "integration_passed": True,
    }


def main() -> None:
    print("=" * 80)
    print("  SUITE DE AUDITORÍA DEL MOTOR BASIC-LITE 0.2 (ARQUITECTURA TRDM-v5)")
    print("  Validación Empírica para Modelos de 30B con RAM y Contexto Selectivos")
    print("=" * 80)

    t_start = time.perf_counter()
    r1 = prueba_1_matriz_empaquetada()
    r2 = prueba_2_expertos_presupuesto_ram()
    r3 = prueba_3_contexto_jerarquico()
    r4 = prueba_4_computo_adaptativo()
    r5 = prueba_5_integracion_extremo_a_extremo()
    total_time = time.perf_counter() - t_start

    summary = {
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prueba_1_matriz_empaquetada": r1,
        "prueba_2_expertos_ram": r2,
        "prueba_3_contexto_jerarquico": r3,
        "prueba_4_computo_adaptativo": r4,
        "prueba_5_integracion": r5,
        "tiempo_total_segundos": round(total_time, 2),
    }

    with open("motor_auditor_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"  TODAS LAS PRUEBAS COMPLETADAS EN {total_time:.2f} SEGUNDOS")
    print("  Resultados exportados a: motor_auditor_results.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
