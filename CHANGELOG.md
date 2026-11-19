# Changelog — Basic-Lite TRACE

Todas las modificaciones, versiones y evoluciones arquitectónicas de **Basic-Lite TRACE** quedan registradas en este documento siguiendo el formato de control de versiones y trazabilidad técnica.

---

## [TRACE-1.0.0] — Arquitectura TRACE Convergente — 2026-11-17

### Añadido y Consolidado
* **Bautismo y Adopción de la Arquitectura TRACE:**
  - Consolidación formal de la arquitectura **TRACE** (*Ternary Resident Architecture for Convergent Execution*), aunando la cuantización ternaria W1.58A8, la residencia selectiva paginada a 4096 bytes en Linux, el ruteo MoE consciente de latencia NVMe y el campo causal en 2 etapas con `detach()`.
  - Transición del proyecto desde su génesis fundacional en **Basic-Lite 0.1** hacia la versión definitiva **Basic-Lite TRACE**, unificando la totalidad de los commits y el historial de desarrollo en el repositorio canónico.
  - Publicación y despliegue del régimen de **Licencia de Investigación, Docencia y Evaluación** en [LICENSE.txt](LICENSE.txt).
* **Reorganización Modular y Limpieza de Código Legacy:**
  - Migración y renombrado del paquete modular raíz de Python de `trdm/` a `trace/` para unificar el espacio de nombres con la arquitectura canónica TRACE.
  - Consolidación de todos los registros de evaluación confirmatoria pre-registrada en el directorio `results/`.
  - Purgado de código experimental histórico y de la versión fundacional (`archive/` y `legacy/`) del árbol de trabajo principal, quedando íntegramente preservados bajo las etiquetas Git congeladas `v0.1-archive`, `prereg-v1` y `phase2-v6-stable`.

---

## [0.2.0-v6] — Fase 2 Estable — 2026-11-16
**Etiqueta Git:** `phase2-v6-stable`  
**Commit principal:** `c874140` / `113ee26`

### Añadido
* **Campo Causal Desacoplado en 2 Etapas:**
  - Implementación en `trdm/models/causal.py` (anteriormente `trdm_v6_causal.py`) separando el cálculo de subtotales multiplicativos ($P_w, P_{ms}$) y la agregación aditiva final ($watt, ms$).
  - Anclaje causal con corte de gradiente (`detach()`) para mitigar de raíz el sesgo de exposición (*exposure bias*).
* **Abacus Positional Embeddings:**
  - Implementación 100% vectorizada en `trdm/layers/abacus.py` de rangos posicionales decimales invertidos (unidades=0, decenas=1, centenas=2) para tokens numéricos ASCII.
* **Hierarchical Byte Patcher ($P=4$):**
  - Implementación en `trdm/layers/byte_patcher.py` que agrupa 4 bytes por parche para reducir la complejidad cuadrática de atención por un factor de $16\times$.
* **Currículo Staged Warmup:**
  - Mecanismo en `train.py` (anteriormente `train_v6.py`) con `sched_p = 1.0` en épocas tempranas hacia `0.70` en fases avanzadas.
* **Evaluación Confirmatoria Oficial de 5 Semillas:**
  - Resultados oficiales en `eval_confirmatorio_5semillas_results.json` (etiqueta histórica `v6_confirmatorio_5semillas_results.json`):
    - Full Exact Match: $5.12\% \pm 4.88\%$ (máximo de $13.33\%$, rompiendo la nulidad histórica).
    - Exact Match Scratchpad ($P_w, P_{ms}$): $81.88\% \pm 10.14\%$ (máximo de $95.00\%$).
    - Validez Sintáctica: $100.00\% \pm 0.00\%$.
    - Exactitud en dígitos aislados: $81.97\% \pm 3.90\%$.

---

## [0.2.0-v5] — Fase 1 Pre-registrada — 2026-11-16
**Etiqueta Git:** `prereg-v1`  
**Commit base:** `f379cb4` / `d0a2b8b`

### Añadido
* **Paged BitCOS v1:**
  - Serialización binaria alineada a 4096 bytes del kernel de Linux con lectura directa vía `mmap` (`trdm/serialization/paged_bitcos.py`).
  - Formatos `fast` (2 bits/peso con LUT) y `compact` (presencia + signo).
* **Kernel C Fusionado Multi-Página:**
  - Implementación en `paged_gemm_kernel.c` con vectorización SIMD (AVX2) y paralelismo OpenMP sobre páginas binarias directamente en memoria mapeada.
* **MoE Consciente de Memoria Física:**
  - Algoritmo de enrutamiento en `trdm/modules/paged_moe.py` con penalización de latencia:
    $$S(e) = U(e) - \lambda C_{\text{lectura}}(e) - \mu C_{\text{cómputo}}(e)$$
  - Implementación de *buddy routing* ante cuellos de botella de disco.
* **Context Manager de 3 Niveles:**
  - Jerarquía desacoplada en `trdm/core/context_manager.py` (RAM activa, latente comprimido y persistencia en SQLite).
* **Protocolo Pre-registrado Oficial:**
  - Especificación formal e inmutable de experimentos E0 a E6 (`Protocolo pre-registrado — Basic-Lite (TRDM-v5), Fase 1.md`).
  - Ejecución confirmatoria inicial de 5 semillas en Fase 1 (`archive/experiments_v4_v5/e1_confirmatorio_5semillas_results.json`).

---

## [0.1.0 - 0.1.4] — Versiones Exploratorias (TRDM-v1 a TRDM-v4) — 2026-11-14 a 2026-11-15
* Exploración de cuantización ternaria pura W1.58A8 con Straight-Through Estimator (STE).
* Implementación preliminar de kernel C para operaciones suma/resta sin multiplicadores flotantes (`ternary_kernel.c`).
* Pruebas de difusión discreta por bloques (CDLM).
* Detección de fallas de sobreajuste de longitud fija y diseño del contrato sintáctico de ancho fijo.
