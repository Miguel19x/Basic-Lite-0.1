# Basic-Lite TRACE — Inventario de Procedencia y Trazabilidad (PROVENANCE)

Este documento contiene el inventario exhaustivo de procedencia de cada archivo, módulo, kernel, dataset y artefacto del proyecto **Basic-Lite TRACE**, conforme a las directrices de la OMPI para el aseguramiento de la cadena de autoría y custodia previa al registro de propiedad intelectual.

---

## 1. Tabla General de Trazabilidad y Procedencia

| Componente / Módulo | Procedencia / Autor | Versión / Commit Base | Licencia o Permiso | Modificaciones Propias Realizadas | Estado Legal Interno |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`trdm/serialization/paged_bitcos.py`** | Creación original (Miguel19x) | Commit inicial v5 (`f379cb4`) | Propiedad exclusiva de Miguel19x | Código 100% original. Serialización y alineación a 4 KB para kernel Linux. | Verificado / Propio |
| **`paged_gemm_kernel.c`** | Creación original (Miguel19x) | Commit inicial v5 (`f379cb4`) | Propiedad exclusiva de Miguel19x | Código 100% original en C con AVX2 y OpenMP para inferencia directa mmap. | Verificado / Propio |
| **`ternary_kernel.c`** | Creación original (Miguel19x) | Basic-Lite 0.1 (`14 Nov 2026`) | Propiedad exclusiva de Miguel19x | Kernel C SIMD de suma/resta sin multiplicadores flotantes. | Verificado / Propio |
| **`trdm/layers/abacus.py`** | Creación original (Miguel19x) | Commit v6 (`b105127`) | Propiedad exclusiva de Miguel19x | Formulación vectorizada original de rangos decimales invertidos para números ASCII. | Verificado / Propio |
| **`trdm/layers/byte_patcher.py`** | Creación original (Miguel19x) | Commit v6 (`b105127`) | Propiedad exclusiva de Miguel19x | Implementación original de agrupamiento de bytes $P=4$ con compresión lineal. | Verificado / Propio |
| **`trdm/models/causal.py`** (histórico `trdm_v6_causal.py`) | Creación original (Miguel19x) | Commit v6 (`b105127`) | Propiedad exclusiva de Miguel19x | Arquitectura causal desacoplada en 2 etapas con anclaje `detach()`. | Verificado / Propio |
| **`trdm/modules/paged_moe.py`** | Creación original (Miguel19x) | Commit v5 (`f379cb4`) | Propiedad exclusiva de Miguel19x | Algoritmo original de ruteo consciente de E/S física NVMe y buddy routing. | Verificado / Propio |
| **`trdm/core/context_manager.py`** | Creación original (Miguel19x) | Commit v5 (`f379cb4`) | Propiedad exclusiva de Miguel19x | Arquitectura original de memoria jerárquica en 3 niveles (RAM, latente, SQLite). | Verificado / Propio |
| **`trdm/layers/bitlinear.py`** | Adaptación conceptual (Miguel19x) | Basado en fórmula pública de BitNet (Wang et al., 2024) | Implementación propia (sin código copiado de Microsoft) | Reescritura propia desde cero en PyTorch utilizando `weight.sign()` y STE. | Verificado / Propio |
| **`trdm/layers/rotary.py`** | Adaptación conceptual (Miguel19x) | Basado en Su et al. (RoPE) | Implementación propia | Implementación adaptada para secuencias de parches de bytes. | Verificado / Propio |
| **`trdm/layers/norm.py`** | Adaptación conceptual (Miguel19x) | Basado en Zhang & Sennrich (RMSNorm) | Implementación propia | RMSNorm estándar implementado en PyTorch con epsilon configurable. | Verificado / Propio |
| **`trdm/modules/mla_attention.py`** | Adaptación conceptual (Miguel19x) | Basado en concepto de DeepSeek-V2 | Implementación propia | Implementación propia de compresión KV latente adaptada a hardware de CPU. | Verificado / Propio |
| **`data.py`** (histórico `data_v2.py`) | Creación original (Miguel19x) | Commit v5 (`e8895f0`) | Propiedad exclusiva de Miguel19x | Generador sintético determinista de 2880 combinaciones y contrato de ancho fijo. | Verificado / Propio |
| **`train.py`** (histórico `train_v6.py`) | Creación original (Miguel19x) | Commit v6 (`29ff4ca`) | Propiedad exclusiva de Miguel19x | Pipeline de entrenamiento con currículo Staged Warmup y logging de métricas. | Verificado / Propio |
| **`eval.py`** (histórico `eval_v6.py`) | Creación original (Miguel19x) | Commit v6 (`113ee26`) | Propiedad exclusiva de Miguel19x | Suite de evaluación con desglose composicional, acarreos y contrastes pareados. | Verificado / Propio |
| **`archive/experiments_v4_v5/`** | Histórico experimental (Miguel19x) | Fase 1 y 2 (Nov 2026) | Propiedad exclusiva de Miguel19x | Suite de scripts experimentales, diagnósticos de ablación y registros de auditoría conservados. | Verificado / Propio |
| **`legacy/basic-lite-0.1/`** | Repositorio originario fundacional (Miguel Angel Aranguren Ramirez) | Commits `aa7ce9e` a `49f64ca` (14-15 Nov 2026) | Propiedad exclusiva de Miguel Angel Aranguren Ramirez | Versión 0.1 completa (50 archivos fuente: TRDM-v3/v4, `ternary_kernel.c`, datasets y diagnósticos). Centralizado físicamente con historial Git. | Verificado / Propio |
| **PyTorch (`torch`)** | PyTorch Foundation / Meta | v2.0+ (instalado vía pip) | BSD-3-Clause | Ninguna (dependencia externa de runtime). | Aprobado / Terceros |
| **NumPy (`numpy`)** | NumPy Developers | v1.24+ (instalado vía pip) | BSD-3-Clause | Ninguna (dependencia externa de runtime). | Aprobado / Terceros |
| **SQLite3 (`sqlite3`)** | SQLite Development Team | Estándar de Python 3 / C | Dominio Público | Ninguna (dependencia de almacenamiento de sistema). | Aprobado / Terceros |

---

## 2. Declaración de No Inclusión de Código Ajeno No Atribuido

* Se certifica que no se ha copiado código privado o protegido por derechos de autor de terceros sin la debida licencia o permiso.
* Los fragmentos que adaptan formulaciones teóricas de la literatura pública (BitNet, RoPE, RMSNorm, MLA) han sido reimplementados de forma independiente por Miguel19x sin reutilización de código fuente no licenciado.
* Ningún componente externo ha sido marcado indebidamente como propio.
