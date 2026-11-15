# Basic-Lite 0.1 (TRDM-v4): Ternary Recursive Discrete Diffusion Language Model

**Basic-Lite 0.1** (arquitectura **TRDM-v4**) es un modelo de lenguaje experimental rigurosamente auditado y fundado en principios de eficiencia extrema para hardware estándar de consumo (**Intel Core i3-1115G4**, 2 núcleos físicos / 4 hilos lógicos, 16 GB RAM) bajo Linux.

El proyecto fusiona cinco ejes clave de la investigación contemporánea en modelos compactos:
1. **Cuantización Ternaria Consciente del Entrenamiento (W1.58A8 / BitNet b1.58)** con serialización empaquetada **BITCOSv2**.
2. **Difusión Discreta Enmascarada Block-Causal (CDLM / LLaDA)** con reutilización estricta de caché KV inmutable.
3. **Recursión Profunda de Pesos Compartidos (TRM / TRLM)** con incrustación sinusoidal continua $\text{PE}(u_r)$ y transferencia latente $(z_R, y_R)$ mediante `carry_gate`.
4. **Enrutamiento Modular Expert-Choice Intra-Bloque** con exclusión estricta de tokens de relleno (`PAD`).
5. **Parada Adaptativa por Token (Mixture-of-Recursions - MoR)**.

---

## 1. Auditoría Estadística del Dataset y Deconstrucción del Techo Trivial (85.5%)

En iteraciones iniciales, se reportaban métricas de *"precisión de tokens"* superiores al 86% sobre secuencias fuera de distribución (OOD). La auditoría estadística formal en Python puro ([auditoria_dataset.py](file:///home/Miguel19x/Escritorio/Importante/Programación%20-%20General/Basic-Lite%200.1/auditoria_dataset.py)) demostró de manera concluyente que dicho porcentaje era un **espejismo estadístico**:

```
=== Auditoría Estadística del Dataset (Python Puro) ===
Combinaciones Totales : 480 (Train: 384 [80.0%] | Held-Out: 96 [20.0%])
Posiciones supervisadas por muestra (media): 28.2 bytes
  - Plantilla fija ('watt: ', 'ms: ', '\n') : 42.5%
  - Tokens [PAD]                            : 39.4%
  - Token [EOS]                             : 3.5%
  - Dígitos reales a calcular ('0'..'9')    : 14.5% (apenas 4 bytes por muestra)
Techo Trivial (acertar solo plantilla+PAD)  : 85.52%
Techo Mayoría (plantilla + dígito frecuente): 88.59%
Techo Dígitos Clase Mayoritaria             : 21.17%
```

### Hallazgo Metodológico Clave
* De las 28.2 posiciones promedio supervisadas en cada respuesta, el **85.52%** está compuesto enteramente por sintaxis estática (`watt: `, `ms: `, saltos de línea) y relleno (`PAD`).
* Un modelo que no compute ninguna operación y memorice únicamente la plantilla fija obtiene un **85.5% de precisión global**.
* Predecir el dígito más frecuente en las posiciones numéricas eleva la precisión global al **88.59%**.
* **Corrección Implementada en TRDM-v4:** Se eliminó la evaluación por token global y se adoptaron métricas estrictas:
  * **Exact Match (EM) Numérico:** Acierto exacto del valor entero predicho para `watt`, `ms` y la tupla completa (`both_em`).
  * **Exactitud Aislada de Dígitos (`DigAcc`):** Medición de precisión restringida exclusivamente a las posiciones numéricas (bytes ASCII `48..57`).
  * **Ponderación de Gradientes en Entrenamiento:** Factor $\times 6.0$ en bytes de dígitos, $\times 1.0$ en plantilla/EOS y $\times 0.2$ en PAD para contrarrestar la dilución de gradiente.

---

## 2. Novedades y Correcciones Arquitectónicas en TRDM-v4

### 2.1. Codificación Sinusoidal Continua de Profundidad ($\text{PE}(u_r)$)
En versiones previas, se utilizaba una tabla discreta `nn.Embedding(num_recursions + 8)`. Al entrenar con recursión fija $R_{\text{train}} = 2$, las filas $r \ge 2$ permanecían en su inicialización aleatoria $\mathcal{N}(0, 1)$, contaminando la inferencia en profundidades $R \in \{4, 8\}$.

En **TRDM-v4**, la tabla de embeddings se reemplazó por un codificador sinusoidal continuo sobre la coordenada normalizada:
$$u_r = \frac{r+1}{R} \in (0, 1]$$
Junto con escalas residuales compartidas $\alpha_z, \beta_y$ normalizadas por $\frac{2}{R+1}$, garantizando que toda profundidad $R_{\text{infer}} \in \{1, 2, 4, 8\}$ evalúe únicamente parámetros regularizados y entrenados al 100%.

### 2.2. Enrutamiento Expert-Choice Intra-Bloque con Exclusión de PAD
* La selección top-$k$ de tokens en el router MoE se confina rígidamente dentro de cada bloque de 16 tokens ($B_{\text{blk}} = 16$), impidiendo cualquier fuga causal hacia bloques futuros o invalidación de la caché KV.
* Los tokens `PAD` se enmascaran antes del cálculo del router, evitando que consuman los cupos de capacidad de los expertos ternarios.

### 2.3. Caché de Pesos Efectivos ($W_{\text{eff}}$) en Inferencia
Para evitar penalizar artificialmente la latencia en benchmarks de CPU recalculando continuamente la cuantización durante el bucle de generación, las capas `BitLinear` cachean la matriz $W_{\text{eff}} = \gamma \cdot \text{RoundClip}(W / \gamma)$ al cambiar a modo `eval()`.

---

## 3. Protocolo de Evaluación Experimental Multi-Semilla (H1)

Para someter a prueba la **Hipótesis H1** (*"un modelo recurrente con pesos compartidos entrenado con bajo cómputo puede mejorar su capacidad de resolución en combinaciones no vistas simplemente incrementando el número de recursiones en inferencia"*), se ejecutó un protocolo riguroso:
* **Semillas Evaluadas:** 3 semillas pseudoaleatorias independientes (`42`, `1337`, `2026`).
* **Pasos de Entrenamiento:** 800 pasos por semilla (lote $B=16$, longitud $L=80$).
* **Baseline Autorregresivo:** Modelo Transformer Causal en FP32 (`baseline_ar.py`), entrenado durante 1,600 pasos ($2\times$) para igualar exactamente las 2 pasadas por paso del modelo de difusión.
* **Muestra de Evaluación:** Partición fija de 384 tuplas de entrenamiento y **96 tuplas Held-Out (20.0%)** jamás vistas, evaluadas en las 3 semillas ($N = 288$ evaluaciones OOD por cada profundidad).
* **Análisis Estadístico:** Prueba pareada exacta de McNemar frente a $R=2$ con corrección de Bonferroni ($\alpha = 0.05 / 3 = 0.0167$).

### Tabla Completa de Resultados Empíricos (Media ± Desv. Estándar)

| Modelo / Régimen | Prof. Inferencia ($R$) | Train Full-EM (%) | OOD Full-EM (%) | OOD Watt-EM (%) | OOD MS-EM (%) | OOD DigAcc (%) | OOD Validez Fmt (%) | Latencia CPU (ms) | McNemar $p$ (vs $R=2$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Techo Trivial Mayoría** | — | — | 0.0% | 0.0% | 0.0% | **21.17%** | — | — | — |
| **TRDM-v4 (`fixed_R2`)** | $R = 1$ | — | 0.0% ± 0.0% | 0.0% ± 0.0% | 0.0% ± 0.0% | 11.22% ± 2.55% | 2.08% ± 2.08% | 38.3 ms | 1.0000 (n.s.) |
| **TRDM-v4 (`fixed_R2`)** | $R = 2$ (*train*) | 0.0% ± 0.0% | 0.0% ± 0.0% | 0.0% ± 0.0% | 0.35% ± 0.60% | 13.95% ± 2.80% | 2.43% ± 4.21% | 85.1 ms | Ref. |
| **TRDM-v4 (`fixed_R2`)** | $R = 4$ | — | 0.0% ± 0.0% | 0.0% ± 0.0% | 0.0% ± 0.0% | 14.12% ± 3.82% | 7.99% ± 3.94% | 164.7 ms | 1.0000 (n.s.) |
| **TRDM-v4 (`fixed_R2`)** | $R = 8$ | — | 0.0% ± 0.0% | 0.0% ± 0.0% | 0.69% ± 1.20% | 15.31% ± 3.10% | 9.38% ± 5.51% | 325.9 ms | 1.0000 (n.s.) |
| **TRDM-v4 (`random_R123`)** | $R = 1$ | — | 0.0% ± 0.0% | 0.0% ± 0.0% | 0.35% ± 0.60% | 11.90% ± 4.68% | 1.04% ± 1.04% | 39.7 ms | 1.0000 (n.s.) |
| **TRDM-v4 (`random_R123`)** | $R = 2$ | 0.0% ± 0.0% | 0.0% ± 0.0% | 0.0% ± 0.0% | 0.0% ± 0.0% | 11.82% ± 3.65% | 3.47% ± 1.59% | 76.1 ms | Ref. |
| **TRDM-v4 (`random_R123`)** | $R = 4$ | — | 0.0% ± 0.0% | 0.0% ± 0.0% | 0.0% ± 0.0% | 14.12% ± 1.66% | 6.94% ± 1.59% | 149.0 ms | 1.0000 (n.s.) |
| **TRDM-v4 (`random_R123`)** | $R = 8$ | — | 0.0% ± 0.0% | 0.0% ± 0.0% | 0.0% ± 0.0% | 14.71% ± 1.28% | 4.86% ± 2.17% | 288.2 ms | 1.0000 (n.s.) |
| **Baseline AR Denso (FP32)** | 1 (Autorr.) | **15.28% ± 3.94%** | 0.0% ± 0.0% | 0.69% ± 0.60% | 4.86% ± 1.59% | **20.75% ± 1.93%** | **94.44% ± 2.62%** | **35.3 ms** | — |

*Datos brutos completos exportados en [multi_seed_h1_results.json](file:///home/Miguel19x/Escritorio/Importante/Programación%20-%20General/Basic-Lite%200.1/multi_seed_h1_results.json).*

---

## 4. Análisis Científico y Conclusiones

### 4.1. Veredicto sobre la Hipótesis H1
* **La Hipótesis H1 no se sostiene empíricamente para el razonamiento aritmético composicional:** En todas las configuraciones, el Exact Match numérico completo sobre el conjunto reservado Held-Out se mantuvo en **0.0%** ($p = 1.0000$ en la prueba de McNemar vs $R=2$).
* **Dinámica de la Recursión Latente:** Incrementar $R$ de 1 a 8 muestra una tendencia positiva monótona en la exactitud aislada de dígitos ($11.22\% \to 15.31\%$) y en la validez del formato sintáctico ($2.08\% \to 9.38\%$). No obstante, este incremento permanece **por debajo del techo del predictor trivial de mayoría (21.17%)**.
* **Principio Demostrado:** La escalabilidad de cómputo en inferencia (*test-time scaling*) es capaz de refinar la coherencia de las representaciones latentes, pero **no puede sintetizar reglas lógicas ni operaciones aritméticas que el modelo no consolidó durante la fase de entrenamiento**.

### 4.2. Comparativa frente al Baseline Autorregresivo Denso
* **Memorización Aritmética:** El modelo autorregresivo denso en FP32 (con 1,600 pasos) logró memorizar parcialmente las relaciones no lineales del conjunto de entrenamiento (**15.28% Full Exact Match**, **48.28% DigAcc** en Train), mientras que el modelo de difusión ternaria con QAT a 800 pasos no logró resolver la tarea aritmética en Train (0.0% Full-EM, 21.07% DigAcc, equivalente a la clase mayoritaria).
* **Latencia y Rendimiento en CPU:**
  * El baseline autorregresivo denso en FP32 genera secuencias completas en **35.3 ms** en CPU.
  * El modelo de difusión recursiva requiere **85.1 ms** a $R=2$ y **325.9 ms** a $R=8$, debido a la necesidad de múltiples pasos de refinamiento bidireccional por bloque.
* **Huella en Disco y Memoria:**
  * El modelo TRDM-v4 empaquetado con BITCOSv2 ocupa **673.2 KiB** en disco (166.2 KiB ternario + 507.0 KiB FP32), representando una **reducción de 5.25×** frente a los 3,531.0 KiB del modelo en FP32 completo.
  * El consumo de memoria RAM en ejecución se mantiene en **~331 MiB**, haciéndolo completamente apto para dispositivos embebidos o CPUs de baja potencia.

---

## 5. Estructura del Repositorio y Reproducibilidad

```
.
├── auditoria_dataset.py       # Auditoría estadística en Python puro (partición 384/96 y techos triviales)
├── baseline_ar.py             # Baseline autorregresivo denso FP32 calibrado a igual cómputo (1600 pasos)
├── benchmark.py               # Medición rigurosa de parámetros, tamaños KiB, latencia y memoria RSS
├── config.py                  # Parámetros arquitectónicos, pesos de pérdida y configuración
├── data.py                    # Generador procedimental con partición verificada mediante assert
├── generate.py                # Generación block-causal con evaluación estricta de Exact Match
├── model.py                   # TRDM-v4: BitLinear con W_eff en eval(), PE(u_r) continuo y Expert-Choice
├── multi_seed_h1_results.json # Datos empíricos brutos de la corrida multi-semilla (3 semillas, N=288)
├── run_multi_seed_h1.py       # Automatización del protocolo multi-semilla y test de McNemar
├── ternary_cpu_kernel.py      # Kernel nativo C compilado para operaciones ternarias
└── train.py                   # Entrenamiento con pérdida reponderada y difusión encadenada de 2 pasos
```

### Comandos de Ejecución

1. **Verificar la partición y el techo trivial (sin dependencias, < 1 seg):**
```bash
python3 auditoria_dataset.py
```

2. **Ejecutar el benchmark de memoria y latencia:**
```bash
python3 benchmark.py
```

3. **Entrenar el modelo TRDM-v4:**
```bash
python3 train.py
```

4. **Evaluar inferencia y Exact Match:**
```bash
python3 generate.py
```

5. **Entrenar y evaluar el baseline autorregresivo:**
```bash
python3 baseline_ar.py
```

6. **Reproducir el experimento multi-semilla completo (H1):**
```bash
python3 run_multi_seed_h1.py --seeds 42 1337 2026 --steps 800
```
