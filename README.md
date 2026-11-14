# Basic-Lite 0.1 (TRDM-v3): Ternary Recursive Discrete Diffusion Language Model

**Basic-Lite 0.1** (arquitectura **TRDM-v3**) es un modelo de lenguaje experimental y científicamente fundado, diseñado desde cero para ejecutarse de forma ágil y eficiente en arquitecturas de CPU estándar de consumo (**Intel Core i3-1115G4**, 2 núcleos físicos / 4 hilos lógicos, 16 GB RAM) bajo Linux.

El proyecto fusiona cinco pilares de la investigación contemporánea en eficiencia:
1. **Cuantización ternaria consciente del entrenamiento (BitNet b1.58 / W1.58A8)** con empaquetado binario **BITCOSv2**.
2. **Difusión discreta enmascarada por bloques causales (LLaDA / CDLM)** con reutilización estricta de caché KV.
3. **Recursión profunda de pesos compartidos (TRM / TRLM)** con transporte de memoria latente $(z_R, y_R)$ mediante compuerta residual adaptativa (`carry_gate`).
4. **Enrutamiento modular Expert-Choice intra-bloque** con capacidad dinámica $C_{\text{cap}}(t)$ dependiente del ruido y exclusión estricta de tokens de relleno (`PAD`).
5. **Parada adaptativa por token (Mixture-of-Recursions - MoR)**.

---

## 1. Pilares de la Arquitectura

### 1.1. Cuantización Ternaria Consciente del Entrenamiento (W1.58A8 + BITCOSv2)
* Capas lineales `BitLinear` con estimador de gradiente directo (*Straight-Through Estimator* - STE).
* Activaciones cuantizadas a 8 bits con signo en el rango $[-127, +127]$ tras normalización previa `SubLN`.
* Pesos restringidos al alfabeto $\{-1, 0, +1\}$ escalados por su media absoluta (`absmean`).
* **Empaquetado BITCOSv2:** Almacenamiento binario compacto a $2-z$ bits/peso (mapa de presencia + signos). Las 14 matrices ternarias del modelo ocupan únicamente **~164 KB** en disco (~1.73 bpw).

### 1.2. Difusión Discreta Enmascarada Block-Causal (LLaDA / CDLM)
* Secuencias organizadas en bloques de 16 tokens exactos ($B_{\text{blk}} = 16$).
* **Atención Block-Causal:** Atención causal estricta hacia bloques pasados consolidados y bidireccional no causal dentro del bloque activo para reconstruir texto en paralelo.
* **Caché KV Inmutable:** Los bloques prefijados del prompt y los bloques resueltos se congelan en la caché KV, eliminando recomputaciones redundantes.
* **Agrupación GQA 2:1:** 4 cabezas de consulta ($Q$) agrupadas sobre 2 cabezas de clave/valor ($K, V$), reduciendo un 50% el consumo de ancho de banda a la memoria.

### 1.3. Recursión Profunda con Separación de Doble Estado (TRM / TRLM)
* Reutilización de un único bloque recurrente compartido `SharedDualStateRecursiveCore` ejecutado $R$ veces por paso.
* Separación estricta de tres componentes latentes:
  * Ancla proyectada: $a = \text{Linear}(x_0)$
  * Estado de razonamiento latente: $z_{r+1} = f_\theta(a, y_r, z_r)$
  * Propuesta de respuesta tentativa: $y_{r+1} = g_\theta(y_r, z_{r+1})$
* **Transporte Latente:** La compuerta aprendible `carry_gate` transfiere $(z_R, y_R)$ desconectado del grafo (`detach`) entre pasos de difusión sucesivos, eliminando la amnesia entre rondas.
* **Condicionamiento Conjunto $(t, \Delta t)$:** Modulación adaptativa según el nivel de ruido actual $t$ y el intervalo temporal $\Delta t = |t_{\text{prev}} - t_{\text{act}}|$.

### 1.4. Enrutamiento Modular Expert-Choice MoE Intra-Bloque
* 4 expertos ternarios con activación $\text{ReLU}^2$.
* **Causalidad Estricta Intra-Bloque:** La selección top-$k$ de tokens se restringe rígidamente al interior de cada bloque de 16 tokens. Esto garantiza que ningún token futuro filtre información hacia bloques pasados ni invalide la caché KV.
* **Exclusión de PAD:** Los tokens de relleno (`PAD`) son enmascarados antes del cálculo de selección, impidiendo que consuman capacidad de cómputo en los expertos.
* **Capacidad Dinámica $C_{\text{cap}}(t)$:** Modula dinámicamente la capacidad de los expertos entre $0.65\times$ en fases ruidosas y $1.35\times$ en fases finales de alta certidumbre.

---

## 2. Generalización Composicional Fuera de Distribución (OOD)

Para evitar que el modelo actúe como una mera tabla de memorización, la evaluación se realiza sobre un entorno procedural estructurado con 4 ejes ortogonales independientes:
* **Nodo (6):** `server`, `switch`, `storage`, `compute`, `sensor`, `router`
* **Perfil de Potencia (5):** `eco`, `base`, `turbo`, `edge`, `batch`
* **Nivel de Redundancia (4):** `simple`, `dual`, `triple`, `mesh`
* **Carga de Trabajo (4):** `baja`, `media`, `alta`, `pico`

Total del espacio de estados: $6 \times 5 \times 4 \times 4 = 480$ combinaciones posibles.
* **Entrenamiento:** 384 combinaciones (80%).
* **Held-Out (OOD):** 96 combinaciones (20%) rigurosamente reservadas que el modelo **jamás observa en entrenamiento**.

### Supervisión de Longitud Estilo LLaDA
El modelo aprende la predicción autónoma de longitud mediante supervisión de la posición de fin de secuencia `[EOS]` seguida de tokens `[PAD]`. Esto elimina cualquier regla heurística de corte en inferencia.

---

## 3. Verificación de la Hipótesis H1 (Cómputo en Inferencia)

La **Hipótesis H1** postula que un modelo recurrente con pesos compartidos entrenado con bajo cómputo ($R_{\text{train}} = 2$) puede mejorar su capacidad de resolución y precisión sobre combinaciones no vistas simplemente incrementando el número de recursiones en inferencia ($R_{\text{infer}} \in \{1, 2, 4, 8\}$) sin ningún reentrenamiento ni ajuste de parámetros.

---

## 4. Estabilidad y Balance Real de Parámetros

## 4. Métricas Empíricas de Rendimiento en Linux CPU (Intel Core i3-1115G4)

Valores empíricos medidos sobre 4 hilos CPU con precisión int8 y punto flotante:

| Métrica | Valor Medido | Observaciones |
| :--- | :--- | :--- |
| **Parámetros Únicos Totales** | **869,014** | **89.1%** alojados en capas ternarias `BitLinear` |
| **Parámetros Ternarios (14 matrices)** | 774,144 | Cuantizados en $\{-1, 0, +1\}$ |
| **Tamaño BITCOSv2 en Disco** | **166.2 KB** | **1.752 bpw** (empaquetado presencia + signo) |
| **Parámetros FP32 Restantes** | 94,870 (~370.6 KB) | Embeddings, RMSNorms, `carry_gate`, router MoE, MLP |
| **Tamaño Desplegable Total** | **536.8 KB** | **6.3× de reducción real** frente a los 3,394.6 KB en FP32 |
| **Atención GQA (4:2)** | Ahorro 50% en KV | Menor tráfico de memoria RAM |
| **Paso de Entrenamiento (QAT + STE)** | **199.1 ms** | Lote $B=16$, longitud $L=80$, 2 pasos encadenados |
| **Generación Block-Causal (FP32)** | 135.9 ms | 11 evaluaciones de bloque local con caché KV |
| **Generación Block-Causal (INT8 BITCOSv2)** | **84.1 ms** | **1.6× más rápido** que FP32 con acumulación entera |
| **Consumo Máximo de RAM (RSS)** | **495.5 MB** | < 3.2% de los 16 GB de RAM del equipo |
| **Precisión en Datos Reservados (Held-Out)** | **86.0%** | Precisión de token en tuplas OOD jamás vistas |

---

## 5. Guía de Ejecución

### Requisitos y Entorno
```bash
# Activar el entorno virtual optimizado para CPU
source venv/bin/activate
```

### A. Medición de Rendimiento y Memoria (Benchmark)
```bash
python3 benchmark.py
```

### B. Entrenamiento con Difusión Encadenada de 2 Pasos
```bash
python3 train.py
```

### C. Inferencia y Evaluación de la Hipótesis H1
```bash
python3 generate.py
```
