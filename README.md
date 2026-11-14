# TRDM-v3: Ternary Recursive Discrete Diffusion Language Model

**TRDM-v3** es un modelo generativo experimental diseñado y optimizado desde cero para ejecutarse de forma ágil sobre CPU en entornos Linux (**Intel Core i3-1115G4**, 2 núcleos físicos / 4 hilos lógicos, 16 GB RAM).

El sistema integra en una única arquitectura los fundamentos del *Transformer Decoder-only* clásico con los avances más recientes de la literatura en modelos eficientes: **cuantización ternaria consciente del entrenamiento (BitNet b1.58)**, **difusión discreta enmascarada (LLaDA / CDLM)**, **recursión profunda de pesos compartidos (TRM / TRLM)**, **enrutamiento modular por selección de experto (Expert-Choice MoE)** y **parada adaptativa por token (Mixture-of-Recursions)**.

---

## 1. Pilares de la Arquitectura

1. **Cuantización Ternaria Consciente del Entrenamiento (QAT W1.58A8):**
   * Capas lineales [BitLinear](file:///home/Miguel19x/Escritorio/Importante/Programación%20-%20General/Basic-Lite%200.1/model.py) con estimador Straight-Through (STE), normalización `SubLN` previa y cuantización de activaciones de 8 bits por token en $[-127, +127]$.
   * Pesos restringidos a $\{-1, 0, +1\}$ escalados por media absoluta (`absmean`).
   * Empaquetador y cargador binario **BITCOSv2** (mapa de presencia + signos a $2-z$ bits/peso), reduciendo el tamaño del modelo de **3.45 MB a solo 164 KB** (~21x de compresión).

2. **Difusión Discreta Enmascarada Block-Causal (CDLM / iLLaDA):**
   * Atención estructurada por bloques de 16 tokens: causal estricta hacia bloques pasados consolidados y bidireccional dentro del bloque activo para desenmascarar en paralelo.
   * **Caché KV estable:** El prefijo y los bloques previos resueltos se congelan en la caché KV, eliminando la necesidad de recomputar la secuencia completa en cada ronda de difusión.
   * **Atención GQA 2:1:** 4 cabezas de consulta ($Q$) agrupadas sobre 2 cabezas de clave/valor ($K, V$), reduciendo a la mitad el tráfico de memoria RAM hacia la CPU.

3. **Recursión Profunda con Separación de Doble Estado (TRM / TRLM):**
   * Reutilización de un único bloque recurrente compartido `SharedDualStateRecursiveCore` ejecutado $R=4$ veces.
   * **Separación estricta de tres vectores:** Ancla de entrada proyectada $a$, estado latente de razonamiento $z_{r+1} = f_\theta(a, y_r, z_r)$ y propuesta de respuesta tentativa $y_{r+1} = g_\theta(y_r, z_{r+1})$.
   * Truncamiento selectivo de gradiente ($R_{\text{nograd}} = 1$) para acotar la memoria de activaciones.

4. **Enrutamiento Modular Expert-Choice MoE Dinámico:**
   * 4 expertos ternarios con activación $\text{ReLU}^2$ equilibrados por diseño sin requerir pérdidas auxiliares de balanceo.
   * **Capacidad dinámica $C_{\text{cap}}(t)$:** Asigna hasta 1.35x de capacidad a los expertos en etapas finales con pocas máscaras (alta certidumbre ortográfica) y 0.65x en etapas iniciales de alto ruido.

---

## 2. Mecanismos Avanzados Integrados en v3

* **Transporte Latente entre Rondas de Difusión:** La compuerta aprendible `carry_gate` transfiere el estado $(z_R, y_R)$ desconectado del grafo (`detach`) entre pasos de difusión sucesivos, eliminando la amnesia entre rondas.
* **Reconsideración de Tokens (*Low-Confidence Remasking*):** Durante la inferencia, si la confianza conjunta de un token ya propuesto decae al desenmascarar letras vecinas, el modelo lo vuelve a enmascarar a `[MASK]` para autocorregir errores.
* **Condicionamiento Conjunto $(t, \Delta t)$:** La red `noise_mlp` recibe simultáneamente el nivel de ruido actual $t$ y el salto temporal $\Delta t = |t_{\text{prev}} - t_{\text{act}}|$.
* **Corte Temprano por `[EOS]`:** La generación local se detiene en cuanto se consolida el punto final o token de fin de secuencia, evitando generar caracteres residuales en bloques vacíos.
* **Planificador BitNet en Dos Etapas:** Calentamiento lineal durante el 10% inicial de pasos, decaimiento coseno posterior y desactivación del `weight_decay` en el último 20% del entrenamiento para estabilizar los signos ternarios.
* **Inferencia Entera / Suma-Resta Pura:** Motor de inferencia que opera directamente sobre enteros (`int32 = matmul(int8, {-1, 0, +1})`) y módulo C++ nativo con OpenMP/AVX2 ([ternary_cpu_kernel.py](file:///home/Miguel19x/Escritorio/Importante/Programación%20-%20General/Basic-Lite%200.1/ternary_cpu_kernel.py)) que sustituye las multiplicaciones por sumas y restas de activaciones.

---

## 3. Dataset en Lenguaje Natural en Español

En lugar de configuraciones numéricas o sintéticas de videojuegos, el modelo entrena y valida sobre conocimiento general en **lenguaje natural estructurado en pares de pregunta y respuesta**:
* **Geografía:** Capitales y continentes (e.g. *¿Cuál es la capital de Francia? -> París.*).
* **Ciencia y Naturaleza:** Colores, astros, estaciones (e.g. *¿De qué color es el cielo despejado? -> Azul.*).
* **Oficios y Profesiones:** Funciones (e.g. *¿Qué profesional apaga los incendios? -> El bombero.*).
* **Animales y Hábitos:** Sonidos y rasgos (e.g. *¿Qué animal ladra y cuida la casa? -> El perro.*).
* **Antónimos:** Pares opuestos (e.g. *¿Cuál es el opuesto de frío? -> Caliente.*).

Esto permite interpretar con total claridad cómo las letras y palabras son reconstruidas ronda a ronda durante la difusión discreta.

---

## 4. Estructura del Proyecto

* [config.py](file:///home/Miguel19x/Escritorio/Importante/Programación%20-%20General/Basic-Lite%200.1/config.py): Hiperparámetros, dimensiones de tensores, bloques causales y umbrales de parada MoR.
* [model.py](file:///home/Miguel19x/Escritorio/Importante/Programación%20-%20General/Basic-Lite%200.1/model.py): Definición de `BitLinear`, `BlockCausalGQATernaryAttention`, `SharedDualStateRecursiveCore` y empaquetador binario `BITCOSv2`.
* [data.py](file:///home/Miguel19x/Escritorio/Importante/Programación%20-%20General/Basic-Lite%200.1/data.py): Generador de texto procedural `ProceduralNaturalTextDataset` y corrupción estocástica de difusión.
* [train.py](file:///home/Miguel19x/Escritorio/Importante/Programación%20-%20General/Basic-Lite%200.1/train.py): Bucle de entrenamiento con las 4 pérdidas unificadas ($\mathcal{L}_{\text{diff}} + \mathcal{L}_{\text{inter}} + \mathcal{L}_{\text{cons}} + \mathcal{L}_{\text{ponder}}$) y planificador BitNet.
* [generate.py](file:///home/Miguel19x/Escritorio/Importante/Programación%20-%20General/Basic-Lite%200.1/generate.py): Generador Block-Causal con caché KV, re-enmascaramiento y corte temprano por fin de respuesta.
* [benchmark.py](file:///home/Miguel19x/Escritorio/Importante/Programación%20-%20General/Basic-Lite%200.1/benchmark.py): Informe de latencia CPU, memoria RAM pico y comparación de inferencia FP32 vs INT8.
* [ternary_cpu_kernel.py](file:///home/Miguel19x/Escritorio/Importante/Programación%20-%20General/Basic-Lite%200.1/ternary_cpu_kernel.py): Extensión C++ inline (OpenMP/AVX2) para cálculo ternario sin multiplicaciones.
* [requirements.txt](file:///home/Miguel19x/Escritorio/Importante/Programación%20-%20General/Basic-Lite%200.1/requirements.txt): Dependencias optimizadas para CPU (`torch` CPU y `numpy`).

---

## 5. Métricas de Rendimiento en Hardware (Intel Core i3-1115G4)

Valores empíricos medidos sobre Linux en 4 hilos CPU:

| Métrica | Valor Medido | Observaciones |
| :--- | :--- | :--- |
| **Parámetros Únicos Totales** | 883,986 | **87.6%** en capas ternarias `BitLinear` |
| **Tamaño de Pesos en FP32** | 3,453.1 KB (~3.45 MB) | Modelo completo en precisión simple |
| **Tamaño en Archivo BITCOSv2** | **164.4 KB** | Factor de compresión de **21.0x** (1.73 bits/peso) |
| **Paso de Entrenamiento (QAT + 4 Pérdidas)** | ~302 ms | Lote $B=16$, longitud $L=80$, $R=4$ recursiones |
| **Generación Block-Causal (FP32)** | 226.0 ms | Inferencia completa con caché KV |
| **Generación Block-Causal (INT8 BITCOSv2)** | **211.1 ms** | Acumulación entera sin punto flotante |
| **Consumo Máximo de RAM (RSS)** | **~633 MB** | < 4% de los 16 GB de RAM del equipo |
| **Efecto de la Recursión Interna ($r_1 \rightarrow r_4$)** | **0.31 $\rightarrow$ 1.00** | Precisión de tokens tras 4 iteraciones del bloque |
| **Parada Temprana MoR** | 18% tokens congelados | Tokens convergidos no consumen ciclos en $r_4$ |

---

## 6. Guía de Ejecución

Activar el entorno virtual:
```bash
source venv/bin/activate
```

### A. Medición de Hardware y Compresión
```bash
python3 benchmark.py
```

### B. Entrenamiento
```bash
python3 train.py
```

### C. Inferencia y Generación Ronda a Ronda
```bash
python3 generate.py
```
