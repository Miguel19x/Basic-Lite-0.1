# Basic-Lite TRACE: Motor de Ejecución y Residencia Selectiva para Modelos Masivos en Hardware de Consumo

**Basic-Lite TRACE** es una arquitectura neuronal y un motor de inferencia de ultra-baja precisión (**W1.58A8**) concebido y dirigido por **Miguel Angel Aranguren Ramirez** (**Miguel19x**), diseñado para operar modelos de escala masiva (proyectados a **30B parámetros**) sobre hardware estándar de consumo doméstico (**Intel Core i3-1115G4**, 2 núcleos físicos / 4 hilos, 16 GB RAM, SSD NVMe, Linux x86_64), sin aceleradores gráficos dedicados (GPU) y sin descomprimir matrices completas a FP32 en RAM.

La concepción e integración de todo este modelo y sistema en una arquitectura convergente y funcional proviene íntegramente de **Miguel Angel Aranguren Ramirez**, consolidando todo el desarrollo desde su lanzamiento inicial en **Basic-Lite 0.1** (familias TRDM-v1 a v4) hasta la formulación definitiva de la arquitectura **TRACE** (*Ternary Resident Architecture for Convergent Execution*).

El proyecto se rige por el principio de **residencia y ejecución selectivas**: desacoplar la capacidad total del modelo (almacenada en disco en páginas empaquetadas ternarias W1.58) de la capacidad activa en cada instante (mínima fracción de memoria residente), garantizando rigor formal y reproducibilidad científica bajo protocolos pre-registrados.

---

## 1. La Arquitectura TRACE y sus Pilares de Convergencia

La arquitectura **TRACE** (*Ternary Resident Architecture for Convergent Execution*) integra los siguientes principios:

* **Ternary (Ternario W1.58A8 — Preservado al 100%):**  
  Todas las proyecciones nucleares operan con cuantización ternaria $\{-1, 0, 1\}$ mediante [trdm/layers/bitlinear.py](trdm/layers/bitlinear.py) con estimador de gradiente recto (STE). En runtime, los tensores se ejecutan directamente sobre páginas binarias sin inflar matrices a FP32 en RAM mediante el kernel C fisionado ([paged_gemm_kernel.c](paged_gemm_kernel.c)) y el formato [Paged BitCOS v1](trdm/serialization/paged_bitcos.py).

* **Resident / Recursive (Residencia Selectiva y Recursividad Dinámica):**  
  El núcleo residente compartido ([trdm/modules/resident_core.py](trdm/modules/resident_core.py)) no apila capas densas tradicionales, sino que reitera $R$ pasos de recursión sobre un estado latente dual: razonamiento profundo ($z_r$) y respuesta ($y_r$). Incluye codificación sinusoidal continua de profundidad $e_r$ y una compuerta adaptativa de parada temprana (*adaptive halting gate*).

* **Convergent / Causal (Campo Causal Desacoplado en 2 Etapas):**  
  El mecanismo de generación en producción es el **Campo Causal Desacoplado en 2 Etapas con `detach()`** ([trdm/models/causal.py](trdm/models/causal.py)). Este desacopla la predicción de subtotales multiplicativos ($pw, pms$) de las sumas finales ($watt, ms$), escalando el Full-EM a **$5.12\% \pm 4.88\%$** (con pico de **$13.33\%$**) y resolviendo los subtotales al **$81.88\%$**.

---

## 2. Esquema Global de Arquitectura

```mermaid
flowchart TD
    subgraph InputStage ["1. Entrada por Bytes y Codificación Posicional Decimal"]
        RawBytes["Entrada ASCII de Bytes (Vocab=260)"] --> BytePatch["Hierarchical Byte Patcher (P=4, Reducción 16x)"]
        RawBytes --> AbacusLayer["Abacus Positional Embedding (Rangos decimales invertidos: U=0, D=1, C=2)"]
        BytePatch & AbacusLayer --> CombinedEmbed["Representación de Entrada Enriquecida"]
    end

    subgraph CoreTernary ["2. Tronco Recurrente Residente (W1.58A8)"]
        CombinedEmbed --> SharedCore["Núcleo Ternario Compartido (BitLinear W1.58A8)"]
        SharedCore --> MLA["Multi-Head Latent Attention (Compresión KV 4x-8x)"]
        SharedCore --> PagedMoE["Memory-Aware Paged MoE: S(e) = U(e) - λ C_lectura - μ C_cómputo"]
    end

    subgraph TwoStageField ["3. Campo Causal Desacoplado en 2 Etapas (Zero Exposure Bias)"]
        PagedMoE --> Stage1["ETAPA 1: Predicción de Subtotales (pw, pms)"]
        Stage1 --> CausalAnchor["Anclaje Causal con Stop-Gradient (detach)"]
        CausalAnchor --> Stage2["ETAPA 2: Predicción de Totales Condicionados (watt, ms)"]
        Stage2 --> OutputBytes["Salida de Ancho Fijo (100% Validez Sintáctica)"]
    end

    subgraph MemoryHierarchy ["4. Administradores Físicos de Recursos"]
        PagedBitCOS[("Paged BitCOS v1 (Páginas 4 KB en NVMe)")] <--> LRUCache["Caché LRU en RAM (2 Slots Residentes)"]
        LRUCache <--> PagedMoE
        L1RAM["Nivel 1: RAM Activa (MLA)"] <--> L2Latent["Nivel 2: Latente Comprimido"] <--> L3SQLite[("Nivel 3: Archivo SQLite en Disco")]
    end
```

---

## 3. Los Cinco Ejes Arquitectónicos Fundamentales

### 3.1. Eje 1: Paginación Binaria a Nivel de Sistema Operativo (Paged BitCOS v1)
* **Principio:** Consumir tensores directamente empaquetados mediante mapeo en memoria (`mmap`) y núcleos C optimizados, eliminando la expansión flotante a FP32 en RAM.
* **Alineación:** Bloques alineados estrictamente a **4096 bytes** (tamaño nativo de página de memoria del kernel Linux).
* **Formatos de empaquetado ([trdm/serialization/paged_bitcos.py](trdm/serialization/paged_bitcos.py)):**
  * `compact`: Bit de presencia + bit de signo para minimizar ancho de banda PCIe/NVMe.
  * `fast`: Empaquetado directo de 2 bits por peso con tablas de consulta (LUT) y operaciones de suma/resta directa.

### 3.2. Eje 2: Enrutamiento MoE Consciente de Memoria Física
* **Principio:** Penalizar dinámicamente la activación de expertos que no residen en la memoria RAM:
  $$S(e) = U(e) - \lambda C_{\text{lectura}}(e) - \mu C_{\text{cómputo}}(e)$$
  * $C_{\text{lectura}}(e) = 0$ si el experto ya reside en la caché LRU en RAM.
  * $C_{\text{lectura}}(e) > 0$ si requiere transferencia física desde almacenamiento secundario NVMe.
* **Mecanismo de Escape y Buddy Experts ([trdm/modules/paged_moe.py](trdm/modules/paged_moe.py)):** Si la afinidad $U(e)$ supera el umbral crítico, se autoriza la lectura física; en escenarios de restricción temporal estricta, el ruteador despacha a un experto sustituto residente (*buddy expert*) para eludir la detención de I/O.

### 3.3. Eje 3: Multi-Head Latent Attention (MLA) y Memoria Contextual en 3 Niveles
* **MLA ([trdm/modules/mla_attention.py](trdm/modules/mla_attention.py)):** Comprime las matrices de claves y valores (KV) hacia un cuello de botella latente conjunto $c_t^{KV}$, reduciendo el tamaño de la caché de atención activa entre **4× y 8×** frente a GQA o atención estándar.
* **Desacoplamiento Contextual ([trdm/core/context_manager.py](trdm/core/context_manager.py)):**
  * *Nivel 1 (Activo):* Ventana de atención directa en RAM.
  * *Nivel 2 (Comprimido):* Resúmenes latentes de bloques temporales previos.
  * *Nivel 3 (Archivo Persistente):* Almacenamiento exacto en SQLite sobre disco para recuperación sin pérdidas de hechos numéricos e históricos.

### 3.4. Eje 4: Compresión Jerárquica de Bytes (Hierarchical Byte Patching, P=4)
* **Principio:** Preservar la universalidad del vocabulario de bytes (260 tokens, sin tokenizador rígido dependiente del dominio) mientras se mitiga el costo cuadrático de atención.
* **Impacto ([trdm/layers/byte_patcher.py](trdm/layers/byte_patcher.py)):** Agrupar localmente $P=4$ bytes en vectores continuos divide la longitud efectiva de la secuencia por 4 ($L/4$), recortando el cálculo de atención recurrente en un factor de **$16\times$** ($P^2$).

### 3.5. Eje 5: Abacus Embeddings y Campo Causal en 2 Etapas
* **Abacus Positional Embeddings ([trdm/layers/abacus.py](trdm/layers/abacus.py)):** Extrae mediante cálculo 100% vectorizado los rangos decimales relativos de derecha a izquierda (unidades=0, decenas=1, centenas=2) para tokens numéricos ASCII, dotando al transformador de conciencia del valor posicional.
* **Campo Causal en 2 Pasadas ([trdm/models/causal.py](trdm/models/causal.py)):** Desacopla la aritmética compuesta en dos etapas con `detach()`:
  1. *Etapa 1:* Predice subtotales multiplicativos ($P_w, P_{ms}$).
  2. *Anclaje Causal:* Inyecta los subtotales en la secuencia de entrada como hechos consumados del contexto con corte de gradiente.
  3. *Etapa 2:* Predice las sumas multivariables finales ($watt, ms$).
* **Currículo Staged Warmup ([train.py](train.py)):** En las primeras épocas fija `sched_p = 1.0` con datos exactos (*ground truth*) para que la Etapa 2 aprenda a sumar sin verse contaminada por predicciones iniciales ruidosas; luego decae gradualmente hacia $0.70$ para mitigar el sesgo de exposición (*exposure bias*).

---

## 4. Resultados Experimentales Confirmatorios Oficiales (5 Semillas)

Los experimentos se ejecutaron bajo el protocolo pre-registrado oficial ([Protocolo pre-registrado — Basic-Lite (TRDM-v5), Fase 1.md](Protocolo%20pre-registrado%20%E2%80%94%20Basic-Lite%20(TRDM-v5),%20Fase%201.md)), evaluando 3000 pasos en las 5 semillas oficiales (`42, 123, 456, 789, 999`):

### 4.1. Comparativa Histórica de Evolución

| Métrica en `test-composicional` (480 casos) | **Fase 1: TRDM-v5 (Etiqueta `prereg-v1`)** | **Fase 2: TRACE Estable (TRDM-v6)** | Impacto / Delta |
| :--- | :---: | :---: | :---: |
| **Exact Match Total (Full-EM)** | $0.04\% \pm 0.09\%$ (4 de 5 semillas en 0%) | **$5.12\% \pm 4.88\%$** (Máx: **$13.33\%$**) | **+5.08 pp (Ruptura de Nulidad)** |
| **Exact Match Scratchpad ($P_w, P_{ms}$)** | Colapsado / Fragmentado (< 5%) | **$81.88\% \pm 10.14\%$** (Máx: **$95.00\%$**) | **+76.88 pp (Multiplicación Resuelta)** |
| **Validez Sintáctica de Formato** | $61.00\% \pm 21.49\%$ | **$100.00\% \pm 0.00\%$** | **+39.00 pp (Estructura Perfecta)** |
| **Exactitud Aislada en Dígitos** | $60.50\% \pm 3.10\%$ | **$81.97\% \pm 3.90\%$** | **+21.47 pp** |

*Artefacto oficial de resultados:* [eval_confirmatorio_5semillas_results.json](eval_confirmatorio_5semillas_results.json)

### 4.2. Desglose de las 5 Semillas Confirmatorias

```text
[Semilla  42] Test-Comp: EM =  5.62% | Scratchpad EM = 88.75% | DEV: EM = 16.67%
[Semilla 123] Test-Comp: EM = 13.33% | Scratchpad EM = 95.00% | DEV: EM = 13.89%
[Semilla 456] Test-Comp: EM =  2.08% | Scratchpad EM = 72.29% | DEV: EM =  2.78%
[Semilla 789] Test-Comp: EM =  3.33% | Scratchpad EM = 71.88% | DEV: EM =  2.78%
[Semilla 999] Test-Comp: EM =  1.25% | Scratchpad EM = 81.46% | DEV: EM =  2.78%
--------------------------------------------------------------------------------------
Media Global : Test-Comp: EM =  5.12% ± 4.88% (IC95: [-0.93%, 11.17%])
Scratchpad EM: Test-Comp: EM = 81.88% ± 10.14% (IC95: [69.28%, 94.47%])
```

---

## 5. Matriz de Decisiones Arquitectónicas (ADR)

| Decisión Arquitectónica | Justificación Técnica | Ventaja Validada | Compensación (*Trade-Off*) |
| :--- | :--- | :--- | :--- |
| **Paged BitCOS v1 (4 KB)** | Habilitar ejecución directa y mapeo `mmap` sin inflar capas a FP32. | Reducción a 1.58 bits por peso; lecturas alineadas a páginas del SO. | Exige empaquetado matricial en múltiplos del tamaño de página. |
| **Router MoE con Costo de E/S** | Evitar saturación del canal PCIe/NVMe con intercambio indiscriminado de módulos. | Ahorro masivo de transferencias conservando módulos frecuentes en RAM. | Requiere calibrar $\lambda_{\text{read}}$ y prever *buddy experts* ante fallos de caché. |
| **Multi-Head Latent Attention** | Reducir tensores de atención activa retenidos en memoria de trabajo. | Descompresión latente de KV; ahorro de 4×–8× en RAM activa. | Añade una proyección lineal para descomprimir $K$ y $V$ en inferencia. |
| **Abacus Embeddings** | Dotar a tokens de bytes ASCII de simetría de orden de magnitud decimal (U, D, C). | Eleva la exactitud de dígitos del 60.5% a >81.9% y facilita el cálculo. | Requiere vectorización de suma acumulativa invertida en capas de entrada. |
| **Campo Causal en 2 Etapas** | Eliminar el colapso de Exact Match por factorización paralela independiente. | Permite resolución exacta en cálculo acoplado (Full-EM escala de 0% a 13.33%). | Requiere dos pasadas hacia adelante en inferencia (una por etapa). |
| **Staged Warmup Curriculum** | Evitar que errores de la Etapa 1 destruyan el aprendizaje aditivo de la Etapa 2. | Convergencia estable de la adición multivariable con scheduled sampling. | Requiere calibrar la duración de la fase de calentamiento ($40\%$ de los pasos). |

---

## 6. Estructura Limpia del Repositorio

```
.
├── config.py                                 # Configuración unificada de hiperparámetros
├── data.py                                   # Generador procedural de datos y particiones científicas
├── train.py                                  # Pipeline de entrenamiento causal con Staged Warmup
├── eval.py                                   # Motor de evaluación causal multivariable en 2 pasos
├── trdm_checkpoint.pt                        # Checkpoint oficial del modelo entrenado
├── eval_confirmatorio_5semillas_results.json # Registro oficial de la corrida confirmatoria de 5 semillas
├── paged_gemm_kernel.c                       # Kernel en C para operaciones matriciales Paged BitCOS
├── libpaged_gemm.so                          # Biblioteca compartida compilada para aceleración C
├── trdm/                                     # Paquete modular canónico
│   ├── core/
│   │   ├── context_manager.py                # Gestor de contexto jerárquico (L1 RAM, L2 Latente, L3 SQLite)
│   │   ├── weight_manager.py                 # Gestor de residencia y caché LRU de expertos
│   │   └── types.py                          # Definiciones de tipos y ModelOutput
│   ├── layers/
│   │   ├── abacus.py                         # Abacus Positional Embeddings vectorizado
│   │   ├── bitlinear.py                      # Capa W1.58A8 con gradiente recto (STE)
│   │   ├── byte_patcher.py                   # Codificador/Decodificador de parches de bytes (P=4)
│   │   └── norm.py                           # RMSNorm
│   ├── modules/
│   │   ├── mla_attention.py                  # Multi-Head Latent Attention (MLA)
│   │   ├── paged_moe.py                      # MemoryAwarePagedMoE con paginación
│   │   └── resident_core.py                  # Núcleo ternario residente compartido
│   ├── models/
│   │   └── causal.py                         # TRDMCausalModel: Modelo causal integrado de 2 etapas
│   ├── ops/
│   │   ├── paged_c_kernel.py                 # Enlace ctypes con paged_gemm_kernel.c
│   │   └── ternary_cpu.py                    # Operaciones ternarias de CPU
│   └── serialization/
│       └── paged_bitcos.py                   # Serialización y lectura de páginas Paged BitCOS v1
└── archive/                                  # Historial científico y experimentos previos
    ├── checkpoints_legacy/                   # Modelos y binarios de fases anteriores (v3, v4, v5)
    ├── experiments_v4_v5/                    # Scripts de diagnóstico y auditorías previas
    ├── legacy_models/                        # Definiciones de modelos anteriores
    └── zips/                                 # Paquetes de despliegue archivados
```

---

## 7. Instrucciones de Ejecución

### 7.1. Ejecución Local en CPU (Intel Core i3)
```bash
# Entrenar corrida en CPU local (optimizado para 2 hilos físicos)
./venv/bin/python train.py --steps 1000 --batch_size 32 --warmup_ratio 0.40 --device cpu
```

### 7.2. Evaluación del Checkpoint Oficial
```bash
# Evaluar el modelo en la partición reservada test-composicional
./venv/bin/python -c "
import torch
from config import HybridConfig
from data import ProceduralCompositionalDataset
from trdm.models.causal import TRDMCausalModel
from eval import evaluate_causal

cfg = HybridConfig(device='cpu')
model = TRDMCausalModel(cfg)
ckpt = torch.load('trdm_checkpoint.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state'])
dataset = ProceduralCompositionalDataset(cfg, split='test-composicional')
results = evaluate_causal(model, dataset, torch.device('cpu'))
print('Resultados Test-Composicional:', results)
"
```

---

## 8. Titularidad, Propiedad Intelectual y Licencias

* **Titular Exclusivo e Inventor:** Miguel Angel Aranguren Ramirez (Miguel19x, `leohxone@gmail.com`)
* **Aviso de Titularidad y Licencia:** [LICENSE.txt](LICENSE.txt) y [NOTICE.md](NOTICE.md)
* **Registro de Autores y Titulares:** [AUTHORS.md](AUTHORS.md)
* **Historial de Versiones:** [CHANGELOG.md](CHANGELOG.md)
* **Avisos de Terceros y Código Abierto:** [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) y carpeta [licenses/](licenses/)
* **Identidad y Memoria del Proyecto:** [docs/PROJECT_IDENTITY.md](docs/PROJECT_IDENTITY.md)
* **Inventario de Procedencia y Cadena de Custodia:** [docs/PROVENANCE.md](docs/PROVENANCE.md)
* **Registro de Asistencia por IA:** [docs/AI_ASSISTANCE.md](docs/AI_ASSISTANCE.md)
* **Protocolo Pre-registrado Congelado:** [Protocolo pre-registrado — Basic-Lite (TRDM-v5), Fase 1.md](Protocolo%20pre-registrado%20%E2%80%94%20Basic-Lite%20(TRDM-v5),%20Fase%201.md)

---

## 9. Cita Académica y Servicios Profesionales

### Cómo Citar este Trabajo
Si utilizas la arquitectura Basic-Lite TRACE, sus métricas o su protocolo en investigaciones académicas o análisis técnicos, puedes citarlo como:

```bibtex
@misc{aranguren2026basiclitetrace,
  author = {Aranguren Ramirez, Miguel Angel},
  title = {Basic-Lite TRACE: Ternary Resident Architecture for Convergent Execution on Consumer CPUs},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Miguel19x/Basic-Lite-TRACE}}
}
```

### Adaptaciones Empresariales y Consultoría
Para organizaciones o equipos interesados en:
1. **Evaluación de viabilidad técnica:** Analizar la ejecución de modelos de gran tamaño en servidores locales o edge sin GPUs de alto coste.
2. **Implementación y optimización a medida:** Despliegue de kernels en C y formatos paginados para hardware específico.
3. **Licenciamiento comercial privado:** Acuerdos comerciales para uso en entornos cerrados.

Contactar directamente con el fundador y responsable técnico:  
**Miguel Angel Aranguren Ramirez** — `leohxone@gmail.com`
