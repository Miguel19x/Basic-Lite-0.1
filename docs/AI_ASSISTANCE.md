# Basic-Lite TRACE — Declaración de Asistencia de Inteligencia Artificial (AI_ASSISTANCE)

**Proyecto:** Basic-Lite TRACE (Arquitectura TRACE)  
**Autor y Director del Proyecto:** Miguel Angel Aranguren Ramirez (Miguel19x)  
**Fecha:** 17 de noviembre de 2026  
**Propósito del documento:** Transparencia técnica sobre las herramientas utilizadas en el proceso de desarrollo y delimitación de la autoría humana.

---

## 1. Principio de Transparencia y Dirección Humana

El proyecto **Basic-Lite TRACE** fue concebido, estructurado y dirigido por **Miguel Angel Aranguren Ramirez**. 

Durante las fases de implementación, análisis y redacción, se utilizaron diversos asistentes de inteligencia artificial como **herramientas de apoyo al desarrollo de software** (análogas a linters, autocompletadores avanzados, generadores de esqueletos de código y motores de búsqueda semántica).

> [!NOTE]
> **Aclaración Importante sobre Validación:**  
> La utilización de modelos de IA no constituye una auditoría externa independiente, certificación pericial ni respaldo institucional por parte de empresas como Google o Perplexity. La validez de Basic-Lite descansa exclusivamente en su **código ejecutable**, su **protocolo experimental pre-registrado** y sus **resultados empíricos reproducibles** sobre hardware físico.

---

## 2. Herramientas Empleadas y Ámbitos de Uso

| Herramienta / Plataforma | Proveedor | Tipo de Interacción |
| :--- | :--- | :--- |
| **Gemini (Flash / Pro) en Antigravity IDE** | Google / Google DeepMind | Asistencia de codificación interactiva: sugerencias de vectorización en PyTorch, generación de código base (*boilerplate*), depuración de scripts y organización de archivos. |
| **Perplexity AI** | Perplexity AI | Búsqueda documental y recopilación de referencias normativas públicas sobre licencias de código abierto y pautas de la OMPI. |

---

## 3. Delimitación de Roles y Contribuciones

### 3.1. Lo que Aportaron las Herramientas de IA
* Sugerencias de optimización para bucles en C con directivas OpenMP / AVX2.
* Generación de plantillas sintácticas de clases en Python y funciones auxiliares de parseo de texto.
* Resúmenes comparativos de literatura pública sobre modelos de lenguaje y directrices legales.

### 3.2. Lo que Diseñó, Decidió y Validó Miguel Angel Aranguren Ramirez
* **La Hipótesis Central:** La idea original de desacoplar capacidad en disco y memoria RAM mediante residencia selectiva para hardware de consumo básico (CPU Intel Core i3-1115G4).
* **Las Decisiones Arquitectónicas:** La formulación matemática de la función de ruteo MoE con penalización por I/O físico ($S(e) = U(e) - \lambda C_{\text{lectura}} - \mu C_{\text{cómputo}}$), el diseño del campo causal en 2 etapas con `detach()`, y la integración de *Abacus Embeddings*.
* **El Control Experimental:** La redacción y aprobación del protocolo pre-registrado congelado, la selección de hiperparámetros y la exigencia de evaluar 5 semillas independientes sin inflar métricas con plantillas estáticas.
* **Inspección y Corrección Manual:** Detección de fallos en la indexación recursiva, corrección de supuestos espurios y comprobación directa de la ejecución en CPU.

---

## 4. Condiciones de Uso y Atribución

De conformidad con los términos de servicio de los proveedores consultados:
* Este proyecto no se presenta como un producto oficial ni avalado por Google o Perplexity.
* El contenido técnico y las decisiones de diseño reflejan el criterio del autor humano.
* El código fuente propio y los artefactos originales se ofrecen bajo los términos de [LICENSE.txt](LICENSE.txt).
