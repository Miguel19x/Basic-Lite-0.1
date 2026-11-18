# Basic-Lite — Avisos de Componentes de Terceros (Third Party Notices)

Este proyecto hace uso de librerías y componentes de software de código abierto bajo sus respectivas licencias. Este documento compila las atribuciones y términos exigidos por dichas licencias de conformidad con los estándares de buenas prácticas de la OMPI.

---

## 1. Inventario de Dependencias Externas

| Componente | Proyecto / Procedencia | Tipo de Licencia | Uso en Basic-Lite | Archivo de Licencia |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch** | PyTorch Foundation / Meta AI / The Linux Foundation | BSD-3-Clause | Tensores base, autograd, capas neuronales y wrappers | `licenses/LICENSE_PYTORCH.txt` |
| **NumPy** | NumPy Developers | BSD-3-Clause | Operaciones numéricas, generación de arrays y evaluación | `licenses/LICENSE_NUMPY.txt` |
| **SQLite3** | SQLite Consortium (D. Richard Hipp) | Dominio Público / SQLite Blessing | Persistencia en disco del Nivel 3 del Context Manager | `licenses/LICENSE_SQLITE.txt` |
| **Linux POSIX / mmap** | Linux Kernel Organization / POSIX standards | GPLv2 (Kernel) / Interfaz POSIX de usuario (Libc) | Mapeo directo a páginas de memoria del OS en Paged BitCOS | Headers del sistema operativo |
| **OpenMP** | OpenMP Architecture Review Board | Runtime de compilador GCC/Clang | Paralelismo multinúcleo en kernels C optimizados | Headers del sistema |

---

## 2. Textos Completos de Licencias de Terceros

Los textos completos de las licencias de código abierto se conservan íntegramente en la carpeta `licenses/`:

1. **PyTorch License:** [licenses/LICENSE_PYTORCH.txt](licenses/LICENSE_PYTORCH.txt)
2. **NumPy License:** [licenses/LICENSE_NUMPY.txt](licenses/LICENSE_NUMPY.txt)
3. **SQLite License & Blessing:** [licenses/LICENSE_SQLITE.txt](licenses/LICENSE_SQLITE.txt)

---

## 3. Declaración de Independencia y No Afectación

* Los componentes de terceros se rigen estrictamente por sus términos de distribución originales.
* El aviso de reserva de derechos y protección legal de **Basic-Lite** contenido en `LICENSE.txt` y `NOTICE.md` aplica **únicamente** a los módulos, algoritmos, kernels y arquitecturas de autoría original desarrollados por Miguel19x, y no pretende en modo alguno atribuirse titularidad sobre librerías o sistemas operativos de terceros.
