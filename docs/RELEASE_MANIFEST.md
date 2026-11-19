# Basic-Lite TRACE — Manifiesto de Versión y Criptografía (RELEASE_MANIFEST)

**Proyecto:** Basic-Lite TRACE  
**Versión Documentada:** TRACE 1.0.0 (Consolidación unificada de arquitectura TRACE)  
**Fecha de corte:** 17 de noviembre de 2026  
**Titular y Custodio:** Miguel Angel Aranguren Ramirez (Miguel19x, `leohxone@gmail.com`)  

> [!NOTE]
> Este manifiesto sirve para identificar fehacientemente la composición e integridad de los archivos del paquete en la fecha especificada mediante huellas criptográficas SHA-256. No constituye por sí solo una certificación legal o notarial de prioridad, sino una prueba técnica inmutable de existencia previa e integridad (*prior art / proof of non-alteration*).

---

## 1. Entorno Físico y de Ejecución Documentado

* **Procesador (CPU):** Intel Core i3-1115G4 @ 3.00 GHz base (2 núcleos físicos / 4 hilos lógicos, 6 MB SmartCache L3).
* **Memoria RAM:** 16 GB DDR4 @ 3200 MT/s.
* **Almacenamiento secundario:** SSD NVMe PCIe Gen4 x4 (MSI M450 500GB).
* **Sistema Operativo:** Linux x86_64 (Kernel 6.x).
* **Herramientas de compilación y runtime:** GCC 13+ con soporte OpenMP y AVX2, Python 3.11+, PyTorch 2.x (CPU-only build).

---

## 2. Hashes Criptográficos SHA-256 de Archivos Fuente y Artefactos Clave (Versión Consolidada v0.2)

Los siguientes hashes han sido generados mediante el algoritmo estándar SHA-256 sobre el contenido exacto de los archivos consolidados de la versión 0.2:

| Archivo Fuente / Artefacto | Hash Criptográfico SHA-256 |
| :--- | :--- |
| `README.md` | `6fb1fd0f960d27d0467fce2730c8bdb0f2f57c5303f7b7411dd5d8e29b6a9211` |
| `LICENSE.txt` | `a18f00146344d2a731354a1c6c1e1b43b1484343fda934a1b05e96c455ed8c68` |
| `CONTRIBUTING.md` | `f04a85ecb69bd61d233baaab84302f9a93cf222d661484b526a59a65ae14728a` |
| `NOTICE.md` | `099669a45546a4696c6f912d0974a42d9ab28f76dc535b56c51904b7f4bb550f` |
| `AUTHORS.md` | `0de798123e10ad9b77c5947572f89f124f8d8e811a7990776ab67dbfd21563d3` |
| `CHANGELOG.md` | `84d2d2ffc14e59f29b36ad75dc93992c0175cfc3aee036cec7170aa0f759d046` |
| `THIRD_PARTY_NOTICES.md` | `e44e6224e7fb5a96696c68df730b5fdd6904b30fa685a553f42754abf8211f4e` |
| `config.py` | `ce58c08ca9b35b7d1ff6239fbd0aa92c8e728d627d922b2474ca3cd7afacaf1f` |
| `train.py` | `b3089c25c0db74d77a76976f4beb8b1d0ce5dbfe3525c8ec0315ef7b4871b486` |
| `eval.py` | `7625cc61c9e685eecb48741f1d65bccfd732c5cbb0fc844700a2652edc39ca1b` |
| `data.py` | `b569cb139842dc340d40115f974ae4a3c88bf85461a5021e7fd1120912438c91` |
| `paged_gemm_kernel.c` | `94b3f28a477adbc47d9fa551a140213a3a445e5ca548682aadbc33d98e01e444` |
| `eval_confirmatorio_5semillas_results.json` | `bf7c8890a1b46dfa7db89d3a6647e1cd64c0f1594d540569f058dad3c1dcc446` |
| `trdm/__init__.py` | `39720b1af438f835e83f318db535ee7d5e4858a78e2df438b36deb9330f082b8` |
| `trdm/core/context_manager.py` | `4959c437ef0b9b3a5ce56486ea199989f75444b40d7d6a62f7ca95bab8e8e491` |
| `trdm/core/weight_manager.py` | `7bc9761e416c6bb02d2e7c75e7020894a5a9eaaefe4f619001a3f465e814c8a1` |
| `trdm/layers/abacus.py` | `bc6b526527292982b1996ce44120d8f179ee75a9f5e36113deec0336fc25833d` |
| `trdm/layers/bitlinear.py` | `76094781948d3770157a27434293a3b1620a5ed841a7f41b70b928dce9727e76` |
| `trdm/layers/byte_patcher.py` | `d25173948d64bfbd33a00fb031e35292a8dbe825763a2d62c9e063c864c95892` |
| `trdm/models/causal.py` | `14df7eaf2424512eb09b19404562be6070906429681de8a24640775260b995d4` |
| `trdm/modules/paged_moe.py` | `758a369c6ba2a8c7f97f9a73b0fec2823d737a2b4daf3ae306a3c4ee25c0edec` |
| `trdm/modules/mla_attention.py` | `cb675fb6e3ecf76ce5901d62df21cc02253feda0703f9e63cee8a7b35ebe41b8` |
| `trdm/modules/resident_core.py` | `f3a596b7a99de9e005d999525a6ae9dbc5eb1b4896d7b388bfdf84530a9075da` |
| `trdm/serialization/paged_bitcos.py` | `0f9f1579d0a9edb025dcdc61c5b5eeaa07b3839b2d3f1a01291d0f69c17ef11b` |
| `docs/PAPER_TECNICO_BASIC_LITE.md` | `b412d4ebe24141d0d05d214f837efe74e7d45777cf1cc0f2fc6833f545eebf39` |
| `docs/PROJECT_IDENTITY.md` | `0fdb8c09b716ded7ae0d0e7647db3b3a11fb9e5257552db263f7c8fc9c5ee92a` |
| `docs/AI_ASSISTANCE.md` | `82ed7f1ee0a0983d5ce4570af24345018e7dd2b772107fb5dac2d327212364f0` |
| `docs/PROVENANCE.md` | `4b382a987bb5e0ebbf3c2e0f1da6ac6c9170b5ec1d6e3a034e79cdd7e73655b2` |

---

## 3. Hashes Criptográficos Históricos de Referencia (Etiqueta congelada `phase2-v6-stable`: `c874140` / `113ee26`)

| Archivo Fuente Histórico | Hash Criptográfico SHA-256 |
| :--- | :--- |
| `train_v6.py` | `5e1ee09f7fd62994834de4c0a35473e7e6a08ac72f07397683480879149a5ee0` |
| `eval_v6.py` | `943f57638174a12c1ccddec07d9deeedea82a68bdf41ad2dad9234577647676e` |
| `eval_composition_audit.py` | `6e13bfa2aaca9ccdd82d7101469ff916520454c9d370bf8fdaa6e0402828a18e` |
| `data_v2.py` | `2b2acd2f2118a280344bc6627bb002f0e0956551f1b5d0787da970f327a45a41` |
| `trdm/models/trdm_v6_causal.py` | `1a8686cff07075ff143c4c5f584ebdefd148560b4ce2201c454d9f32009f1e28` |
| `Protocolo pre-registrado — Basic-Lite (TRDM-v5), Fase 1.md` | `60668166afc9c904e25553ba29c3f8b91892153da301069e76ab32e485cd0f39` |
| `v6_confirmatorio_5semillas_results.json` | `bf7c8890a1b46dfa7db89d3a6647e1cd64c0f1594d540569f058dad3c1dcc446` |

---

## 3. Pruebas Realizadas y Resultados Vinculantes Asociados

1. **Higiene de Medición (E0):** Verificación de causalidad temporal, aislamiento de muestras del lote y consistencia de parsers numéricos (`e0_higiene_results.json`).
2. **Evaluación Multi-Semilla Fase 1 (E1):** Evaluación pre-registrada con 5 semillas fijadas (`42, 123, 456, 789, 999`) reportada en `e1_confirmatorio_5semillas_results.json`.
3. **Aislamiento Causal (E2):** Matriz 2x2 para desacoplar decodificación iterativa frente a capacidad representacional (`e2_aislamiento_causal_results.json`).
4. **Caracterización Física NVMe (E4):** Caracterización de latencia y throughput de E/S en disco SSD NVMe (`e4_caracterizacion_nvme_results.json`).
5. **Evaluación Confirmatoria Fase 2 (TRDM-v6):** 3000 pasos en 5 semillas sobre 480 casos composicionales reportados en `eval_confirmatorio_5semillas_results.json` (etiqueta histórica `v6_confirmatorio_5semillas_results.json`, Full-EM $5.12\% \pm 4.88\%$, Scratchpad EM $81.88\% \pm 10.14\%$, Validez sintáctica $100\%$).

---

## 4. Instrucción de Verificación de Integridad

Para verificar la integridad criptográfica de cualquier archivo en una auditoría o peritaje técnico:
```bash
sha256sum <ruta_del_archivo>
```
La salida debe coincidir unívocamente con el valor registrado en la tabla anterior.
