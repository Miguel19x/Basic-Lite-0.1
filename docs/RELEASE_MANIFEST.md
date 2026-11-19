# Basic-Lite TRACE — Manifiesto de Versión y Criptografía (RELEASE_MANIFEST)

**Proyecto:** Basic-Lite TRACE  
**Versión Documentada:** TRACE 1.0.0 (Consolidación unificada de arquitectura TRACE)  
**Fecha de corte:** 17 de noviembre de 2026  
**Titular y Custodio:** Miguel19x (`miguelaranguren19x@outlook.com`)  

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
| `README.md` | `ed88f0dfa3a2f5c269f9d070167a5f7f374b700ba699a68e9b47d922a46294b1` |
| `LICENSE.txt` | `7c4d8c21fddee9f087e23349cbeec3d8d513f37aaf54957f5c21ab8dfb3d2609` |
| `CONTRIBUTING.md` | `ed625c617b50377005202033c3b2d96a1c9e37b886904b029f340ffb206a8ff2` |
| `NOTICE.md` | `cff498beb420200e7237b98792229f9acdf3b0c5ebe935c7cad51bba5836351b` |
| `AUTHORS.md` | `2c7a099203dfdc56a25bb1eea549bce129600039d53ce9ff52b3ee24dc39d84c` |
| `CHANGELOG.md` | `e3f9c1faf99c545f2763e84e99fc5204558f4805e8930ba6aca9b5b3d9f68fbb` |
| `THIRD_PARTY_NOTICES.md` | `0aa1c14c11e32027a936a366616dafea6139ef02a619d19c861503784fc238c9` |
| `config.py` | `ce58c08ca9b35b7d1ff6239fbd0aa92c8e728d627d922b2474ca3cd7afacaf1f` |
| `train.py` | `b3089c25c0db74d77a76976f4beb8b1d0ce5dbfe3525c8ec0315ef7b4871b486` |
| `eval.py` | `7625cc61c9e685eecb48741f1d65bccfd732c5cbb0fc844700a2652edc39ca1b` |
| `data.py` | `b569cb139842dc340d40115f974ae4a3c88bf85461a5021e7fd1120912438c91` |
| `paged_gemm_kernel.c` | `94b3f28a477adbc47d9fa551a140213a3a445e5ca548682aadbc33d98e01e444` |
| `results/eval_confirmatorio_5semillas_results.json` | `bf7c8890a1b46dfa7db89d3a6647e1cd64c0f1594d540569f058dad3c1dcc446` |
| `results/e0_higiene_results.json` | `91c1fc2c55bff3f51dd6632db27f8e537eb9691c8376a5b2d23b42d87b0709dc` |
| `results/e1_confirmatorio_5semillas_results.json` | `4729c36ee1a0b2bd2505c7d3cf66409b75eb97b63f90291a63d5e7431f9d9d36` |
| `results/e2_aislamiento_causal_results.json` | `31c8cdbac8ab62ccad849fe2f27d03a6bd0a315792e2ad1e7cd3295e412ae587` |
| `results/e4_caracterizacion_nvme_results.json` | `cc473e1a22671efe7768ced1d4357ddf40a400bc2a60b399c81c77e94efd87dc` |
| `trace/__init__.py` | `7885f433847794294d1799bedeb752fb8c41d21fea8f334378878dc7a2bcc71c` |
| `trace/core/context_manager.py` | `63b4a45827760db192b88a3f4bc71100967d6afd1a50ac93273fb61e2a3329b7` |
| `trace/core/weight_manager.py` | `7bc9761e416c6bb02d2e7c75e7020894a5a9eaaefe4f619001a3f465e814c8a1` |
| `trace/layers/abacus.py` | `418e1b1d8335106cce276d50616abf451373805bea844bc5ade4df4f02425959` |
| `trace/layers/bitlinear.py` | `8e87951fa1763428250306fa0714c40f8f68aa1ae215590eb0f510033cfe20e9` |
| `trace/layers/byte_patcher.py` | `ec307159e1d89a1f2ffd2fa65051dac2fd5abf483bf078f0dfb71bdbd3533641` |
| `trace/models/causal.py` | `37a6dd44ff6c8f8954017dd9629a09cda124d0bddab4846c8c0bbfcfe898de2a` |
| `trace/modules/paged_moe.py` | `15e86f36680d3203ea32251e67d52a2c0d2ecbafac1afc83b055ebe1e22ddd9a` |
| `trace/modules/mla_attention.py` | `c94eedf5999c6aa13f787d8ae24be8f7bff3ed0b557a61b1c0dbd70e53843c78` |
| `trace/modules/resident_core.py` | `5ce5835dafeca357995b03342e086674008c47e6e00b995fd6f2583ff514a247` |
| `trace/serialization/paged_bitcos.py` | `4f3ce2820a0c8c42c0445cdc870502042d3deea6fab3a84b3ff2e0d30070be1c` |
| `docs/PAPER_TECNICO_BASIC_LITE.md` | `3b26c2353741bb216118099162fff1a077ffc7df9154bf9592b944e3ae90c617` |
| `docs/PROJECT_IDENTITY.md` | `4cc4138fabd8b8dc9ee85d8006f93b5a0c1ee538500d99ca1626161b79b65bcd` |
| `docs/AI_ASSISTANCE.md` | `15072cdece67e7830fbe0c0bf61395bbcb5f653e5cdd7be8ddddd4fe729ead28` |
| `docs/PROVENANCE.md` | `fc563b48a81669ae3ba86f8b26d1009877478b58055fcf50184712d5ad12a766` |

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
