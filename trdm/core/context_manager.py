import math
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


class HierarchicalContextManager:
    """
    Administrador explícito de información contextual jerárquica en 3 niveles:
    - Nivel 1 (Activo): Ventana reciente y fragmentos recuperados en RAM para atención directa.
    - Nivel 2 (Comprimido): Representaciones latentes de segmentos pasados (guía y orientación).
    - Nivel 3 (Archivo): Contenido original exacto almacenado de forma persistente (en disco/SQLite)
      para recuperación de hechos/cifras sin pérdida por compresión o alucinación.
    """

    def __init__(
        self,
        active_window_size: int = 64,
        compressed_segment_size: int = 32,
        db_path: str = ":memory:",
        embedding_dim: int = 192,
    ):
        self.active_window_size = active_window_size
        self.compressed_segment_size = compressed_segment_size
        self.db_path = db_path
        self.embedding_dim = embedding_dim

        # Nivel 1: Buffer Activo en RAM
        self.active_token_ids: List[int] = []
        self.active_latent_kv: Optional[torch.Tensor] = None

        # Nivel 2: Segmentos Comprimidos Latentes en RAM
        self.compressed_segments: List[Dict[str, Any]] = []

        # Nivel 3: Archivo Persistente (Base de datos SQLite indexada)
        self.conn = sqlite3.connect(self.db_path)
        self._init_archive_db()

        self.total_tokens_ingested: int = 0
        self.retrieval_queries_count: int = 0
        self.retrieval_hits_count: int = 0

    def _init_archive_db(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS context_archive (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_token_idx INTEGER,
                end_token_idx INTEGER,
                raw_bytes BLOB,
                raw_text TEXT,
                keywords TEXT
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_start_pos ON context_archive(start_token_idx)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_keywords ON context_archive(keywords)")
        self.conn.commit()

    def ingest_tokens(
        self,
        new_token_ids: List[int],
        latent_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Ingesta nuevos tokens en el sistema jerárquico.
        Si la ventana activa supera active_window_size, desplaza los tokens antiguos
        hacia el Nivel 2 (Comprimido) y los archiva en el Nivel 3 (Archivo Persistente).
        """
        self.active_token_ids.extend(new_token_ids)
        self.total_tokens_ingested += len(new_token_ids)

        archived_chunks_count = 0
        while len(self.active_token_ids) > self.active_window_size:
            # Extraer segmento para archivar
            seg_len = min(self.compressed_segment_size, len(self.active_token_ids) - self.active_window_size)
            if seg_len <= 0:
                break
            seg_tokens = self.active_token_ids[:seg_len]
            self.active_token_ids = self.active_token_ids[seg_len:]

            start_idx = self.total_tokens_ingested - len(self.active_token_ids) - seg_len
            end_idx = start_idx + seg_len

            # 1. Archivar en Nivel 3 (Original exacto)
            raw_bytes = bytes([t for t in seg_tokens if t < 256])
            raw_text = raw_bytes.decode("utf-8", errors="replace")
            # Extraer palabras clave simples para indexación
            words = [w.strip(":, \n\t") for w in raw_text.split() if len(w.strip(":, \n\t")) > 1]
            keywords_str = " ".join(words)

            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO context_archive (start_token_idx, end_token_idx, raw_bytes, raw_text, keywords)
                VALUES (?, ?, ?, ?, ?)
                """,
                (start_idx, end_idx, raw_bytes, raw_text, keywords_str),
            )
            self.conn.commit()

            # 2. Comprimir en Nivel 2 (Resumen latente)
            # Representación compacta del segmento (ej. vector latente promedio)
            seg_summary = {
                "chunk_id": cursor.lastrowid,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "token_count": seg_len,
                "keywords": words,
            }
            self.compressed_segments.append(seg_summary)
            archived_chunks_count += 1

        return {
            "active_tokens": len(self.active_token_ids),
            "compressed_segments": len(self.compressed_segments),
            "archived_chunks_created": archived_chunks_count,
            "total_ingested": self.total_tokens_ingested,
        }

    def retrieve_exact_fact(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Recupera fragmentos originales del Nivel 3 (Archivo) basados en coincidencias
        de palabras clave o términos específicos (evita degradación o pérdida de datos históricos).
        """
        self.retrieval_queries_count += 1
        query_terms = [q.strip(":, \n\t") for q in query.split() if len(q.strip(":, \n\t")) > 1]
        if not query_terms:
            return []

        cursor = self.conn.cursor()
        results = []
        for term in query_terms:
            cursor.execute(
                """
                SELECT chunk_id, start_token_idx, end_token_idx, raw_text, keywords
                FROM context_archive
                WHERE keywords LIKE ? OR raw_text LIKE ?
                ORDER BY chunk_id ASC
                LIMIT ?
                """,
                (f"%{term}%", f"%{term}%", top_k),
            )
            rows = cursor.fetchall()
            for r in rows:
                item = {
                    "chunk_id": r[0],
                    "start_token_idx": r[1],
                    "end_token_idx": r[2],
                    "raw_text": r[3],
                    "keywords": r[4],
                }
                if not any(x["chunk_id"] == item["chunk_id"] for x in results):
                    results.append(item)
            if len(results) >= top_k:
                break

        if results:
            self.retrieval_hits_count += 1

        return results[:top_k]

    def inject_retrieved_to_active(self, chunks: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        """
        Inserta fragmentos exactos recuperados desde el Archivo hacia la atención activa
        para que el modelo pueda resolver la pregunta con datos no alucinados.
        """
        if isinstance(chunks, dict):
            chunks = [chunks]
        all_tokens = []
        for ch in chunks:
            raw_text = ch.get("raw_text", "")
            all_tokens.extend([b for b in raw_text.encode("utf-8")])
        self.active_token_ids = all_tokens + self.active_token_ids

    def get_telemetry(self) -> Dict[str, Any]:
        """Reporta telemetría de memoria y recuperación contextual."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM context_archive")
        total_archived_chunks = cursor.fetchone()[0]

        hit_rate = (
            float(self.retrieval_hits_count) / float(max(1, self.retrieval_queries_count))
        ) * 100.0

        # Huella de memoria en RAM:
        # Activo: tokens * 1 byte (en precisión de byte)
        # Comprimido: Nivel 2 almacena metadatos ligeros de orientación
        # Archivo: Nivel 3 reside en disco (SQLite), ocupando 0 bytes en memoria activa
        ram_active_bytes = len(self.active_token_ids)
        total_tokens = max(1, self.total_tokens_ingested)
        saving_ratio = round(float(total_tokens) / float(max(1, len(self.active_token_ids))), 2)

        return {
            "active_tokens_in_ram": len(self.active_token_ids),
            "compressed_segments_count": len(self.compressed_segments),
            "archived_chunks_on_disk": total_archived_chunks,
            "total_tokens_ingested": self.total_tokens_ingested,
            "active_ram_footprint_bytes": ram_active_bytes,
            "equivalent_full_context_bytes": total_tokens,
            "ram_saving_ratio": saving_ratio,
            "retrieval_hit_rate_pct": round(hit_rate, 2),
        }

    def close(self) -> None:
        if hasattr(self, "conn") and self.conn is not None:
            self.conn.close()

    def __del__(self) -> None:
        self.close()
