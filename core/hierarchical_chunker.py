"""Hierarchical Parent-Child Chunking and SQLite Document Store.

Solves the context-vs-granularity dilemma:
- Small Child Chunks (~250-300 tokens) are indexed in ChromaDB for high vector recall.
- Large Parent Chunks (~1200-1500 tokens) are stored in SQLite and injected into LLM context upon retrieval.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ParentChunk:
    id: str
    content: str
    metadata: dict[str, Any]


class ParentDocStore:
    """Thread-safe SQLite-backed key-value store for parent document passages."""

    def __init__(self, db_path: str | Path = "data/docstore.sqlite3") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS parent_chunks (
                    id TEXT PRIMARY KEY,
                    source TEXT,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_parent_source ON parent_chunks (source)")
            conn.commit()

    def save_parents(self, parents: Sequence[ParentChunk]) -> None:
        now = time.time()
        rows = [
            (
                p.id,
                str(p.metadata.get("source", "")),
                p.content,
                json.dumps(p.metadata, default=str),
                now,
            )
            for p in parents
        ]
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO parent_chunks (id, source, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def get_parent(self, parent_id: str) -> ParentChunk | None:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT id, content, metadata_json FROM parent_chunks WHERE id = ?", (parent_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return ParentChunk(
                id=row["id"],
                content=row["content"],
                metadata=json.loads(row["metadata_json"]),
            )

    def delete_by_source(self, source: str) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM parent_chunks WHERE source = ?", (source,))
            conn.commit()
            return cursor.rowcount


class HierarchicalChunker:
    """Generates linked Parent Chunks (for context) and Child Chunks (for embedding similarity)."""

    def __init__(
        self,
        parent_chunk_size: int = 1500,
        parent_overlap: int = 150,
        child_chunk_size: int = 300,
        child_overlap: int = 50,
    ) -> None:
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split_text_hierarchical(
        self,
        text: str,
        base_metadata: dict[str, Any],
    ) -> tuple[list[ParentChunk], list[Document]]:
        clean_text = text.strip()
        if not clean_text:
            return [], []

        source = str(base_metadata.get("source", "unknown"))
        page = str(base_metadata.get("page", "1"))

        raw_parents = self.parent_splitter.split_text(clean_text)
        parent_chunks: list[ParentChunk] = []
        child_documents: list[Document] = []

        for p_idx, p_text in enumerate(raw_parents, start=1):
            p_clean = p_text.strip()
            if not p_clean:
                continue

            p_hash = hashlib.sha1(p_clean.encode("utf-8")).hexdigest()[:8]
            parent_id = f"{source}:{page}:p{p_idx}:{p_hash}"

            p_meta = dict(base_metadata)
            p_meta["parent_id"] = parent_id
            p_meta["parent_index"] = p_idx
            parent_chunks.append(ParentChunk(id=parent_id, content=p_clean, metadata=p_meta))

            # Split parent into child chunks
            raw_children = self.child_splitter.split_text(p_clean)
            for c_idx, c_text in enumerate(raw_children, start=1):
                c_clean = c_text.strip()
                if not c_clean:
                    continue

                c_hash = hashlib.sha1(c_clean.encode("utf-8")).hexdigest()[:8]
                child_id = f"{parent_id}:c{c_idx}:{c_hash}"

                c_meta = dict(base_metadata)
                c_meta["id"] = child_id
                c_meta["parent_id"] = parent_id
                c_meta["child_index"] = c_idx
                c_meta["char_count"] = len(c_clean)
                c_meta["word_count"] = len(c_clean.split())

                child_documents.append(Document(page_content=c_clean, metadata=c_meta))

        return parent_chunks, child_documents
