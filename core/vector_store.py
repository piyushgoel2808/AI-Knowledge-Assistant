"""Persistent ChromaDB helpers for storing, listing, and retrieving document chunks."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any, Sequence

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_core.documents import Document

from core.config import AppConfig, load_config
from core.hierarchical_chunker import ParentDocStore
from core.hybrid_search import bm25_index, reciprocal_rank_fusion


DEFAULT_COLLECTION_NAME = "ai_knowledge_assistant"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class RetrievedChunk:
    """Chunk returned from vector search."""

    id: str
    content: str
    metadata: dict[str, object]
    distance: float | None = None
    parent_content: str | None = None
    rerank_score: float | None = None


def _resolve_config(config: AppConfig | None = None) -> AppConfig:
    return config or load_config()


def _ensure_persist_dir(config: AppConfig) -> Path:
    config.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return config.chroma_persist_dir


@lru_cache(maxsize=1)
def _get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=DEFAULT_EMBEDDING_MODEL)


@lru_cache(maxsize=4)
def _get_cached_client(persist_path: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=persist_path)


def get_client(config: AppConfig | None = None) -> chromadb.PersistentClient:
    """Create or reuse the persistent Chroma client."""

    resolved_config = _resolve_config(config)
    persist_dir = _ensure_persist_dir(resolved_config)
    return _get_cached_client(str(persist_dir.resolve()))


def get_collection(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    config: AppConfig | None = None,
) -> Collection:
    """Return the persistent collection used for document chunks."""

    client = get_client(config)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=_get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(
    documents: Sequence[Document],
    collection_name: str = DEFAULT_COLLECTION_NAME,
    config: AppConfig | None = None,
) -> list[str]:
    """Upsert LangChain documents into the persistent Chroma collection, replacing prior chunks of re-uploaded files."""

    if not documents:
        return []

    collection = get_collection(collection_name=collection_name, config=config)

    # Deduplicate by deleting prior chunks belonging to any file being re-uploaded
    sources_in_batch = {
        str(doc.metadata.get("source"))
        for doc in documents
        if doc.metadata and doc.metadata.get("source")
    }
    for source_name in sources_in_batch:
        try:
            collection.delete(where={"source": source_name})
        except Exception:
            pass

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, object]] = []

    for index, document in enumerate(documents, start=1):
        chunk_id = _build_chunk_id(document, index)
        ids.append(chunk_id)
        texts.append(document.page_content)
        metadatas.append(_sanitize_metadata(document.metadata))

    collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    # Rebuild in-memory BM25 lexical index with all active chunks
    try:
        all_records = collection.get(include=["documents", "metadatas"])
        if all_records and all_records.get("ids"):
            corpus = [
                {"id": doc_id, "content": doc_text, "metadata": doc_meta}
                for doc_id, doc_text, doc_meta in zip(
                    all_records["ids"],
                    all_records.get("documents") or [],
                    all_records.get("metadatas") or [],
                )
            ]
            bm25_index.build(corpus)
    except Exception:
        pass

    return ids


def search_documents(
    query: str,
    top_k: int = 5,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    config: AppConfig | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the most relevant chunks using Hybrid Search (Dense Vector + BM25) with RRF."""

    collection = get_collection(collection_name=collection_name, config=config)
    total = collection.count()
    if total == 0:
        return []

    # Ensure BM25 is populated if empty
    if bm25_index.count() == 0:
        try:
            all_records = collection.get(include=["documents", "metadatas"])
            if all_records and all_records.get("ids"):
                corpus = [
                    {"id": doc_id, "content": doc_text, "metadata": doc_meta}
                    for doc_id, doc_text, doc_meta in zip(
                        all_records["ids"],
                        all_records.get("documents") or [],
                        all_records.get("metadatas") or [],
                    )
                ]
                bm25_index.build(corpus)
        except Exception:
            pass

    fetch_k = max(1, min(top_k * 3, total))
    dense_result = collection.query(
        query_texts=[query],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )

    dense_docs = dense_result.get("documents", [[]])[0] if dense_result.get("documents") else []
    dense_metas = dense_result.get("metadatas", [[]])[0] if dense_result.get("metadatas") else []
    dense_dists = dense_result.get("distances", [[]])[0] if dense_result.get("distances") else []
    dense_ids = dense_result.get("ids", [[]])[0] if dense_result.get("ids") else []

    # Dense candidate map
    candidate_map: dict[str, dict[str, Any]] = {}
    for idx, d_id in enumerate(dense_ids):
        candidate_map[d_id] = {
            "id": d_id,
            "content": dense_docs[idx],
            "metadata": dense_metas[idx] if idx < len(dense_metas) else {},
            "distance": dense_dists[idx] if idx < len(dense_dists) else None,
        }

    # BM25 lexical candidate search
    bm25_matches = bm25_index.search(query, top_k=fetch_k)
    sparse_ids = [m["id"] for m in bm25_matches]
    for m in bm25_matches:
        if m["id"] not in candidate_map:
            candidate_map[m["id"]] = {
                "id": m["id"],
                "content": m["content"],
                "metadata": m.get("metadata") or {},
                "distance": None,
            }

    # Apply Reciprocal Rank Fusion
    fused_ranks = reciprocal_rank_fusion([dense_ids, sparse_ids], k=60)
    top_fused_ids = [cand_id for cand_id, _ in fused_ranks][:top_k]

    doc_store = ParentDocStore()
    chunks: list[RetrievedChunk] = []

    for cand_id in top_fused_ids:
        cand = candidate_map.get(cand_id)
        if not cand:
            continue

        meta = cand["metadata"]
        parent_id = meta.get("parent_id")
        parent_content = None
        if parent_id:
            parent_chunk = doc_store.get_parent(str(parent_id))
            if parent_chunk:
                parent_content = parent_chunk.content

        chunks.append(
            RetrievedChunk(
                id=cand["id"],
                content=cand["content"],
                metadata=dict(meta),
                distance=cand["distance"],
                parent_content=parent_content,
            )
        )

    return chunks


def count_documents(collection_name: str = DEFAULT_COLLECTION_NAME, config: AppConfig | None = None) -> int:
    """Return the number of stored chunks in the collection."""

    collection = get_collection(collection_name=collection_name, config=config)
    return collection.count()


def list_indexed_documents(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    config: AppConfig | None = None,
) -> list[dict[str, Any]]:
    """Return a summary list of all unique documents currently indexed in ChromaDB."""

    collection = get_collection(collection_name=collection_name, config=config)
    total = collection.count()
    if total == 0:
        return []

    records = collection.get(include=["metadatas"])
    metadatas = records.get("metadatas") or []

    by_source: dict[str, dict[str, Any]] = {}
    for meta in metadatas:
        if not meta:
            continue
        source = str(meta.get("source", "unknown"))
        entry = by_source.setdefault(
            source,
            {
                "source": source,
                "source_type": str(meta.get("source_type", "document")).upper(),
                "chunks": 0,
                "pages": set(),
                "sections": set(),
                "total_words": 0,
            },
        )
        entry["chunks"] += 1
        page = meta.get("page")
        if isinstance(page, int):
            entry["pages"].add(page)
        section = meta.get("section")
        if isinstance(section, str) and section.strip():
            entry["sections"].add(section.strip())
        words = meta.get("word_count")
        if isinstance(words, int):
            entry["total_words"] += words

    summaries: list[dict[str, Any]] = []
    for source in sorted(by_source.keys()):
        item = by_source[source]
        summaries.append(
            {
                "source": item["source"],
                "source_type": item["source_type"],
                "chunks": item["chunks"],
                "page_count": len(item["pages"]) if item["pages"] else None,
                "section_count": len(item["sections"]),
                "total_words": item["total_words"],
            }
        )
    return summaries


def delete_document(
    source_name: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    config: AppConfig | None = None,
) -> int:
    """Delete all chunks belonging to a specific source document."""

    collection = get_collection(collection_name=collection_name, config=config)
    before = collection.count()
    if before == 0:
        return 0
    collection.delete(where={"source": source_name})
    after = collection.count()
    return max(0, before - after)


def clear_collection(collection_name: str = DEFAULT_COLLECTION_NAME, config: AppConfig | None = None) -> None:
    """Delete the collection and recreate it empty."""

    client = get_client(config)
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    client.get_or_create_collection(
        name=collection_name,
        embedding_function=_get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def _build_chunk_id(document: Document, index: int) -> str:
    metadata = document.metadata or {}
    source = str(metadata.get("source", "document"))
    page = metadata.get("page", "na")
    chunk_index = metadata.get("chunk_index", index)
    digest = hashlib.sha1(document.page_content.encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"{source}:{page}:{chunk_index}:{index}:{digest}"


def _build_fallback_id(index: int) -> str:
    return f"chunk-{index}"


def _sanitize_metadata(metadata: dict[str, object] | None) -> dict[str, object]:
    if not metadata:
        return {}

    cleaned: dict[str, object] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned
