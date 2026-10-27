"""Hybrid Search combining Dense Vector ANN and BM25 with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import re
from threading import Lock
from typing import Any, Sequence

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Simple alphanumeric tokenizer preserving codes, hyphens, and section references."""
    # Split by whitespace, punctuation except hyphens/underscores inside words
    tokens = re.findall(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*", text.lower())
    return tokens if tokens else text.lower().split()


def reciprocal_rank_fusion(
    rank_lists: Sequence[Sequence[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine multiple ranked candidate lists into a single fused ranking using RRF."""
    rrf_scores: dict[str, float] = {}

    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    sorted_candidates = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_candidates


class BM25Index:
    """Thread-safe in-memory inverted index using BM25Okapi for exact lexical queries."""

    def __init__(self) -> None:
        self._index: BM25Okapi | None = None
        self._documents: list[dict[str, Any]] = []
        self._lock = Lock()

    def build(self, documents: Sequence[dict[str, Any]]) -> None:
        """Build or replace the inverted BM25 index with a list of document dicts."""
        doc_list = list(documents)
        tokenized_corpus = [_tokenize(doc.get("content", "")) for doc in doc_list]

        with self._lock:
            self._documents = doc_list
            if tokenized_corpus and any(tokenized_corpus):
                self._index = BM25Okapi(tokenized_corpus)
            else:
                self._index = None

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Return the top-k scoring documents matching the query tokens."""
        tokenized_query = _tokenize(query)
        if not tokenized_query or not self._documents:
            return []

        with self._lock:
            if self._index is None:
                return []
            scores = self._index.get_scores(tokenized_query)

        # Pair documents with scores
        scored_docs = []
        for doc, score in zip(self._documents, scores):
            if score > 0.0:
                doc_copy = dict(doc)
                doc_copy["score"] = float(score)
                scored_docs.append(doc_copy)

        scored_docs.sort(key=lambda d: d["score"], reverse=True)
        return scored_docs[:top_k]

    def count(self) -> int:
        with self._lock:
            return len(self._documents)


# Global singleton BM25 index instance
bm25_index = BM25Index()
