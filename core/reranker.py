"""Cross-Encoder Reranker and Deterministic Abstention Gate (TICK-005).

Scores candidate chunks against the user query using cross-attention,
and deterministically abstains when relevance falls below threshold tau.
"""

from __future__ import annotations

import math
from typing import Sequence

from core.vector_store import RetrievedChunk


def sigmoid(x: float) -> float:
    """Compute numerically stable sigmoid for mapping logits to [0, 1]."""
    if x >= 40.0:
        return 1.0
    if x <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderReranker:
    """Cross-encoder reranking engine with deterministic refusal gating."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        threshold: float = 0.30,
        lazy: bool = True,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self._model = None
        if not lazy:
            self._get_model()

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                # Fallback: if weights cannot be loaded, keep _model None
                self._model = None
        return self._model

    def _score_raw(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Produce raw cross-encoder logit scores for query-passage pairs."""
        model = self._get_model()
        if model is None:
            # Fallback if model not available: assign neutral scores
            return [0.0] * len(pairs)
        scores = model.predict(pairs)
        if hasattr(scores, "tolist"):
            return [float(s) for s in scores.tolist()]
        if isinstance(scores, (list, tuple)):
            return [float(s) for s in scores]
        return [float(scores)]

    def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Rerank candidate chunks against the query and sort descending by score."""
        if not chunks:
            return []

        # Form query-passage pairs using the most descriptive text available
        pairs = [
            (query, (chunk.parent_content or chunk.content).strip())
            for chunk in chunks
        ]

        raw_scores = self._score_raw(pairs)
        scored_chunks: list[RetrievedChunk] = []

        for chunk, raw_score in zip(chunks, raw_scores):
            normalized_score = round(sigmoid(float(raw_score)), 4)
            scored_chunks.append(
                RetrievedChunk(
                    id=chunk.id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    distance=chunk.distance,
                    parent_content=chunk.parent_content,
                    rerank_score=normalized_score,
                )
            )

        scored_chunks.sort(key=lambda c: (c.rerank_score if c.rerank_score is not None else 0.0), reverse=True)

        if top_k is not None:
            return scored_chunks[:top_k]
        return scored_chunks

    def should_abstain(
        self,
        reranked_chunks: Sequence[RetrievedChunk],
        threshold: float | None = None,
    ) -> bool:
        """Return True if top rerank score is below threshold tau, triggering abstention."""
        tau = self.threshold if threshold is None else threshold
        if not reranked_chunks:
            return True
        top_score = reranked_chunks[0].rerank_score
        if top_score is None:
            return False
        return top_score < tau


# Default singleton instance
default_reranker = CrossEncoderReranker()
