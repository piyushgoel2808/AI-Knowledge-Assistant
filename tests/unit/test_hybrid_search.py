"""Unit tests for Ticket 04: Hybrid Search (Dense + BM25) and Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest
from core.hybrid_search import (
    BM25Index,
    reciprocal_rank_fusion,
)


def test_rrf_scoring_algorithm():
    """reciprocal_rank_fusion should correctly calculate RRF scores and rank candidates."""
    dense_ranks = ["doc_a", "doc_b", "doc_c"]
    sparse_ranks = ["doc_b", "doc_d", "doc_a"]

    fused = reciprocal_rank_fusion([dense_ranks, sparse_ranks], k=60)

    # doc_b is rank 2 in dense, rank 1 in sparse -> highest total RRF score
    assert fused[0][0] == "doc_b", f"Expected doc_b at rank 1, got {fused[0]}"
    # doc_a is rank 1 in dense, rank 3 in sparse -> second highest
    assert fused[1][0] == "doc_a", f"Expected doc_a at rank 2, got {fused[1]}"
    # doc_d is rank 2 in sparse only -> third
    assert fused[2][0] == "doc_d"
    # doc_c is rank 3 in dense only -> fourth
    assert fused[3][0] == "doc_c"


def test_bm25_exact_token_matching():
    """BM25Index should find exact alphanumeric policy tokens that dense vectors miss."""
    documents = [
        {"id": "doc_1", "content": "General corporate guidance and team collaboration expectations."},
        {"id": "doc_2", "content": "Mandatory security protocol ISO-27001-A9 requires hardware keys."},
        {"id": "doc_3", "content": "Travel expense reimbursement and flight booking per diems."},
    ]

    index = BM25Index()
    index.build(documents)

    results = index.search("ISO-27001-A9", top_k=2)

    assert len(results) >= 1
    assert results[0]["id"] == "doc_2"
    assert "ISO-27001-A9" in results[0]["content"]
    assert results[0]["score"] > 0
