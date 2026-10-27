"""Unit tests for Two-Tier Attribution Verification & Multi-Document Disagreement Engine (TICK-007)."""

import pytest
from unittest.mock import MagicMock, patch

from core.attribution_verifier import (
    verify_citation_entailment,
    detect_multi_document_disagreements,
    verify_answer_attributions,
)
from core.vector_store import RetrievedChunk
from core.ai_service import LLMResponse
from core.rag_engine import answer_question


def test_verify_claim_entailment_verified():
    snippet = "Full-time employees receive 18 days of paid annual leave per calendar year."
    claim = "Employees receive 18 days annual leave."

    result = verify_citation_entailment(claim, snippet)
    assert result["verified"] is True
    assert result["status"] == "verified"
    assert result["overlap_score"] > 0.5


def test_verify_claim_entailment_unverified_hallucination():
    snippet = "The company matches up to 5% of salary contributions to the 401k retirement plan."
    # Fabricated claim citing this snippet
    claim = "Employees receive 35 days of paid annual vacation."

    result = verify_citation_entailment(claim, snippet)
    assert result["verified"] is False
    assert result["status"] == "unverified"
    assert result["overlap_score"] < 0.2


def test_detect_multi_document_disagreement():
    chunks = [
        RetrievedChunk(
            id="c1",
            content="The standard probation period is 3 months from start date.",
            metadata={"source": "Handbook_US.pdf", "page": 4},
        ),
        RetrievedChunk(
            id="c2",
            content="The statutory probation period is 6 months for all staff.",
            metadata={"source": "Handbook_UK.pdf", "page": 6},
        ),
    ]

    disagreements = detect_multi_document_disagreements(chunks)
    assert len(disagreements) > 0
    d = disagreements[0]
    assert "probation" in d["topic"].lower()
    sources_involved = [s["source"] for s in d["conflicting_sources"]]
    assert "Handbook_US.pdf" in sources_involved
    assert "Handbook_UK.pdf" in sources_involved


def test_answer_question_enriches_sources_with_verification():
    mock_provider = MagicMock()
    mock_provider.generate.return_value = LLMResponse(
        text="Employees receive 18 days annual leave [S1]."
    )

    chunks = [
        RetrievedChunk(
            id="c1",
            content="Full-time staff receive 18 days annual leave.",
            metadata={"source": "benefits.pdf", "page": 12, "section": "Leave"},
            distance=0.1,
            rerank_score=0.92,
        ),
    ]

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = chunks
    mock_reranker.should_abstain.return_value = False

    with patch("core.rag_engine.count_documents", return_value=1):
        with patch("core.rag_engine.search_documents", return_value=chunks):
            result = answer_question(
                question="How much annual leave do employees receive?",
                provider=mock_provider,
                reranker=mock_reranker,
            )

    assert result["refusal"] is False
    assert len(result["sources"]) == 1
    source = result["sources"][0]
    assert source["verified"] is True
    assert source["verification_status"] == "verified"
    assert "has_disagreements" in result
