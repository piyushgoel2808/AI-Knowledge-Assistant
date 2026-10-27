"""Unit tests for Cross-Encoder Reranking and Deterministic Abstention Gate (TICK-005)."""

import pytest
from unittest.mock import MagicMock, patch

from core.reranker import CrossEncoderReranker, sigmoid
from core.vector_store import RetrievedChunk
from core.rag_engine import answer_question, ChatMessage


def test_sigmoid_math():
    assert sigmoid(0.0) == 0.5
    assert sigmoid(100.0) > 0.99
    assert sigmoid(-100.0) < 0.01


def test_reranker_scores_and_orders_chunks():
    reranker = CrossEncoderReranker(threshold=0.30)
    # Mock the internal score calculation to avoid downloading weights during fast unit test
    reranker._score_raw = MagicMock(return_value=[-2.0, 3.5, 0.1])

    chunks = [
        RetrievedChunk(id="c1", content="Chunk about unrelated recipes", metadata={"source": "doc1"}),
        RetrievedChunk(id="c2", content="Chunk about parental leave policy details", metadata={"source": "doc2"}),
        RetrievedChunk(id="c3", content="Chunk about general office hours", metadata={"source": "doc3"}),
    ]

    reranked = reranker.rerank("What is the parental leave policy?", chunks)

    assert len(reranked) == 3
    # c2 had raw score 3.5 -> highest sigmoid -> should be first
    assert reranked[0].id == "c2"
    assert reranked[0].rerank_score is not None
    assert reranked[0].rerank_score > 0.9
    # c3 had raw score 0.1 -> second
    assert reranked[1].id == "c3"
    # c1 had raw score -2.0 -> last
    assert reranked[2].id == "c1"


def test_abstention_gate_triggers_when_top_score_below_threshold():
    reranker = CrossEncoderReranker(threshold=0.30)
    
    # Empty candidate list -> should abstain
    assert reranker.should_abstain([]) is True

    # Low score chunks (< 0.30) -> should abstain
    low_chunks = [
        RetrievedChunk(id="c1", content="low score", metadata={}, rerank_score=0.25),
        RetrievedChunk(id="c2", content="lower score", metadata={}, rerank_score=0.10),
    ]
    assert reranker.should_abstain(low_chunks) is True

    # High score chunks (>= 0.30) -> should NOT abstain
    high_chunks = [
        RetrievedChunk(id="c1", content="relevant doc", metadata={}, rerank_score=0.85),
        RetrievedChunk(id="c2", content="partial doc", metadata={}, rerank_score=0.20),
    ]
    assert reranker.should_abstain(high_chunks) is False


def test_answer_question_deterministic_refusal_fast_path():
    # Mock LLM provider that would raise an error if invoked
    mock_provider = MagicMock()
    mock_provider.generate.side_effect = AssertionError("LLM should not be called when abstaining!")

    # Mock search_documents to return candidate chunks with low rerank scores
    candidate_chunks = [
        RetrievedChunk(id="c1", content="Irrelevant policy", metadata={"source": "handbook.pdf"}),
    ]

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        RetrievedChunk(id="c1", content="Irrelevant policy", metadata={"source": "handbook.pdf"}, rerank_score=0.15),
    ]
    mock_reranker.should_abstain.return_value = True

    with patch("core.rag_engine.count_documents", return_value=10):
        with patch("core.rag_engine.search_documents", return_value=candidate_chunks):
            result = answer_question(
                question="What is the Q3 corporate revenue?",
                provider=mock_provider,
                reranker=mock_reranker,
            )

    assert result["refusal"] is True
    assert result["confidence"] == "low"
    assert "could not find the answer" in result["answer"].lower()
    assert result["sources"] == []
    mock_provider.generate.assert_not_called()
