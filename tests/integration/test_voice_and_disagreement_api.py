"""Integration tests for Hands-Free Voice Layer and Disagreement API Contracts (TICK-008)."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from backend.main import app
from core.vector_store import RetrievedChunk
from core.ai_service import LLMResponse


@pytest.fixture
def client():
    return TestClient(app)


def test_chat_endpoint_returns_extended_attribution_contracts(client):
    mock_provider = MagicMock()
    mock_provider.generate.return_value = LLMResponse(
        text="The annual leave allowance is 20 days per year [S1]."
    )

    chunks = [
        RetrievedChunk(
            id="c1",
            content="Employees are granted an annual leave allowance of 20 days.",
            metadata={"source": "handbook_2026.pdf", "page": 5, "section": "Leave"},
            distance=0.15,
            rerank_score=0.91,
        ),
    ]

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = chunks
    mock_reranker.should_abstain.return_value = False

    with patch("core.rag_engine.count_documents", return_value=1):
        with patch("core.rag_engine.search_documents", return_value=chunks):
            with patch("core.rag_engine.create_llm_provider", return_value=mock_provider):
                with patch("core.rag_engine.default_reranker", mock_reranker):
                    response = client.post(
                        "/chat",
                        json={
                            "question": "What is the annual leave allowance?",
                            "history": [],
                            "top_k": 3,
                        },
                    )

    assert response.status_code == 200
    data = response.json()
    assert data["refusal"] is False
    assert "standalone_query" in data
    assert "has_disagreements" in data
    assert "disagreements" in data
    assert len(data["sources"]) == 1
    src = data["sources"][0]
    assert src["verified"] is True
    assert src["verification_status"] == "verified"
    assert src["rerank_score"] == 0.91


def test_chat_endpoint_abstention_fast_path(client):
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        RetrievedChunk(id="c1", content="Unrelated topic", metadata={"source": "faq.pdf"}, rerank_score=0.18),
    ]
    mock_reranker.should_abstain.return_value = True

    mock_provider = MagicMock()
    mock_provider.generate.side_effect = AssertionError("LLM should not be called on abstention!")

    with patch("core.rag_engine.count_documents", return_value=5):
        with patch("core.rag_engine.search_documents", return_value=[
            RetrievedChunk(id="c1", content="Unrelated topic", metadata={"source": "faq.pdf"}),
        ]):
            with patch("core.rag_engine.create_llm_provider", return_value=mock_provider):
                with patch("core.rag_engine.default_reranker", mock_reranker):
                    response = client.post(
                        "/chat",
                        json={
                            "question": "What is the rocket launch schedule for Mars?",
                            "history": [],
                        },
                    )

    assert response.status_code == 200
    data = response.json()
    assert data["refusal"] is True
    assert data["confidence"] == "low"
    assert "could not find the answer" in data["answer"].lower()
    assert data["sources"] == []
    mock_provider.generate.assert_not_called()
