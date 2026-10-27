"""Unit tests for Context-Aware Conversational Query Condensation (TICK-006)."""

from unittest.mock import MagicMock, patch
import pytest

from core.ai_service import LLMResponse
from core.query_condenser import condense_query
from core.rag_engine import ChatMessage, answer_question
from core.vector_store import RetrievedChunk


def test_condense_query_without_history_returns_verbatim():
    mock_provider = MagicMock()
    result = condense_query("What is the leave policy?", chat_history=None, provider=mock_provider)
    assert result == "What is the leave policy?"
    mock_provider.generate.assert_not_called()

    result_empty = condense_query("What is the leave policy?", chat_history=[], provider=mock_provider)
    assert result_empty == "What is the leave policy?"
    mock_provider.generate.assert_not_called()


def test_condense_query_with_history_calls_llm_few_shot():
    mock_provider = MagicMock()
    mock_provider.generate.return_value = LLMResponse(
        text="Does parental leave apply to independent contractors for Software Engineers?"
    )

    history = [
        ChatMessage(role="user", content="What is parental leave for Software Engineers?"),
        ChatMessage(role="assistant", content="Software Engineers receive 16 weeks paid parental leave."),
    ]

    result = condense_query("Does it also apply to contractors?", chat_history=history, provider=mock_provider)

    assert result == "Does parental leave apply to independent contractors for Software Engineers?"
    mock_provider.generate.assert_called_once()
    called_prompt = mock_provider.generate.call_args[1]["prompt"]
    assert "Software Engineers" in called_prompt
    assert "Does it also apply to contractors?" in called_prompt


def test_condense_query_cleans_llm_output():
    mock_provider = MagicMock()
    mock_provider.generate.return_value = LLMResponse(
        text='Standalone query: "What is the deductible for out-of-network dental care?"'
    )

    history = [
        ChatMessage(role="user", content="Tell me about dental benefits."),
        ChatMessage(role="assistant", content="In-network dental is covered 100%."),
    ]

    result = condense_query("What about out-of-network?", chat_history=history, provider=mock_provider)
    assert result == "What is the deductible for out-of-network dental care?"


def test_answer_question_uses_condensed_query_for_retrieval():
    mock_provider = MagicMock()
    mock_provider.generate.side_effect = [
        LLMResponse(text="Does parental leave apply to contractors for Software Engineers?"),  # condense_query call
        LLMResponse(text="Parental leave does not apply to contractors [S1]."),              # main answer call
    ]

    history = [
        ChatMessage(role="user", content="Parental leave for Software Engineers"),
        ChatMessage(role="assistant", content="Engineers receive 16 weeks leave."),
    ]

    mock_search = MagicMock(return_value=[
        RetrievedChunk(id="c1", content="Contractors are not eligible for parental leave.", metadata={"source": "handbook.pdf"}),
    ])

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        RetrievedChunk(id="c1", content="Contractors are not eligible for parental leave.", metadata={"source": "handbook.pdf"}, rerank_score=0.88),
    ]
    mock_reranker.should_abstain.return_value = False

    with patch("core.rag_engine.count_documents", return_value=5):
        with patch("core.rag_engine.search_documents", mock_search):
            result = answer_question(
                question="Does it also apply to contractors?",
                chat_history=history,
                provider=mock_provider,
                reranker=mock_reranker,
            )

    # Search should have been called with rewritten query
    mock_search.assert_called_once()
    assert mock_search.call_args[1]["query"] == "Does parental leave apply to contractors for Software Engineers?"
    assert result["refusal"] is False
    assert result.get("standalone_query") == "Does parental leave apply to contractors for Software Engineers?"
