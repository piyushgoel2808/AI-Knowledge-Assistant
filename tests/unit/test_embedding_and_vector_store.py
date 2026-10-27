"""Unit tests for EmbeddingGemma-2, OllamaEmbeddingFunction, and dynamic collection isolation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from core.config import AppConfig, load_config
from core.vector_store import (
    OllamaEmbeddingFunction,
    get_default_collection_name,
    get_embedding_function,
)


def test_ollama_embedding_function_generates_embeddings():
    """Verify OllamaEmbeddingFunction calls /api/embed and formats output."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "embeddings": [[0.1] * 768, [0.2] * 768],
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        ef = OllamaEmbeddingFunction(base_url="http://localhost:11434", model_name="embeddinggemma-2")
        vectors = ef(["Chunk one", "Chunk two"])

        assert len(vectors) == 2
        assert len(vectors[0]) == 768
        assert vectors[0][0] == 0.1
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["model"] == "embeddinggemma-2"
        assert call_kwargs["json"]["input"] == ["Chunk one", "Chunk two"]


def test_ollama_embedding_function_single_document():
    """Single document list should generate embedding."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "embeddings": [[0.5] * 768],
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        ef = OllamaEmbeddingFunction()
        vectors = ef(["Single document"])
        assert len(vectors) == 1
        assert len(vectors[0]) == 768
        mock_post.assert_called_once()


def test_ollama_embedding_function_config_roundtrip():
    """Verify ChromaDB get_config and build_from_config contract."""
    ef = OllamaEmbeddingFunction(base_url="http://remote-ollama:11434", model_name="embeddinggemma-2", timeout=30.0)
    config = ef.get_config()
    assert config["base_url"] == "http://remote-ollama:11434"
    assert config["model_name"] == "embeddinggemma-2"
    assert config["timeout"] == 30.0

    restored = OllamaEmbeddingFunction.build_from_config(config)
    assert restored.base_url == "http://remote-ollama:11434"
    assert restored.model_name == "embeddinggemma-2"
    assert restored.timeout == 30.0


def test_dynamic_collection_naming_prevents_dimension_collision():
    """Different embedding models should resolve to unique collection names to isolate dimensions."""
    config_gemma = AppConfig(
        llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="gemma4:e4b",
        embedding_provider="ollama",
        embedding_model="embeddinggemma-2",
        gemini_api_key="",
        gemini_model="",
        groq_api_key="",
        groq_model="",
        chroma_persist_dir=MagicMock(),
        upload_dir=MagicMock(),
    )
    config_minilm = AppConfig(
        llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="gemma4:e4b",
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        gemini_api_key="",
        gemini_model="",
        groq_api_key="",
        groq_model="",
        chroma_persist_dir=MagicMock(),
        upload_dir=MagicMock(),
    )

    col_gemma = get_default_collection_name(config_gemma)
    col_minilm = get_default_collection_name(config_minilm)

    assert col_gemma != col_minilm
    assert "embeddinggemma_2" in col_gemma
    assert "all_minilm_l6_v2" in col_minilm


def test_custom_collection_override():
    """Explicit collection name in config should take priority over automatic naming."""
    config_override = AppConfig(
        llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="gemma4:e4b",
        embedding_provider="ollama",
        embedding_model="embeddinggemma-2",
        chroma_collection_name="my_custom_vault",
        gemini_api_key="",
        gemini_model="",
        groq_api_key="",
        groq_model="",
        chroma_persist_dir=MagicMock(),
        upload_dir=MagicMock(),
    )
    assert get_default_collection_name(config_override) == "my_custom_vault"
