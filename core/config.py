"""Application configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    """Runtime settings loaded from environment variables."""

    llm_provider: str
    ollama_base_url: str
    ollama_model: str
    gemini_api_key: str
    gemini_model: str
    groq_api_key: str
    groq_model: str
    chroma_persist_dir: Path
    upload_dir: Path
    sample_doc_dir: Path = Path("Sample_doc")
    top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 150

    @property
    def active_model(self) -> str:
        """Return the model identifier for the currently selected LLM provider."""
        provider = self.llm_provider.lower()
        if provider == "ollama":
            return self.ollama_model
        if provider == "gemini":
            return self.gemini_model
        if provider == "groq":
            return self.groq_model
        return "unknown"


def load_config() -> AppConfig:
    """Load config values with sensible local defaults (Ollama + gemma4:e4b)."""

    return AppConfig(
        llm_provider=os.getenv("LLM_PROVIDER", "ollama").strip().lower(),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
        ollama_model=os.getenv("OLLAMA_MODEL", "gemma4:e4b").strip(),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip(),
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip(),
        chroma_persist_dir=Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma")).expanduser(),
        upload_dir=Path(os.getenv("UPLOAD_DIR", "data/uploads")).expanduser(),
        sample_doc_dir=Path(os.getenv("SAMPLE_DOC_DIR", "Sample_doc")).expanduser(),
        top_k=max(1, int(os.getenv("RAG_TOP_K", "5"))),
        chunk_size=max(200, int(os.getenv("CHUNK_SIZE", "1000"))),
        chunk_overlap=max(0, int(os.getenv("CHUNK_OVERLAP", "150"))),
    )