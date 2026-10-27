"""Application configuration helpers with live .env hot-reloading and Docker host translation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"


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
    embedding_provider: str = "ollama"
    embedding_model: str = "embeddinggemma-2"
    chroma_collection_name: str | None = None

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


def _resolve_ollama_url(raw_url: str) -> str:
    """Translate localhost/127.0.0.1 to host.docker.internal automatically when running in Docker."""
    cleaned = raw_url.strip()
    if os.getenv("RUNNING_IN_DOCKER", "").lower() in {"1", "true", "yes"}:
        parsed = urlparse(cleaned)
        if parsed.hostname in {"localhost", "127.0.0.1"}:
            port_suffix = f":{parsed.port}" if parsed.port else ":11434"
            return urlunparse(parsed._replace(netloc=f"host.docker.internal{port_suffix}"))
    return cleaned


def load_config() -> AppConfig:
    """Reload .env dynamically on each call so config changes never require a container rebuild."""

    if DOTENV_PATH.exists():
        load_dotenv(dotenv_path=DOTENV_PATH, override=True)
    else:
        load_dotenv(override=True)

    raw_ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    return AppConfig(
        llm_provider=os.getenv("LLM_PROVIDER", "ollama").strip().lower(),
        ollama_base_url=_resolve_ollama_url(raw_ollama_url),
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
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "ollama").strip().lower(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "embeddinggemma-2").strip(),
        chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "").strip() or None,
    )