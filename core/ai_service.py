"""LLM provider adapter layer with connection pooling and Apple Silicon Ollama optimizations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Any
import requests
from groq import Groq

from core.config import AppConfig, load_config


# Shared HTTP session for connection pooling across LLM requests
_HTTP_SESSION = requests.Session()


@dataclass(frozen=True)
class LLMResponse:
    """Standard response returned by any provider implementation."""

    text: str
    raw: Any | None = None


def _strip_thinking_tags(text: str) -> str:
    """Remove any inline <think>...</think> blocks if emitted by reasoning models."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


class BaseLLMProvider(ABC):
    """Common interface for all LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """Generate a completion for the supplied prompt."""


class OllamaProvider(BaseLLMProvider):
    """Talk to a local Ollama instance over HTTP, optimized for Apple Silicon Unified Memory."""

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.1,
                "num_ctx": 8192,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = _HTTP_SESSION.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to reach Ollama at {self.base_url} with model '{self.model}': {exc}"
            ) from exc

        data = response.json()
        raw_text = str(data.get("response", "")).strip()
        return LLMResponse(text=_strip_thinking_tags(raw_text), raw=data)


class GeminiProvider(BaseLLMProvider):
    """Use the Google Gemini API for hosted inference."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for the Gemini provider")
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1},
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        try:
            response = _HTTP_SESSION.post(
                url,
                headers={"Content-Type": "application/json", "X-goog-api-key": self.api_key},
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Gemini API request failed for model '{self.model}': {exc}") from exc

        data = response.json()
        return LLMResponse(text=_strip_thinking_tags(self._extract_text(data)), raw=data)

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts") or []
        return "\n".join([str(p.get("text", "")) for p in parts if p.get("text")])


class GroqProvider(BaseLLMProvider):
    """Use the Groq API for high-speed inference."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("GROQ_API_KEY is required for the Groq provider")
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.1,
            )
        except Exception as exc:
            raise RuntimeError(f"Groq API request failed for model '{self.model}': {exc}") from exc

        text = chat_completion.choices[0].message.content or ""
        return LLMResponse(
            text=_strip_thinking_tags(text),
            raw={"id": getattr(chat_completion, "id", None), "model": self.model},
        )


def create_llm_provider(config: AppConfig | None = None) -> BaseLLMProvider:
    """Create the active provider from environment-driven config."""

    resolved_config = config or load_config()
    provider_name = resolved_config.llm_provider.lower()

    if provider_name == "ollama":
        return OllamaProvider(
            base_url=resolved_config.ollama_base_url,
            model=resolved_config.ollama_model,
        )

    if provider_name == "gemini":
        return GeminiProvider(
            api_key=resolved_config.gemini_api_key,
            model=resolved_config.gemini_model,
        )

    if provider_name == "groq":
        return GroqProvider(
            api_key=resolved_config.groq_api_key,
            model=resolved_config.groq_model,
        )

    raise ValueError(f"Unsupported LLM provider: {resolved_config.llm_provider}")