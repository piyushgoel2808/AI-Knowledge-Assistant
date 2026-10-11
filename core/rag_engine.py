"""Retrieval-augmented generation orchestration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from core.ai_service import BaseLLMProvider, create_llm_provider
from core.config import AppConfig, load_config
from core.vector_store import RetrievedChunk, count_documents, search_documents


REFUSAL_PHRASES = (
    "could not find the answer in the uploaded documents",
    "could not find it in the uploaded documents",
    "not mentioned in the uploaded documents",
    "not present in the retrieved document context",
    "cannot find the answer in the provided",
)


@dataclass(frozen=True)
class ChatMessage:
    """Minimal chat history entry used by the prompt builder."""

    role: str
    content: str


def answer_question(
    question: str,
    chat_history: Sequence[ChatMessage] | None = None,
    top_k: int | None = None,
    config: AppConfig | None = None,
    provider: BaseLLMProvider | None = None,
) -> dict[str, object]:
    """Answer a question using only retrieved document context."""

    resolved_config = config or load_config()
    total_indexed_chunks = count_documents(config=resolved_config)

    if total_indexed_chunks == 0:
        return {
            "answer": "No documents are indexed yet. Please upload and index one or more PDF, DOCX, or TXT files first.",
            "sources": [],
            "confidence": "low",
            "refusal": True,
        }

    # Cap retrieval at top_k (default 5) instead of loading the entire database into the prompt
    desired_k = top_k if top_k is not None else resolved_config.top_k
    search_limit = max(1, min(desired_k, total_indexed_chunks))

    retrieved_chunks = search_documents(
        query=question,
        top_k=search_limit,
        config=resolved_config,
    )

    if not retrieved_chunks:
        return {
            "answer": "I could not find the answer in the uploaded documents.",
            "sources": [],
            "confidence": "low",
            "refusal": True,
        }

    resolved_provider = provider or create_llm_provider(resolved_config)
    system_prompt = _build_system_prompt()
    prompt = _build_prompt(question=question, chat_history=chat_history, chunks=retrieved_chunks)
    llm_response = resolved_provider.generate(prompt=prompt, system_prompt=system_prompt)

    answer_text = llm_response.text.strip()
    if not answer_text:
        answer_text = "I could not find the answer in the uploaded documents."

    is_refusal = _is_refusal_response(answer_text)

    # Parse citations like [S1], [S2] even if combined like [S1, S2]
    cited_indices: set[int] = set()
    for match in re.findall(r"\[(.*?)\]", answer_text):
        if "s" in match.lower() or match.strip().isdigit() or "," in match:
            numbers = re.findall(r"\d+", match)
            cited_indices.update(
                int(n) for n in numbers if 1 <= int(n) <= len(retrieved_chunks)
            )

    sources: list[dict[str, object]] = []
    if not is_refusal:
        if cited_indices:
            for index, chunk in enumerate(retrieved_chunks, start=1):
                if index in cited_indices:
                    sources.append(_build_source_entry(chunk, index, answer_text))
            confidence = "high"
        else:
            # Fallback: if the model answered substantively without bracket citations,
            # attach the top matching retrieved chunks for transparency
            for index, chunk in enumerate(retrieved_chunks[:2], start=1):
                sources.append(_build_source_entry(chunk, index, answer_text))
            confidence = "medium" if sources else "low"
    else:
        confidence = "low"

    return {
        "answer": answer_text,
        "sources": sources,
        "confidence": confidence,
        "refusal": is_refusal,
        "model": resolved_config.active_model,
        "provider": resolved_config.llm_provider,
    }


def _is_refusal_response(answer_text: str) -> bool:
    normalized = answer_text.strip().lower()
    return any(phrase in normalized for phrase in REFUSAL_PHRASES)


def _build_source_entry(chunk: RetrievedChunk, cite_index: int, answer_text: str) -> dict[str, object]:
    source_dict = _chunk_to_source(chunk)
    source_dict["cite_index"] = cite_index
    source_dict["snippet"] = _extract_and_highlight_match(
        chunk_text=chunk.content.strip(),
        answer_text=answer_text,
    )
    return source_dict


def _extract_and_highlight_match(chunk_text: str, answer_text: str) -> str:
    """Find overlapping sentences between the document chunk and the AI answer,
    returning relevant sentences with matched keywords highlighted.
    """
    clean_answer = re.sub(r"\[.*?\]", "", answer_text).lower()
    sentences = re.split(r"(?<=[.!?])\s+", chunk_text)
    matched_sentences = []

    for sentence in sentences:
        words = [w.strip(",.()\"'").lower() for w in sentence.split() if len(w) > 4]
        match_count = sum(1 for word in words if word in clean_answer)

        if match_count >= 2:
            highlighted_sentence = sentence
            for word in set(
                w
                for w in sentence.split()
                if len(w) > 4 and w.lower().strip(",.()\"'") in clean_answer
            ):
                safe_word = re.escape(word.strip(",.()\"'"))
                if safe_word:
                    highlighted_sentence = re.sub(
                        rf"\b({safe_word})\b",
                        r"**\1**",
                        highlighted_sentence,
                        flags=re.IGNORECASE,
                    )
            matched_sentences.append(highlighted_sentence)

    return "\n... ".join(matched_sentences) if matched_sentences else chunk_text


def _build_system_prompt() -> str:
    return (
        "You are a strict retrieval-augmented knowledge assistant. "
        "Answer ONLY from the provided document context. "
        "If the context does not contain the answer, respond exactly with: "
        "'I could not find the answer in the uploaded documents.' "
        "Do not use outside knowledge. Do not guess. "
        "Format your response clearly (use bullet points when listing multiple items) and only state claims supported by the context. "
        "IMPORTANT: You MUST cite the specific source context you used by appending its exact marker (e.g., [S1], [S2]) to the end of every relevant sentence or bullet."
    )


def _build_prompt(
    question: str,
    chat_history: Sequence[ChatMessage] | None,
    chunks: Sequence[RetrievedChunk],
) -> str:
    history_block = _format_chat_history(chat_history)
    context_block = _format_context(chunks)

    return (
        "Conversation history:\n"
        f"{history_block}\n\n"
        "Retrieved document context:\n"
        f"{context_block}\n\n"
        "User question:\n"
        f"{question}\n\n"
        "Instructions:\n"
        "- Use only the retrieved document context.\n"
        "- Cite sources inline using [S1], [S2], etc.\n"
        "- If the answer is not present, say: I could not find the answer in the uploaded documents.\n"
        "- Keep the answer accurate, structured, and direct.\n"
    )


def _format_chat_history(chat_history: Sequence[ChatMessage] | None) -> str:
    if not chat_history:
        return "No prior conversation."

    # Keep the last 6 messages to prevent context window bloat
    recent_messages = list(chat_history)[-6:]
    lines: list[str] = []
    for message in recent_messages:
        role = message.role.strip().lower() or "user"
        content = message.content.strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "No prior conversation."


def _format_context(chunks: Sequence[RetrievedChunk]) -> str:
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "n/a")
        section = chunk.metadata.get("section", "Document")
        lines.append(
            f"[S{index}] source={source} page={page} section={section}\n{chunk.content.strip()}"
        )
    return "\n\n".join(lines)


def _chunk_to_source(chunk: RetrievedChunk) -> dict[str, object]:
    distance = chunk.distance
    similarity = round(max(0.0, 1.0 - float(distance)), 3) if isinstance(distance, (int, float)) else None
    return {
        "source": chunk.metadata.get("source", "unknown"),
        "page": chunk.metadata.get("page"),
        "section": chunk.metadata.get("section", "Document"),
        "snippet": chunk.content.strip(),
        "chunk_id": chunk.id,
        "distance": distance,
        "similarity": similarity,
    }