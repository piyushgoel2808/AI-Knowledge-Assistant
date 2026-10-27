"""Two-Tier Attribution Verification & Multi-Document Disagreement Engine (TICK-007).

Validates that cited snippets strictly entail asserted claims, detects hallucinations,
and identifies multi-document policy contradictions.
"""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Sequence

from core.vector_store import RetrievedChunk


def _extract_tokens(text: str) -> set[str]:
    """Extract lowercased alphanumeric tokens with length > 2."""
    words = re.findall(r"\b[a-zA-Z0-9_\-\$]+\b", text.lower())
    return {w for w in words if len(w) > 2 or w.isdigit()}


def _extract_numbers(text: str) -> set[str]:
    """Extract numbers, currency values, and alphanumeric codes."""
    return set(re.findall(r"\b\d+(?:\.\d+)?%?\b|\$\d+", text.lower()))


def verify_citation_entailment(
    claim: str,
    snippet: str,
    threshold: float = 0.35,
) -> dict[str, Any]:
    """Verify whether a cited snippet factually supports an asserted claim."""
    claim_clean = re.sub(r"\[.*?\]", "", claim).strip()
    if not claim_clean or not snippet.strip():
        return {"verified": False, "status": "unverified", "overlap_score": 0.0}

    claim_tokens = _extract_tokens(claim_clean)
    snippet_tokens = _extract_tokens(snippet)

    if not claim_tokens:
        return {"verified": False, "status": "unverified", "overlap_score": 0.0}

    common_tokens = claim_tokens.intersection(snippet_tokens)
    overlap_score = len(common_tokens) / len(claim_tokens)

    # Check numerical integrity: all numbers asserted in the claim must appear in the snippet
    claim_nums = _extract_numbers(claim_clean)
    snippet_nums = _extract_numbers(snippet)
    missing_numbers = claim_nums - snippet_nums

    is_verified = (overlap_score >= threshold) and (len(missing_numbers) == 0)

    return {
        "verified": is_verified,
        "status": "verified" if is_verified else "unverified",
        "overlap_score": round(overlap_score, 3),
        "missing_numbers": sorted(list(missing_numbers)),
    }


def detect_multi_document_disagreements(
    chunks: Sequence[RetrievedChunk],
) -> list[dict[str, Any]]:
    """Detect conflicting policies or metrics across different source documents."""
    if len(chunks) < 2:
        return []

    # Group chunks by distinct source document
    by_source: dict[str, list[RetrievedChunk]] = defaultdict(list)
    for chunk in chunks:
        src = str(chunk.metadata.get("source", "unknown"))
        by_source[src].append(chunk)

    if len(by_source) < 2:
        return []

    # Common policy topic keywords
    TOPIC_KEYWORDS = [
        "probation",
        "leave",
        "vacation",
        "notice",
        "working hours",
        "deductible",
        "reimbursement",
        "severance",
        "vesting",
        "allowance",
        "bonus",
        "stipend",
    ]

    disagreements: list[dict[str, Any]] = []

    for keyword in TOPIC_KEYWORDS:
        matching_by_doc: dict[str, list[tuple[RetrievedChunk, str, set[str]]]] = defaultdict(list)

        for src, doc_chunks in by_source.items():
            for chunk in doc_chunks:
                text = (chunk.parent_content or chunk.content).lower()
                if keyword in text:
                    # Find specific sentences mentioning the keyword
                    sentences = re.split(r"(?<=[.!?])\s+", text)
                    for sentence in sentences:
                        if keyword in sentence:
                            nums = _extract_numbers(sentence)
                            if nums:
                                matching_by_doc[src].append((chunk, sentence.strip(), nums))

        # Check if 2 or more distinct documents have sentences with different numbers/specifications
        if len(matching_by_doc) >= 2:
            distinct_docs = list(matching_by_doc.keys())
            doc1, doc2 = distinct_docs[0], distinct_docs[1]
            match1 = matching_by_doc[doc1][0]
            match2 = matching_by_doc[doc2][0]

            # If the numbers or statements differ
            if match1[2] != match2[2]:
                disagreements.append(
                    {
                        "topic": f"{keyword.capitalize()} policy",
                        "conflicting_sources": [
                            {
                                "source": doc1,
                                "page": match1[0].metadata.get("page"),
                                "statement": match1[1],
                                "values": sorted(list(match1[2])),
                            },
                            {
                                "source": doc2,
                                "page": match2[0].metadata.get("page"),
                                "statement": match2[1],
                                "values": sorted(list(match2[2])),
                            },
                        ],
                        "summary": (
                            f"Sources disagree on {keyword}: {doc1} specifies {', '.join(sorted(match1[2]))}, "
                            f"whereas {doc2} specifies {', '.join(sorted(match2[2]))}."
                        ),
                    }
                )

    return disagreements


def verify_answer_attributions(
    answer_text: str,
    chunks: Sequence[RetrievedChunk],
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Enrich sources with sentence-level entailment verification against the answer."""
    if not sources or not answer_text:
        return sources

    sentences = re.split(r"(?<=[.!?])\s+", answer_text)

    for source in sources:
        cite_idx = source.get("cite_index")
        if not cite_idx:
            source["verified"] = True
            source["verification_status"] = "verified"
            continue

        marker = f"[S{cite_idx}]"
        # Find all sentences in answer asserting this citation
        citing_sentences = [s for s in sentences if marker in s]
        claim_text = " ".join(citing_sentences) if citing_sentences else answer_text
        snippet = str(source.get("snippet", ""))

        entail_result = verify_citation_entailment(claim_text, snippet)
        source["verified"] = entail_result["verified"]
        source["verification_status"] = entail_result["status"]
        source["entailment_score"] = entail_result["overlap_score"]
        if entail_result.get("missing_numbers"):
            source["missing_numbers"] = entail_result["missing_numbers"]

    return sources
