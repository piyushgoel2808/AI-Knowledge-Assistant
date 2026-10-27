# TICK-007: Two-Tier Attribution Verification & Multi-Document Disagreement Engine

- **Status:** COMPLETED & VERIFIED
- **User Story:** US-004 (Verifiable Source Attribution), US-005 (Multi-Document Disagreement)
- **Dependencies:** TICK-005

## Problem
1. LLMs frequently hallucinate citation brackets `[S1]` for assertions not present in the referenced snippet.
2. When multiple documents contained conflicting clauses (e.g., US Handbook: 3 months probation; UK Handbook: 6 months probation), systems arbitrarily picked one or averaged numbers.

## Solution Implemented
1. Created `core/attribution_verifier.py`:
   - `verify_citation_entailment`: Checks sentence-level word overlap and strictly verifies that numerical figures/metrics in the claim match the cited snippet. Flags hallucinated citations as `unverified`.
   - `detect_multi_document_disagreements`: Analyzes chunks across distinct source documents to detect conflicting numbers or clauses on common policy topics.
   - `verify_answer_attributions`: Enriches source dictionaries with verification status and entailment scores.
2. Updated `core/rag_engine.py`:
   - Enforced multi-document discrepancy reporting in the system prompt.
   - Returned `has_disagreements` and structured `disagreements` list in API response.

## Files Touched
- `core/attribution_verifier.py` (Created)
- `core/rag_engine.py` (Updated)
- `tests/unit/test_attribution_verifier.py` (Created)

## Verification
`PYTHONPATH=. .venv/bin/pytest -v tests/unit/test_attribution_verifier.py` (4 passed).
