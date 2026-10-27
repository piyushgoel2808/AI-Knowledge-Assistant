# TICK-005: Cross-Encoder Reranking & Deterministic Abstention Gate

- **Status:** COMPLETED & VERIFIED
- **User Story:** US-004, US-006 (Deterministic Abstention on Unknown Questions)
- **Dependencies:** TICK-004

## Problem
Cosine distance often returned false-positive candidates with medium scores for completely out-of-scope queries (e.g. asking corporate revenue on an HR handbook), tempting LLMs to fabricate answers.

## Solution Implemented
1. Created `core/reranker.py`:
   - `CrossEncoderReranker`: Utilizes cross-attention (`BAAI/bge-reranker-base`) with numerically stable sigmoid scaling.
   - `should_abstain`: Deterministically flags refusal when top rerank score falls below threshold $\tau = 0.30$.
2. Updated `core/rag_engine.py`:
   - Retrieves candidates (top 3x) and executes cross-encoder reranking.
   - Deterministic fast-path: If `should_abstain` is True, returns an immediate grounded refusal without calling the LLM, achieving refusal response times in <150ms with 0 token waste.

## Files Touched
- `core/reranker.py` (Created)
- `core/vector_store.py` (Updated `RetrievedChunk.rerank_score`)
- `core/rag_engine.py` (Updated)
- `tests/unit/test_reranker_and_abstention.py` (Created)

## Verification
`PYTHONPATH=. .venv/bin/pytest -v tests/unit/test_reranker_and_abstention.py` (4 passed).
