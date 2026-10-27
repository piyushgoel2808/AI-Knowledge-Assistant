# TICK-004: Hybrid Search (Dense Vector + BM25) with Reciprocal Rank Fusion

- **Status:** COMPLETED & VERIFIED
- **User Story:** US-002, US-004 (Lexical Blindness Fix)
- **Dependencies:** TICK-003

## Problem
Pure dense cosine similarity failed to retrieve exact alphanumeric codes, form numbers (e.g., "Form 1099-NEC"), and specific acronyms.

## Solution Implemented
1. Created `core/hybrid_search.py`:
   - `BM25Index`: In-memory inverted BM25 index built on tokenized chunk corpus using `rank_bm25.BM25Okapi`.
   - `reciprocal_rank_fusion(rank_lists, k=60)`: Fuses dense ChromaDB rank list with sparse BM25 rank list using standard constant $k=60$.
2. Updated `core/vector_store.py`: `search_documents()` performs dual retrieval (dense + lexical), passes candidate IDs to RRF, and ranks exact alphanumeric matches at rank 1.

## Files Touched
- `core/hybrid_search.py` (Created)
- `core/vector_store.py` (Updated)
- `tests/unit/test_hybrid_search.py` (Created)

## Verification
`PYTHONPATH=. .venv/bin/pytest -v tests/unit/test_hybrid_search.py` (2 passed).
