# TICK-006: Context-Aware Conversational Query Condensation & History Buffer

- **Status:** COMPLETED & VERIFIED
- **User Story:** US-003 (Conversational Multi-Turn Follow-Up Resolution)
- **Dependencies:** TICK-004

## Problem
In conversational threads, follow-up turns containing pronouns or ellipsis (e.g., Turn 1: "Parental leave for Software Engineers", Turn 2: "Does it apply to contractors?") failed retrieval because raw pronouns yielded irrelevant document chunks.

## Solution Implemented
1. Created `core/query_condenser.py`:
   - `condense_query`: Fast LLM context condenser using few-shot prompt templates to resolve conversational pronouns into fully qualified standalone retrieval queries.
   - Zero-overhead bypass when chat history is empty.
2. Updated `core/rag_engine.py`:
   - Generates `standalone_query` prior to vector and BM25 search.
   - Uses rewritten query for indexing & reranking, while passing original conversational dialogue to the final synthesis prompt.
   - Returns `standalone_query` in API response for auditability and UI display.

## Files Touched
- `core/query_condenser.py` (Created)
- `core/rag_engine.py` (Updated)
- `tests/unit/test_query_condenser.py` (Created)

## Verification
`PYTHONPATH=. .venv/bin/pytest -v tests/unit/test_query_condenser.py` (4 passed).
