# TICK-003: Hierarchical Parent-Child Chunking & Dual Storage Architecture

- **Status:** COMPLETED & VERIFIED
- **User Story:** US-002, US-004 (Context Granularity & Verification)
- **Dependencies:** TICK-002

## Problem
Fixed-size 1000-character chunking produced a dilemma: small chunks lost broader policy section context, while large chunks diluted dense embedding vectors.

## Solution Implemented
1. Created `core/hierarchical_chunker.py`:
   - `HierarchicalChunker`: Splits documents into large parent chunks (~1200 tokens) and small child leaf chunks (~250 tokens). Each child references its `parent_id`.
   - `ParentDocStore`: SQLite-backed document store (`data/docstore.sqlite3`) preserving full parent context.
2. Updated `core/vector_store.py`: Child chunks indexed into ChromaDB; on retrieval, parent context is re-attached via `ParentDocStore`.
3. Updated `core/rag_engine.py`: Injects full parent context into LLM prompt while keeping child granularity for vector similarity search.

## Files Touched
- `core/hierarchical_chunker.py` (Created)
- `core/ingestion.py` (Updated)
- `core/vector_store.py` (Updated)
- `core/rag_engine.py` (Updated)
- `tests/unit/test_hierarchical_chunker.py` (Created)

## Verification
`PYTHONPATH=. .venv/bin/pytest -v tests/unit/test_hierarchical_chunker.py` (2 passed).
