# TICK-001: Asynchronous Ingestion & Task Progress Tracking

- **Status:** COMPLETED & VERIFIED
- **User Story:** US-001 (Large File Ingestion with Real-Time Progress)
- **Dependencies:** None

## Problem
Synchronous HTTP uploads blocked FastAPI's event loop for 35+ seconds on large PDFs, causing 504 Gateway Timeouts, browser freezes, and zero progress feedback.

## Solution Implemented
1. Created `core/task_registry.py` with an in-memory `TaskRegistry` supporting `queued`, `processing`, `completed`, and `failed` lifecycles.
2. Updated `backend/main.py`: `POST /documents/upload` accepts files, enqueues background processing, and returns HTTP 202 with `job_id` within <50ms.
3. Added `GET /documents/tasks/{job_id}` endpoint for non-blocking client polling.
4. Updated frontend `api.ts` and `App.tsx` with polling loop and real-time stage progress bar.

## Files Touched
- `core/task_registry.py` (Created)
- `backend/main.py` (Updated)
- `frontend/src/api.ts` (Updated)
- `frontend/src/App.tsx` (Updated)
- `tests/integration/test_async_ingestion.py` (Created)

## Verification
`PYTHONPATH=. .venv/bin/pytest -v tests/integration/test_async_ingestion.py` (3 passed).
