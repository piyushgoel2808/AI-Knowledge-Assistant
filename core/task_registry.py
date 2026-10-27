"""Thread-safe task registry for asynchronous document ingestion jobs."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any


@dataclass
class IngestionTask:
    job_id: str
    status: str  # "queued" | "processing" | "completed" | "failed"
    stage: str  # "queued" | "parsing" | "chunking" | "embedding" | "completed" | "failed"
    progress_percent: float
    files: list[str]
    processed_pages: int = 0
    total_pages: int = 0
    stored_chunks: int = 0
    error: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskRegistry:
    """In-memory thread-safe registry tracking asynchronous background ingestion tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, IngestionTask] = {}
        self._lock = Lock()

    def create_task(self, files: list[str]) -> IngestionTask:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = time.time()
        task = IngestionTask(
            job_id=job_id,
            status="queued",
            stage="queued",
            progress_percent=0.0,
            files=files,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._tasks[job_id] = task
        return task

    def get_task(self, job_id: str) -> IngestionTask | None:
        with self._lock:
            return self._tasks.get(job_id)

    def update_task(
        self,
        job_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        progress_percent: float | None = None,
        processed_pages: int | None = None,
        total_pages: int | None = None,
        stored_chunks: int | None = None,
        error: str | None = None,
    ) -> IngestionTask | None:
        with self._lock:
            task = self._tasks.get(job_id)
            if not task:
                return None

            if status is not None:
                task.status = status
            if stage is not None:
                task.stage = stage
            if progress_percent is not None:
                task.progress_percent = round(progress_percent, 1)
            if processed_pages is not None:
                task.processed_pages = processed_pages
            if total_pages is not None:
                task.total_pages = total_pages
            if stored_chunks is not None:
                task.stored_chunks = stored_chunks
            if error is not None:
                task.error = error

            task.updated_at = time.time()
            return task


# Global singleton registry instance
task_registry = TaskRegistry()
