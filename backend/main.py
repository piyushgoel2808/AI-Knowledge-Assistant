"""FastAPI backend for the AI Knowledge Assistant React frontend."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.config import AppConfig, load_config
from core.ingestion import collect_supported_files, ingest_file, is_supported_file
from core.rag_engine import ChatMessage, answer_question
from core.task_registry import task_registry
from core.vector_store import (
    add_documents,
    clear_collection,
    count_documents,
    delete_document,
    list_indexed_documents,
)


app = FastAPI(
    title="AI Knowledge Assistant API",
    version="2.0.0",
    description="Local RAG API powered by ChromaDB and Ollama (gemma4:e4b) / Gemini / Groq",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: list[dict[str, Any]] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=20)


@app.get("/health")
def health() -> dict[str, Any]:
    config = load_config()
    docs = list_indexed_documents(config=config)
    return {
        "status": "ok",
        "provider": config.llm_provider,
        "model": config.active_model,
        "stored_chunks": count_documents(config=config),
        "document_count": len(docs),
        "documents": docs,
    }


@app.get("/documents")
def get_documents() -> dict[str, Any]:
    config = load_config()
    docs = list_indexed_documents(config=config)
    return {
        "documents": docs,
        "document_count": len(docs),
        "stored_chunks": count_documents(config=config),
    }


def execute_ingestion_job(job_id: str, file_paths: list[Path], config: AppConfig) -> None:
    """Asynchronous background worker function to parse, chunk, and embed documents."""
    try:
        task_registry.update_task(job_id, status="processing", stage="parsing", progress_percent=15.0)
        all_documents = []
        total_files = len(file_paths)

        for idx, file_path in enumerate(file_paths, start=1):
            chunks = ingest_file(file_path, config=config)
            all_documents.extend(chunks)
            file_progress = 15.0 + (idx / total_files) * 45.0
            task_registry.update_task(
                job_id,
                stage="chunking",
                progress_percent=file_progress,
                stored_chunks=len(all_documents),
            )

        if not all_documents:
            task_registry.update_task(
                job_id,
                status="failed",
                stage="failed",
                error="No extractable text found in uploaded files.",
            )
            return

        task_registry.update_task(job_id, stage="embedding", progress_percent=75.0)
        add_documents(all_documents, config=config)

        task_registry.update_task(
            job_id,
            status="completed",
            stage="completed",
            progress_percent=100.0,
            stored_chunks=count_documents(config=config),
        )
    except Exception as exc:
        task_registry.update_task(
            job_id,
            status="failed",
            stage="failed",
            error=str(exc),
        )


@app.post("/documents/upload", status_code=202)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
) -> JSONResponse:
    config = load_config()
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    upload_dir = config.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)

    accepted_files: list[str] = []
    saved_paths: list[Path] = []

    for uploaded_file in files:
        filename = uploaded_file.filename or ""
        if not is_supported_file(filename):
            continue

        safe_name = Path(filename).name
        target_path = upload_dir / safe_name
        with target_path.open("wb") as target_file:
            shutil.copyfileobj(uploaded_file.file, target_file)

        accepted_files.append(target_path.name)
        saved_paths.append(target_path)

    if not accepted_files:
        raise HTTPException(
            status_code=400,
            detail="No supported files found. Supported formats: PDF, DOCX, TXT.",
        )

    task = task_registry.create_task(accepted_files)
    background_tasks.add_task(execute_ingestion_job, task.job_id, saved_paths, config)

    return JSONResponse(
        status_code=202,
        content={
            "job_id": task.job_id,
            "status": "queued",
            "message": "Documents uploaded and ingestion queued.",
            "files": accepted_files,
        },
    )


@app.get("/documents/tasks/{job_id}")
def get_task_status(job_id: str) -> dict[str, Any]:
    """Poll the status and progress of an asynchronous document ingestion job."""
    task = task_registry.get_task(job_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {job_id} not found.")
    return task.to_dict()


@app.post("/documents/sample")
def load_sample_documents() -> dict[str, Any]:
    """Index the bundled sample PDF documents from Sample_doc/ for quick exploration."""
    config = load_config()
    sample_dir = config.sample_doc_dir
    if not sample_dir.exists():
        raise HTTPException(status_code=404, detail="Sample_doc directory not found.")

    sample_files = collect_supported_files(sorted(sample_dir.iterdir()))
    if not sample_files:
        raise HTTPException(status_code=404, detail="No sample documents found in Sample_doc/.")

    upload_dir = config.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)

    accepted_files: list[str] = []
    all_documents = []
    for sample_path in sample_files:
        target_path = upload_dir / sample_path.name
        shutil.copy2(sample_path, target_path)
        accepted_files.append(target_path.name)
        all_documents.extend(ingest_file(target_path, config=config))

    add_documents(all_documents, config=config)
    docs = list_indexed_documents(config=config)

    return {
        "message": f"Loaded {len(accepted_files)} sample document(s).",
        "files": accepted_files,
        "chunks_indexed": len(all_documents),
        "stored_chunks": count_documents(config=config),
        "documents": docs,
    }


@app.delete("/documents/{source_name}")
def remove_document(source_name: str) -> dict[str, Any]:
    config = load_config()
    safe_name = Path(source_name).name
    removed_chunks = delete_document(safe_name, config=config)

    upload_file = config.upload_dir / safe_name
    if upload_file.exists():
        try:
            upload_file.unlink()
        except OSError:
            pass

    docs = list_indexed_documents(config=config)
    return {
        "message": f"Removed '{safe_name}' ({removed_chunks} chunks).",
        "removed_chunks": removed_chunks,
        "stored_chunks": count_documents(config=config),
        "documents": docs,
    }


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    config = load_config()
    chat_history = [
        ChatMessage(role=str(item.get("role", "user")), content=str(item.get("content", "")))
        for item in request.history
    ]

    try:
        return answer_question(
            question=request.question,
            chat_history=chat_history,
            top_k=request.top_k,
            config=config,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/documents/clear")
def clear_documents() -> dict[str, Any]:
    config = load_config()
    clear_collection(config=config)
    if config.upload_dir.exists():
        for file_path in config.upload_dir.iterdir():
            if file_path.is_file():
                try:
                    file_path.unlink()
                except OSError:
                    pass
    return {
        "message": "Vector store cleared.",
        "stored_chunks": 0,
        "documents": [],
    }
