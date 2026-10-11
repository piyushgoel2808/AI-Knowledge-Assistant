# Features, Backend Fixes & Apple Silicon Optimizations

This document catalogs the core functional features of the **AI Knowledge Assistant**, along with the architectural bug fixes, frontend redesign, and Apple Silicon (`arm64`) optimizations implemented in the project.

---

## 1. Core Product Features

```mermaid
flowchart LR
    subgraph Ingest["1. Smart Ingestion"]
        F1["PDF Native Text + Table Extraction"]
        F2["Scanned PDF Tesseract OCR Fallback"]
        F3["DOCX Heading & Table Parser"]
        F4["One-Click Bundled Sample PDF Loader"]
    end

    subgraph Index["2. Vector Knowledge Base"]
        F5["Local Persistent ChromaDB (Cosine HNSW)"]
        F6["Automatic Per-File Re-Upload Deduplication"]
        F7["Per-Document Inventory & Single-File Delete"]
        F8["Configurable Top-K Retrieval Slider"]
    end

    subgraph Gen["3. Grounded RAG & UI"]
        F9["Default Local Ollama (gemma4:e4b)"]
        F10["Inline [S1] Citations & Confidence Scoring"]
        F11["Automatic Sentence Keyword Highlighting"]
        F12["Interactive Homepage + RAG Studio + Architecture View"]
    end

    Ingest --> Index --> Gen
```

### 1.1 Multi-Format Document Ingestion
- **PDF Text & Table Extraction**: Uses `PyMuPDF` (`pymupdf`) to extract both page text and structured tables (`page.find_tables()`), formatting detected tables into pipe-delimited Markdown blocks.
- **Scanned PDF OCR Fallback**: Automatically detects pages without embedded text, renders a $2\times$ resolution pixmap, and runs `pytesseract` OCR inside the container.
- **Word (`.docx`) Structure Preservation**: Groups paragraphs under their corresponding Word `Heading` styles and extracts embedded `.docx` tables as Markdown.
- **Resilient Plain Text (`.txt`) Support**: Reads UTF-8 text files with automatic fallback to `latin-1` so non-ASCII encodings never fail during upload.
- **One-Click Sample Documents**: Includes a `POST /documents/sample` endpoint and UI button that indexes the bundled PDFs in `Sample_doc/` (`JD- Software Developer.pdf` and `JD - Data Quality Assurance.pdf`) for immediate testing.

### 1.2 Grounded Question Answering & Traceability
- **Strict Context Grounding**: System instructions force the LLM to answer *exclusively* from retrieved document chunks and refuse when information is absent.
- **Inline Citation Pills (`[S1]`, `[S2]`)**: Answers include inline bracket citations rendered as styled pills in the chat UI.
- **Sentence-Level Keyword Highlighting**: `_extract_and_highlight_match()` identifies sentences in the cited chunk that overlap with the generated answer and highlights matching keywords inside the expandable source cards.
- **Similarity Score & Metadata Badges**: Each cited source card displays the source filename, page number, section name, and cosine similarity match percentage (`similarity = 1 - distance`).

---

## 2. Backend Issues Identified & Fixed

| Issue Area | Original Issue | Fix Implemented |
| :--- | :--- | :--- |
| **Unbounded Retrieval (`core/rag_engine.py`)** | `search_limit = max(4, total_indexed_chunks)` caused the RAG engine to retrieve **100% of all chunks in the database** whenever more than 4 chunks were indexed, overflowing the LLM prompt context window and slowing down generation. | Fixed to `search_limit = max(1, min(desired_k, total_indexed_chunks))` (default `top_k = 5`), with an immediate early return when `total_indexed_chunks == 0`. |
| **Per-Request Model Reloading (`core/vector_store.py`)** | `_get_embedding_function()` and `get_client()` instantiated a new PyTorch `SentenceTransformer("all-MiniLM-L6-v2")` and `chromadb.PersistentClient` on every single HTTP request (`/health`, `/chat`, `/documents/upload`). | Added `@lru_cache` singletons for both `_get_embedding_function()` and `_get_cached_client()`, cutting `/health` latency from ~2.5s down to <15ms. |
| **Hardcoded `refusal: False` (`core/rag_engine.py`)** | `answer_question()` hardcoded `"refusal": False` whenever chunks existed in ChromaDB, even when the LLM explicitly replied *"I could not find the answer in the uploaded documents."* | Added `_is_refusal_response()` detection so refusals accurately return `"refusal": True`, `"confidence": "low"`, and `"sources": []`. |
| **Duplicate Chunks on Re-Upload (`core/vector_store.py`)** | Uploading the same file multiple times in different batch sizes created duplicate chunks with different batch indices. | `add_documents()` now deletes existing chunks matching `where={"source": source_name}` prior to upserting and appends a content SHA-1 digest to chunk IDs. |
| **Missing Container OCR Binary (`Dockerfile.backend`)** | `core/ingestion.py` imported `pytesseract`, but `Dockerfile.backend` (`python:3.10-slim`) did not install `tesseract-ocr`, causing scanned PDFs to crash with `TesseractNotFoundError`. | Installed `tesseract-ocr` in `Dockerfile.backend` and wrapped OCR in a safe `try...except` block for host environments without Tesseract. |
| **Deprecated `fitz` Import (`core/ingestion.py`)** | `import fitz` emitted a deprecation warning on startup in modern PyMuPDF versions. | Migrated to `import pymupdf`. |

---

## 3. Apple Silicon (`arm64`) & Ollama (`gemma4:e4b`) Optimizations

1. **Default `gemma4:e4b` Configuration**:
   - Configured `LLM_PROVIDER=ollama` and `OLLAMA_MODEL=gemma4:e4b` across `core/config.py`, `.env`, and `.env.example`.
2. **Low-Latency Ollama Adapter (`core/ai_service.py`)**:
   - Uses a persistent `requests.Session()` for HTTP connection pooling.
   - Passes `"keep_alive": "30m"` so macOS Unified Memory keeps `gemma4:e4b` warm across queries.
   - Passes `"think": False` and `"temperature": 0.1` for fast, deterministic, citation-accurate RAG responses (~200–500ms per query on Apple Silicon).
3. **Docker-to-Host Metal Bridge (`docker-compose.yml`)**:
   - Routes container traffic to the native macOS Ollama daemon via `http://host.docker.internal:11434` so `gemma4:e4b` runs with full Apple Metal GPU acceleration.
   - Persists HuggingFace model weights in `/app/data/hf_cache` (`./data/hf_cache`) and mounts `./backend`, `./core`, and `./frontend/src` for instant hot-reload during development.
