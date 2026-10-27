# End-to-End Data Flow

This document traces the complete lifecycle of data through the **AI Knowledge Assistant** across two primary production pipelines:
1. **Asynchronous Document Ingestion, Layout Parsing & Hierarchical Indexing Pipeline**
2. **Context-Aware Query Condensation, Two-Stage Hybrid Retrieval, Grounding & Verification Pipeline**

---

## 1. Asynchronous Document Ingestion Pipeline

When a user uploads files or triggers the sample document loader, the system decouples upload acceptance from heavy ingestion processing. A background worker streams pages, extracts Markdown tables, splits into parent/child chunks, and updates an in-memory task registry polled by the UI.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as React Frontend (:5174)
    participant API as FastAPI Gateway (:8001)
    participant TaskReg as Task Registry (core/task_registry.py)
    participant Worker as Background Ingestion Worker
    participant Ingest as Layout Parser (core/ingestion.py)
    participant Chunker as Hierarchical Chunker (core/hierarchical_chunker.py)
    participant DocStore as ParentDocStore (data/docstore.sqlite3)
    participant Chroma as ChromaDB (data/chroma/)
    participant BM25 as BM25 Index (core/hybrid_search.py)

    User->>UI: Select PDF / DOCX / TXT & click "Upload Documents"
    UI->>API: POST /documents/upload (multipart/form-data)
    API->>API: Save raw files to data/uploads/
    API->>TaskReg: create_task(files) -> IngestionTask (status='queued')
    API->>Worker: Enqueue background ingestion job
    API-->>UI: 202 Accepted {job_id: "task_123", status: "queued"}

    loop Every 400ms polling
        UI->>API: GET /documents/tasks/task_123
        API->>TaskReg: get_task("task_123")
        TaskReg-->>UI: 200 OK {stage: "parsing", progress_percent: 35.0}
        UI->>User: Animate progress bar & stage badge
    end

    Worker->>TaskReg: update_task(stage="parsing", progress_percent=20.0)
    Worker->>Ingest: ingest_file(file_path)
    Ingest->>Ingest: PyMuPDF bounding-box column sorting
    Ingest->>Ingest: Extract structured Markdown tables
    Ingest->>Ingest: Prune recurring running headers & footers
    Ingest-->>Worker: List of structured Document objects

    Worker->>TaskReg: update_task(stage="chunking", progress_percent=55.0)
    Worker->>Chunker: split_documents(documents)
    Chunker-->>Worker: ParentChunks (~1200t) + ChildChunks (~250t)

    Worker->>DocStore: upsert_parents(parent_chunks) (SQLite)
    Worker->>TaskReg: update_task(stage="embedding", progress_percent=80.0)
    Worker->>Chroma: upsert(child_chunks with parent_id metadata)
    Worker->>BM25: build(active_corpus) (In-Memory Inverted Index)

    Worker->>TaskReg: update_task(status="completed", progress_percent=100.0)
    UI->>API: GET /documents/tasks/task_123
    API-->>UI: 200 OK {status: "completed", progress_percent: 100.0, stored_chunks: 48}
    UI->>User: Display success toast & switch to Workspace
```

---

## 2. Grounded Conversational RAG Query & Attribution Pipeline

The query lifecycle solves the conversational pronoun problem, lexical blindness, and attribution fabrication through a 7-step pipeline:

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as React Studio (:5174)
    participant API as FastAPI Router (:8001)
    participant QC as Query Condenser (core/query_condenser.py)
    participant Chroma as ChromaDB ANN (384-dim Cosine)
    participant BM25 as Lexical Search (BM25Okapi)
    participant RRF as Reciprocal Rank Fusion (k=60)
    participant Rerank as Cross-Encoder (bge-reranker-base)
    participant DocStore as ParentDocStore (SQLite)
    participant LLM as Ollama (gemma4:e4b)
    participant Verifier as Attribution Verifier (core/attribution_verifier.py)

    User->>UI: Ask follow-up: "Does it apply to contractors?"
    UI->>API: POST /chat {question, history, top_k: 5}
    
    API->>QC: condense_query(question, history)
    QC-->>API: Standalone Query: "Does parental leave apply to contractors?"

    par Dual Candidate Retrieval
        API->>Chroma: Dense query(standalone_query, n_results=15)
        Chroma-->>API: Dense candidate list [c2, c7, c1, ...]
    and
        API->>BM25: Sparse search(standalone_query, top_k=15)
        BM25-->>API: Lexical candidate list [c1, c4, c2, ...]
    end

    API->>RRF: reciprocal_rank_fusion([dense_ids, sparse_ids], k=60)
    RRF-->>API: Unified candidate ranking

    API->>Rerank: rerank(standalone_query, candidate_chunks)
    Rerank-->>API: Cross-attention scored chunks [c1 (0.92), c2 (0.78), ...]

    alt Top Rerank Score < 0.30 (Deterministic Abstention Gate)
        API-->>UI: Grounded refusal: "I could not find the answer in the uploaded documents." (refusal=True, <150ms)
    else Top Rerank Score >= 0.30
        API->>DocStore: Fetch rich parent contexts by parent_id
        DocStore-->>API: Full 1200-token section context
        API->>LLM: generate(system_prompt, prompt with [S1], [S2] markers)
        LLM-->>API: Raw response: "Contractors are not eligible for parental leave [S1]."
        
        API->>Verifier: verify_answer_attributions(answer, chunks, sources)
        Verifier->>Verifier: Sentence-to-snippet token overlap & number audit
        Verifier-->>API: Enriched sources (verified=True, rerank_score=0.92)
        
        API->>Verifier: detect_multi_document_disagreements(chunks)
        Verifier-->>API: Contradiction report (disagreements=[...])

        API-->>UI: 200 OK {answer, sources, refusal: false, standalone_query, has_disagreements}
        UI->>User: Display answer, verified pill badges, and highlight citations
        opt Voice output enabled
            UI->>User: Read clean text aloud via Web Speech API (TTS)
        end
    end
```
