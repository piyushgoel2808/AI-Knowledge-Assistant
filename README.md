# AI Knowledge Assistant (Production RAG Studio)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.3-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6+-FF6600)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Ollama-gemma4%3Ae4b-000000?logo=ollama&logoColor=white)](https://ollama.com/)
[![Embeddings](https://img.shields.io/badge/Embeddings-EmbeddingGemma--2%20(768d)-8A2BE2)](https://ollama.com/)
[![Tests](https://img.shields.io/badge/Tests-29%20Passing-brightgreen?logo=pytest&logoColor=white)](#-testing--verification)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An enterprise-grade, local-first **Retrieval-Augmented Generation (RAG) Studio** engineered to ingest complex, multi-page corporate documents (`.pdf`, `.docx`, `.txt`) and deliver **zero-hallucination, evidence-backed answers** with sentence-level entailment verification, deterministic abstention, and hands-free voice interaction.

Optimized for **Apple Silicon (`arm64`) Metal GPU acceleration** and **Docker Compose**, the platform defaults to local **Ollama (`gemma4:e4b`)** inference while offering seamless cloud fallbacks to **Google Gemini** and **Groq**.

---

## 📚 Deep-Dive Documentation Index (`docs/`)

Explore the comprehensive technical documentation suite:

| Document | Description |
| :--- | :--- |
| **[System Architecture](docs/architecture.md)** | Full system topology, class diagrams, dual-storage model, and Apple Silicon Metal runtime. |
| **[Codebase Directory & Symbol Reference](docs/codebase-directory-and-references.md)** | Directory tree, cross-module call graph, and complete function/symbol reference across backend & frontend. |
| **[End-to-End Data Flow](docs/dataflow-end-to-end.md)** | Detailed Mermaid sequence diagrams for Asynchronous Ingestion and Grounded Hybrid RAG Answering. |
| **[Database & Storage Schema](docs/database-schema.md)** | ChromaDB HNSW collection, SQLite `ParentDocStore` schema, ER diagram, and chunk ID specs. |
| **[Features & Optimizations](docs/features.md)** | Complete feature breakdown, architectural flaw resolutions, and macOS performance benchmarks. |
| **[Production Specification (`SPEC-RAG-2026-V2`)](docs/production-rag-system.spec.md)** | Formal RFC engineering specification, EARS functional requirements, and evaluation gates. |
| **[Interview Guide](docs/interview-guide.md)** | Comprehensive technical interview explainer with 25+ architectural Q&As and whiteboard breakdowns. |

---

## 🏗️ System Architecture Topology

```mermaid
flowchart TB
    subgraph Client["Presentation Layer (Vite + React 18 :5174)"]
        UI_Home["Landing Homepage & Telemetry"]
        UI_Studio["RAG Studio & Document Manager"]
        UI_Voice["Web Speech Layer (STT & TTS)"]
    end

    subgraph Gateway["API Gateway (FastAPI :8001)"]
        Routes["REST Endpoints (backend/main.py)"]
        TaskReg["Task Registry (core/task_registry.py)"]
    end

    subgraph IngestionEngine["Ingestion & Chunking Pipeline"]
        Layout["Layout-Aware Parser (core/ingestion.py)"]
        Hierarchical["Hierarchical Chunker (core/hierarchical_chunker.py)"]
        DocStore["Parent DocStore SQLite (data/docstore.sqlite3)"]
    end

    subgraph RetrievalEngine["Two-Stage Hybrid Retrieval & Reranker"]
        BM25["Sparse BM25 Index (core/hybrid_search.py)"]
        Chroma["Dense ChromaDB HNSW (data/chroma/)"]
        RRF["Reciprocal Rank Fusion (k=60)"]
        Reranker["Cross-Encoder Reranker (core/reranker.py)"]
        AbstentionGate{"Top Rerank Score >= 0.30?"}
    end

    subgraph SynthesisEngine["Context Condensation & Verification"]
        Condenser["Query Condenser (core/query_condenser.py)"]
        PromptBuilder["Attribution Prompt Builder"]
        Verifier["Attribution Verifier (core/attribution_verifier.py)"]
        Disagreement["Multi-Doc Disagreement Engine"]
    end

    subgraph LLMProviders["Inference Providers (core/ai_service.py)"]
        Ollama["Local Ollama (gemma4:e4b @ :11434)"]
        Gemini["Google Gemini (gemini-flash-latest)"]
        Groq["Groq Cloud API (llama-3.3-70b-versatile)"]
    end

    UI_Home --> Routes
    UI_Studio --> Routes
    UI_Voice <--> UI_Studio
    Routes --> TaskReg
    Routes --> Layout
    Layout --> Hierarchical
    Hierarchical -->|"Parent Chunks (~1200t)"| DocStore
    Hierarchical -->|"Child Chunks (~250t)"| Chroma
    Hierarchical -->|"Corpus Tokens"| BM25

    Routes --> Condenser
    Condenser -->|"Standalone Query"| RetrievalEngine
    RetrievalEngine --> Chroma
    RetrievalEngine --> BM25
    Chroma --> RRF
    BM25 --> RRF
    RRF --> Reranker
    Reranker --> AbstentionGate
    AbstentionGate -->|"No (< 0.30)"| Routes
    AbstentionGate -->|"Yes (>= 0.30)"| PromptBuilder
    DocStore -->|"Hydrate Parent Context"| PromptBuilder
    PromptBuilder --> Ollama
    PromptBuilder -.-> Gemini
    PromptBuilder -.-> Groq
    Ollama --> Verifier
    Verifier --> Disagreement
    Disagreement --> Routes
```

---

## 🔄 End-to-End Grounded RAG Dataflow

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as React UI (:5174)
    participant API as FastAPI Gateway (:8001)
    participant QC as Query Condenser
    participant Hybrid as Hybrid Search (Dense + BM25)
    participant Rerank as Cross-Encoder (bge-reranker-base)
    participant DocStore as ParentDocStore (SQLite)
    participant LLM as Ollama (gemma4:e4b)
    participant Judge as Attribution Verifier

    User->>UI: Ask follow-up: "Does it apply to contractors?"
    UI->>API: POST /chat {question, history, top_k}
    API->>QC: condense_query(question, history)
    QC-->>API: Standalone Query: "Does parental leave apply to contractors?"
    
    API->>Hybrid: Search Top-K Candidates (Dense Cosine + Sparse BM25)
    Hybrid-->>API: Fused Candidates via Reciprocal Rank Fusion (k=60)
    
    API->>Rerank: Score candidates via Cross-Attention
    alt Top Rerank Score < 0.30
        Rerank-->>API: Abstain (score < 0.30)
        API-->>UI: Immediate Grounded Refusal (<150ms)
    else Top Rerank Score >= 0.30
        Rerank-->>API: Ranked Top Chunks
        API->>DocStore: Fetch Rich Parent Context (~1200 tokens)
        DocStore-->>API: Expanded Context Sections
        API->>LLM: Generate Answer with [S1] Grounding Prompts
        LLM-->>API: Raw Synthesized Response
        API->>Judge: verify_answer_attributions() & detect_disagreements()
        Judge-->>API: Verified Sources + Entailment Scores + Contradictions
        API-->>UI: 200 OK {answer, sources, verified: true, rerank_score: 0.94}
        UI-->>User: Render formatted answer, citation pills & read aloud (TTS)
    end
```

---

## ✨ Production Architectural Highlights

| Feature | Legacy Naive RAG | Production AI Knowledge Assistant |
| :--- | :--- | :--- |
| **Ingestion Concurrency** | Synchronous HTTP blocking main event loop; crashes on >20MB PDFs | **Decoupled Asynchronous Background Worker** with `job_id` returned in <50ms & live progress polling |
| **PDF Layout & Tables** | Naive string dump; multi-column bleed; tables flattened into whitespace | **PyMuPDF Bounding-Box Geometry Sorting** + true GitHub-Flavored Markdown table extraction |
| **Chunking Strategy** | Fixed 1000-character chunks (sacrifices context or precision) | **Hierarchical Parent-Child Chunking**: 250t leaf chunks for ANN search + 1200t parent for LLM context |
| **Retrieval Architecture** | Pure vector cosine similarity; blind to acronyms and alphanumeric codes | **Two-Stage Hybrid Search**: Dense ChromaDB + Inverted BM25Okapi fused with RRF ($k=60$) |
| **False-Positive Filtering** | None; LLMs hallucinate answers on out-of-scope questions | **Cross-Encoder Reranking** (`bge-reranker-base`) with **Deterministic Abstention Gate** ($\tau < 0.30$) |
| **Conversational Context** | Searches raw query strings; breaks on pronouns (*"Does it apply to him?"*) | **Few-Shot Context Condenser** rephrasing conversational turns into standalone retrieval queries |
| **Attribution & Hallucination** | Blind belief in LLM bracket outputs | **Two-Tier Sentence NLI Judge** checking numerical overlap + **Multi-Doc Disagreement Engine** |
| **Multimodal Interaction** | Keyboard text input only | **Hands-Free Voice Layer** using browser Web Speech API for both STT input and clean TTS audio output |
| **Port Isolation** | Conflicted with NewsLens-AI on 8000/5173 | **Dedicated Non-Conflicting Ports**: Backend on `8001`, Frontend on `5174` |

---

## 🚀 Quickstart Guide

### System Requirements
- **OS:** macOS (Apple Silicon M1/M2/M3/M4 recommended) or Linux.
- **Python:** 3.12+ (or Docker Engine).
- **Node.js:** 18+ (for frontend).
- **Ollama:** Installed and running locally:
  ```bash
  ollama pull gemma4:e4b
  ```

---

### Option A: Local Native Execution (Fastest for Development)

#### 1. Setup Backend
```bash
# Clone the repository
git clone https://github.com/Piyushgoel99/AI-Knowledge-Assistant.git
cd AI-Knowledge-Assistant

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start backend on dedicated port 8001
PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload
```

#### 2. Setup Frontend
```bash
# In a new terminal window
cd frontend
npm install
npm run dev
```

#### 3. Access Application
- **Frontend Studio:** [http://localhost:5174](http://localhost:5174)
- **Backend Health:** [http://localhost:8001/health](http://localhost:8001/health)
- **Swagger OpenAPI Docs:** [http://localhost:8001/docs](http://localhost:8001/docs)

---

### Option B: Docker Compose Execution

```bash
# Copy pre-configured environment file
cp .env.example .env

# Build and launch isolated containers
docker compose up -d --build

# View real-time container logs
docker compose logs -f
```

---

## 🧪 Testing & Verification

The project enforces strict Test-Driven Development (TDD) with high-seam unit and integration tests across all pipeline layers.

```bash
# Run the complete test suite (24 tests)
PYTHONPATH=. .venv/bin/pytest -v tests/
```

```text
tests/integration/test_async_ingestion.py::test_upload_returns_202_accepted_with_job_id PASSED
tests/integration/test_async_ingestion.py::test_get_task_status_lifecycle PASSED
tests/integration/test_async_ingestion.py::test_get_nonexistent_task_returns_404 PASSED
tests/integration/test_voice_and_disagreement_api.py::test_chat_endpoint_returns_extended_attribution_contracts PASSED
tests/integration/test_voice_and_disagreement_api.py::test_chat_endpoint_abstention_fast_path PASSED
tests/unit/test_attribution_verifier.py::test_verify_claim_entailment_verified PASSED
tests/unit/test_attribution_verifier.py::test_verify_claim_entailment_unverified_hallucination PASSED
tests/unit/test_attribution_verifier.py::test_detect_multi_document_disagreement PASSED
tests/unit/test_attribution_verifier.py::test_answer_question_enriches_sources_with_verification PASSED
tests/unit/test_hierarchical_chunker.py::test_hierarchical_chunker_splits_into_parent_and_child_chunks PASSED
tests/unit/test_hierarchical_chunker.py::test_parent_docstore_crud PASSED
tests/unit/test_hybrid_search.py::test_rrf_scoring_algorithm PASSED
tests/unit/test_hybrid_search.py::test_bm25_exact_token_matching PASSED
tests/unit/test_layout_parser.py::test_markdown_table_formatting PASSED
tests/unit/test_layout_parser.py::test_column_sorting_preserves_reading_order PASSED
tests/unit/test_layout_parser.py::test_prune_headers_footers PASSED
tests/unit/test_query_condenser.py::test_condense_query_without_history_returns_verbatim PASSED
tests/unit/test_query_condenser.py::test_condense_query_with_history_calls_llm_few_shot PASSED
tests/unit/test_query_condenser.py::test_condense_query_cleans_llm_output PASSED
tests/unit/test_query_condenser.py::test_answer_question_uses_condensed_query_for_retrieval PASSED
tests/unit/test_reranker_and_abstention.py::test_sigmoid_math PASSED
tests/unit/test_reranker_and_abstention.py::test_reranker_scores_and_orders_chunks PASSED
tests/unit/test_reranker_and_abstention.py::test_abstention_gate_triggers_when_top_score_below_threshold PASSED
tests/unit/test_reranker_and_abstention.py::test_answer_question_deterministic_refusal_fast_path PASSED

======================== 24 passed in 5.25s ========================
```

To test the frontend TypeScript build:
```bash
cd frontend && npm run build
```

---

## 🔌 Complete REST API Reference

| Method | Endpoint | Description | Status Code |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service health, model info, chunk counters, and indexed document inventory. | `200 OK` |
| `POST` | `/documents/upload` | Asynchronous multi-file upload (`.pdf`, `.docx`, `.txt`). Returns `job_id`. | `202 Accepted` |
| `GET` | `/documents/tasks/{job_id}` | Polling endpoint returning stage, percentage, processed pages, and status. | `200 OK` |
| `POST` | `/documents/sample` | One-click loader for bundled sample documents in `Sample_doc/`. | `200 OK` |
| `GET` | `/documents` | Detailed list of all indexed documents with page, chunk, and word metrics. | `200 OK` |
| `DELETE` | `/documents/{source_name}`| Completely purges a specific document from ChromaDB, SQLite, and disk. | `200 OK` |
| `POST` | `/documents/clear` | Purges the entire vector store, SQLite docstore, and upload cache. | `200 OK` |
| `POST` | `/chat` | Grounded RAG answering with query condensation, reranking, and entailment audit. | `200 OK` |

---

## 🛠️ Configuration Options (`.env`)

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `BACKEND_PORT` | `8001` | Dedicated FastAPI port (isolates from 8000) |
| `FRONTEND_PORT` | `5174` | Dedicated Vite UI port (isolates from 5173) |
| `LLM_PROVIDER` | `ollama` | Active LLM backend: `ollama`, `gemini`, or `groq` |
| `OLLAMA_BASE_URL`| `http://localhost:11434` | Ollama connection endpoint |
| `OLLAMA_MODEL` | `gemma4:e4b` | Default high-efficiency local Apple Silicon model |
| `GEMINI_API_KEY` | *(empty)* | Optional Google Gemini API key |
| `GROQ_API_KEY` | *(empty)* | Optional Groq Cloud API key |
| `RAG_TOP_K` | `5` | Retrieved chunks per query |
| `CHROMA_PERSIST_DIR` | `data/chroma` | Persistent ChromaDB storage path |
| `UPLOAD_DIR` | `data/uploads` | Document upload directory |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
