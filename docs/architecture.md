# System Architecture

This document provides a comprehensive architectural blueprint of the **AI Knowledge Assistant**, a local-first Retrieval-Augmented Generation (RAG) platform engineered for zero-hallucination document question answering and optimized for Apple Silicon (`arm64`) macOS and Docker environments.

---

## 1. High-Level System Topology

The system is structured into four decoupled layers:
1. **Presentation Layer (`frontend/`)**: React 18 + TypeScript + Vite single-page application featuring a Landing Homepage, a Split-Pane RAG Studio Workspace, and an interactive System Architecture view.
2. **API Gateway Layer (`backend/`)**: FastAPI + Uvicorn REST server managing file uploads, sample dataset loading, collection lifecycle, and conversational endpoints.
3. **Core RAG & Ingestion Engine (`core/`)**: Modular Python package responsible for multi-format document parsing (`PyMuPDF`, `python-docx`, `pytesseract`), recursive text chunking, persistent vector indexing, and strict citation-grounded prompt orchestration.
4. **Storage & Inference Layer (`data/` + LLM Providers)**: Local persistent `ChromaDB` vector storage (384-dim `all-MiniLM-L6-v2` embeddings) paired with a pluggable LLM Provider Adapter defaulting to local **Ollama (`gemma4:e4b`)** alongside **Google Gemini** and **Groq**.

```mermaid
flowchart TB
    subgraph Client["Presentation Layer (Browser :5173)"]
        UI_Home["Homepage & Live Telemetry"]
        UI_Work["RAG Workspace & Document Manager"]
        UI_Arch["System Architecture Blueprint"]
    end

    subgraph Containers["Docker Compose Network"]
        subgraph FrontendContainer["Frontend Container (node:18-alpine)"]
            Vite["Vite Dev Server (:5173)"]
        end

        subgraph BackendContainer["Backend Container (python:3.10-slim :8000)"]
            FastAPI["FastAPI Gateway (backend/main.py)"]
            Config["Config Manager (core/config.py)"]
            Ingestion["Ingestion Engine (core/ingestion.py)"]
            RAGEngine["RAG Orchestrator (core/rag_engine.py)"]
            VectorStore["Vector Store Manager (core/vector_store.py)"]
            AIService["LLM Adapter Layer (core/ai_service.py)"]
        end
    end

    subgraph Storage["Persistent Host Volume (./data)"]
        Uploads["Raw Files (data/uploads/)"]
        Chroma["ChromaDB HNSW Cosine Index (data/chroma/)"]
        HFCache["HuggingFace Model Cache (data/hf_cache/)"]
    end

    subgraph Providers["LLM Inference Providers"]
        Ollama["Host Ollama Server (gemma4:e4b @ :11434)"]
        Gemini["Google Gemini API (gemini-flash-latest)"]
        Groq["Groq Cloud API (llama-3.3-70b-versatile)"]
    end

    UI_Home --> FastAPI
    UI_Work --> FastAPI
    FastAPI --> Config
    FastAPI --> Ingestion
    FastAPI --> RAGEngine
    FastAPI --> VectorStore

    Ingestion --> Uploads
    RAGEngine --> VectorStore
    RAGEngine --> AIService
    VectorStore --> Chroma
    VectorStore --> HFCache

    AIService -->|"host.docker.internal:11434"| Ollama
    AIService -->|"HTTPS REST"| Gemini
    AIService -->|"HTTPS SDK"| Groq
```

---

## 2. Core Module Class & Component Design

The backend follows a clean separation of concerns where `backend/main.py` handles HTTP transport and validation, while `core/` encapsulates all domain logic and external integrations.

```mermaid
classDiagram
    class AppConfig {
        +str llm_provider
        +str ollama_base_url
        +str ollama_model
        +str gemini_api_key
        +str gemini_model
        +str groq_api_key
        +str groq_model
        +Path chroma_persist_dir
        +Path upload_dir
        +Path sample_doc_dir
        +int top_k
        +int chunk_size
        +int chunk_overlap
        +active_model() str
    }

    class BaseLLMProvider {
        <<abstract>>
        +generate(prompt: str, system_prompt: str) LLMResponse
    }

    class OllamaProvider {
        +str base_url
        +str model
        +generate(prompt: str, system_prompt: str) LLMResponse
    }

    class GeminiProvider {
        +str api_key
        +str model
        +generate(prompt: str, system_prompt: str) LLMResponse
    }

    class GroqProvider {
        +Groq client
        +str model
        +generate(prompt: str, system_prompt: str) LLMResponse
    }

    class SourceChunk {
        +str text
        +dict metadata
    }

    class RetrievedChunk {
        +str id
        +str content
        +dict metadata
        +float distance
    }

    BaseLLMProvider <|-- OllamaProvider
    BaseLLMProvider <|-- GeminiProvider
    BaseLLMProvider <|-- GroqProvider
```

---

## 3. Apple Silicon (`arm64`) & Docker Architecture

When running on an Apple Silicon Mac, the Docker setup uses native `linux/arm64` container images (`python:3.10-slim` and `node:18-alpine`) while delegating heavy LLM inference to the native macOS **Ollama** process (`gemma4:e4b`) over `host.docker.internal:11434` so that Apple Metal GPU acceleration and Unified Memory are fully utilized.

```mermaid
flowchart LR
    subgraph MacHost["macOS Host (Apple Silicon arm64)"]
        Browser["Web Browser (localhost:5173)"]
        OllamaMetal["Native Ollama Daemon (gemma4:e4b on Apple Metal GPU)"]
        HostData["./data Bind Mount"]
        HostCode["./backend, ./core, ./frontend Bind Mounts"]
    end

    subgraph DockerVM["Docker Desktop Linux VM (arm64)"]
        FE["frontend (:5173)"]
        BE["backend (:8000)"]
    end

    Browser -->|"HTTP :5173"| FE
    Browser -->|"REST :8000"| BE
    BE -->|"http://host.docker.internal:11434"| OllamaMetal
    BE <-->|"Volume Mount"| HostData
    HostCode -.->|"Live Hot-Reload"| BE
    HostCode -.->|"Vite HMR"| FE
```

### Key Architectural Optimizations
- **Metal-Accelerated Local Inference**: Running `gemma4:e4b` natively on the macOS Ollama host (`keep_alive="30m"`, `think=False`) achieves sub-second response latency (~200–500ms) while keeping the backend container lightweight.
- **Singleton Embedding & Client Caching**: Both `SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")` and `chromadb.PersistentClient` are cached in memory via `@lru_cache`, eliminating per-request PyTorch model reloads and SQLite connection churn.
- **Persistent HuggingFace Cache (`HF_HOME=/app/data/hf_cache`)**: The 90MB `all-MiniLM-L6-v2` weights are stored on the mounted `./data` volume so container rebuilds or restarts never re-download the embedding model.
- **Bounded Top-$K$ Context Retrieval**: Retrieval is strictly bounded by `min(top_k, total_indexed_chunks)` (default `5`), preventing context-window saturation regardless of how many documents are indexed.
