# Interview Preparation Guide — AI Knowledge Assistant

This guide provides ready-to-speak scripts (30-second elevator pitch, 2-minute end-to-end project walkthrough, and deep-dive talking points) along with 15 high-probability technical interview questions and answers grounded in this codebase.

---

## Part 1: Ready-to-Speak Interview Introductions

### 1. The 30-Second Elevator Pitch
> "I built **AI Knowledge Assistant**, a local-first Retrieval-Augmented Generation (RAG) web application that lets users upload unstructured documents—PDFs, Word files, and plain text—and ask complex questions with zero hallucinations.
>
> Under the hood, it uses a **React + TypeScript** frontend, a **FastAPI** backend, **PyMuPDF + Tesseract OCR** for multi-modal document and table extraction, **ChromaDB** with `all-MiniLM-L6-v2` embeddings for local vector search, and a pluggable LLM adapter layer that defaults to local **Ollama (`gemma4:e4b`)** alongside **Google Gemini** and **Groq**. Every generated answer includes inline `[S1]` citations and sentence-level keyword highlighting so the user can verify the exact page and paragraph every claim came from."

---

### 2. The 2-Minute Complete Project Walkthrough ("Tell me about this project from start to finish")

Use this 5-step structure (**Problem → Architecture → Ingestion Pipeline → Retrieval & Generation → Key Engineering Optimizations**):

#### Step 1: The Problem & Goal (15 seconds)
> "Standard LLMs hallucinate when asked about private documents, and uploading sensitive documents to cloud APIs isn't always acceptable. I designed **AI Knowledge Assistant** to solve both problems: it runs 100% locally on Apple Silicon with Docker, forces the LLM to answer strictly from retrieved document chunks, and proves every claim with verifiable source citations and highlighted snippets."

#### Step 2: System Architecture (20 seconds)
> "I architected the system into four clean layers:
> 1. **Frontend (`frontend/`)**: A React 18 and TypeScript SPA built with Vite featuring a live telemetry homepage, a split-pane RAG workspace, document inventory management, and a `top_k` retrieval tuning slider.
> 2. **API Gateway (`backend/main.py`)**: A FastAPI REST server exposing endpoints for multi-file upload, one-click sample document indexing, per-document deletion, collection clearing, and conversational `/chat`.
> 3. **Core RAG Engine (`core/`)**: Modular Python services handling ingestion, vector storage, RAG prompt orchestration, and an LLM provider adapter pattern.
> 4. **Storage & Inference**: Local persistent **ChromaDB** using HNSW cosine similarity, paired with a `BaseLLMProvider` interface supporting **Ollama (`gemma4:e4b`)**, **Gemini**, and **Groq**."

#### Step 3: The Document Ingestion Pipeline (`core/ingestion.py` & `core/vector_store.py`) (30 seconds)
> "When a user uploads files via `POST /documents/upload`:
> - First, for **PDFs**, `PyMuPDF` extracts native text page-by-page and runs `page.find_tables()` to convert PDF tables into structured Markdown. If a page returns zero text—meaning it's a scanned PDF—it automatically renders a 2x resolution pixmap and falls back to **Tesseract OCR** (`pytesseract`).
> - For **DOCX** files, `python-docx` groups paragraphs under their Word `Heading` styles so section hierarchy is preserved, and extracts Word tables into Markdown.
> - Next, LangChain's `RecursiveCharacterTextSplitter` splits the extracted sections into **1,000-character chunks with a 150-character overlap** to preserve context across chunk boundaries, tagging each chunk with `source`, `page`, `section`, and `word_count`.
> - Finally, before upserting into **ChromaDB**, the vector store deletes any existing chunks for that filename to prevent duplicate chunks on re-uploads, embeds the text into **384-dimensional vectors** using `all-MiniLM-L6-v2`, and persists them to disk."

#### Step 4: The Question Answering & Citation Pipeline (`core/rag_engine.py`) (30 seconds)
> "When the user asks a question via `POST /chat`:
> - The query is embedded with the same `all-MiniLM-L6-v2` model and ChromaDB performs a **top-$K$ cosine similarity search** (defaulting to `top_k = 5`, capped dynamically by total indexed chunks).
> - Each retrieved chunk is formatted with a strict source marker like `[S1] source=JD.pdf page=2 section=Requirements`.
> - The prompt combines the last 6 turns of conversation history, the `[S1..Sn]` context blocks, and a strict system prompt instructing the model to answer *only* from context, cite `[S1]` at the end of every sentence, or reply with an explicit refusal if the answer isn't present.
> - Once the LLM (`gemma4:e4b`) responds, a post-processor checks for refusal phrases, regex-parses the cited `[S1]` indices, calculates cosine similarity percentages (`1 - distance`), and runs sentence-level keyword matching to highlight the exact overlapping words in the source cards."

#### Step 5: Key Engineering Optimizations I Implemented (25 seconds)
> "Three engineering optimizations made a huge difference in performance:
> 1. **Singleton Model & DB Caching**: I cached `SentenceTransformerEmbeddingFunction` and `chromadb.PersistentClient` with `@lru_cache` so PyTorch doesn't reload the 90MB embedding model on every request—dropping health and search overhead from ~2.5 seconds to under 15 milliseconds.
> 2. **Bounded Top-$K$ Retrieval**: I bounded retrieval to `min(top_k, total_indexed_chunks)` and windowed chat history to the last 6 messages so the LLM context window never overflows as the knowledge base grows.
> 3. **Zero-Rebuild Docker + Apple Silicon Metal Bridge**: I used `uv` in `Dockerfile.backend` for 10x faster builds, mounted live code and `.env` with `uvicorn --reload` and Vite HMR so code changes never require rebuilding containers, and bridged the backend container to the native macOS Ollama daemon via `host.docker.internal:11434` with `keep_alive='30m'` and `think=False` so `gemma4:e4b` runs on Apple Metal GPU in ~200–500ms."

---

## Part 2: Top 15 Interview Questions & Answers

### Architecture & Design Decisions

#### Q1: Why did you choose ChromaDB over cloud vector databases like Pinecone, or heavier self-hosted engines like Milvus/Qdrant?
**Answer:**
- For a local-first single-node RAG assistant, **ChromaDB (`PersistentClient`)** runs embedded in-process backed by SQLite + local HNSW binary segments (`data/chroma/`), requiring zero external database containers or cloud API keys.
- It natively supports metadata filtering (`where={"source": filename}`) which I use for per-document deletion and re-upload deduplication, and lets us configure cosine distance (`{"hnsw:space": "cosine"}`) directly on collection creation.
- If scaling to multi-tenant production with millions of vectors and concurrent writers, I would swap `core/vector_store.py` to **Qdrant** or **pgvector** while keeping the rest of `core/rag_engine.py` unchanged.

#### Q2: How did you design the LLM integration so you can switch between Ollama, Gemini, and Groq without changing business logic?
**Answer:**
- In `core/ai_service.py`, I implemented the **Adapter + Factory Pattern**:
  - `BaseLLMProvider` is an abstract base class with a single contract: `generate(prompt: str, system_prompt: str | None) -> LLMResponse`.
  - `OllamaProvider`, `GeminiProvider`, and `GroqProvider` each implement `BaseLLMProvider` and normalize their responses into a frozen `LLMResponse(text, raw)` dataclass.
  - `create_llm_provider(config)` inspects `LLM_PROVIDER` from `AppConfig` and returns the concrete adapter.
- Additionally, `core/config.py` reloads `.env` dynamically via `load_dotenv(override=True)`, so switching providers or models in `.env` takes effect immediately on the next request without restarting the server.

#### Q3: How does your Docker container use Apple Silicon GPU acceleration if Docker on macOS runs inside a Linux VM without direct Metal GPU passthrough?
**Answer:**
- That is a classic macOS Docker bottleneck: Linux containers inside Docker Desktop's VM cannot access Apple's Metal GPU directly, so running Ollama *inside* a Docker container on a Mac forces slow CPU-only inference.
- To solve this, I run the FastAPI + ChromaDB backend inside Docker, and run **Ollama natively on macOS** where it has full access to Apple Metal and Unified Memory.
- In `docker-compose.yml`, I configured `extra_hosts: ["host.docker.internal:host-gateway"]`, and in `core/config.py`, when `RUNNING_IN_DOCKER=true`, `localhost:11434` is automatically rewritten to `http://host.docker.internal:11434`.
- In `OllamaProvider`, I also set `keep_alive: "30m"` so the 6.6GB `gemma4:e4b` weights stay pinned in Unified Memory across requests, and `think: False` so it skips long reasoning chains and returns grounded RAG answers in ~200–500ms.

---

### Document Ingestion & Chunking

#### Q4: Walk me through how you handle different document formats (`.pdf`, `.docx`, `.txt`) and edge cases like scanned PDFs or tables.
**Answer:**
- **PDFs (`_extract_pdf_chunks`)**:
  1. Opens the file with `pymupdf` (`fitz`) and extracts text per page.
  2. Runs `page.find_tables()` to detect tabular structures on the page and converts non-empty cells into pipe-delimited Markdown tables (`[Table 1] col1 | col2`).
  3. **OCR Fallback**: If `page.get_text("text").strip()` is empty (indicating a scanned image page), it renders the page at $2\times$ resolution (`pymupdf.Matrix(2, 2)`) into a PNG byte stream, opens it with `Pillow`, and runs `pytesseract.image_to_string()`. It's wrapped in a `try...except` block so missing Tesseract on a bare host won't crash the upload.
- **DOCX (`_extract_docx_chunks`)**:
  - Iterates through paragraphs, using Word `Heading` styles (`style.name.lower().startswith("heading")`) as section boundaries to flush buffered paragraphs with accurate `section` metadata, then iterates over `doc.tables` to append Markdown tables.
- **TXT (`_extract_txt_chunks`)**:
  - Reads with `utf-8` and falls back to `latin-1` (`errors="replace"`) to prevent `UnicodeDecodeError` crashes on legacy files.

#### Q5: Why did you choose `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=150`?
**Answer:**
- **`RecursiveCharacterTextSplitter`** splits hierarchically using `["\n\n", "\n", ". ", " ", ""]`. That means it tries to keep paragraphs together first, then lines, then complete sentences, and only splits mid-sentence as a last resort.
- **`chunk_size=1000` characters (~180–220 words / ~250 tokens)**:
  - Our embedding model (`all-MiniLM-L6-v2`) has a **256 word-piece token window** (~1,000–1,200 characters). If chunks are larger than ~1,000 characters, `all-MiniLM-L6-v2` silently truncates the tail of the chunk when computing embeddings! Keeping `chunk_size=1000` ensures 100% of every chunk is represented in its vector embedding.
- **`chunk_overlap=150` characters**:
  - Prevents context loss when a key fact or sentence spans the boundary between two consecutive chunks.

#### Q6: What happens if a user uploads the same document twice after editing it? How do you prevent duplicate or stale chunks?
**Answer:**
- In `core/vector_store.py`, inside `add_documents()`, I first collect the set of unique `source` filenames in the incoming upload batch and execute:
  ```python
  collection.delete(where={"source": source_name})
  ```
- Then I generate deterministic chunk IDs combining filename, page, chunk index, batch index, and an 8-character SHA-1 content hash (`{source}:{page}:{chunk_index}:{index}:{sha1_8}`) and call `collection.upsert()`.
- This guarantees that re-uploading an updated version of a file atomically replaces its old chunks rather than leaving stale or duplicate vectors in ChromaDB.

---

### Vector Search, RAG & Hallucination Prevention

#### Q7: Why did you use cosine similarity (`hnsw:space = cosine`) instead of L2 (Euclidean) distance?
**Answer:**
- Cosine distance measures the **angle** between two embedding vectors (`1 - cos(θ)`), making it invariant to vector magnitude differences caused by varying chunk lengths (e.g. a 200-character table chunk vs. a 950-character paragraph chunk).
- It also maps cleanly to an intuitive similarity percentage for the UI: `similarity = round(max(0.0, 1.0 - distance), 3)`.

#### Q8: What was the `max(4, total_indexed_chunks)` bug in the original RAG engine, and why was it dangerous?
**Answer:**
- Originally, `core/rag_engine.py` calculated `search_limit = max(4, total_indexed_chunks)`.
- When the database had 100 chunks, `max(4, 100)` evaluated to `100`—meaning every single question retrieved **the entire database** and stuffed all 100 chunks into the LLM prompt! That caused massive latency, diluted attention ("lost in the middle" phenomenon), and exceeded token limits on larger uploads.
- Conversely, if only 2 chunks existed in ChromaDB, requesting `n_results=4` exceeded the collection size.
- I fixed this to:
  ```python
  search_limit = max(1, min(desired_k, total_indexed_chunks))
  ```
  with `desired_k` defaulting to `5` (and configurable via slider from `2` to `10`), plus an early return when `total_indexed_chunks == 0`.

#### Q9: How do you prevent hallucinations in your RAG pipeline?
**Answer:**
I enforce grounding at **three levels**:
1. **Pre-LLM Guardrail**: If `total_indexed_chunks == 0` or `retrieved_chunks` is empty, `answer_question()` short-circuits without calling the LLM and returns a deterministic refusal.
2. **Prompt & Sampling Controls**:
   - Temperature is locked to `0.1` (`core/ai_service.py`) to suppress creative speculation.
   - The system prompt explicitly forbids outside knowledge, requires `[S1]` citations at the end of every claim, and prescribes an exact refusal string (`"I could not find the answer in the uploaded documents."`).
3. **Post-LLM Verification & Refusal Detection**:
   - `_is_refusal_response()` checks the LLM output against known refusal phrases and sets `refusal=True`, `confidence="low"`, and `sources=[]`.
   - When the model cites `[S1]`, only valid indices (`1 <= idx <= len(retrieved_chunks)`) are attached as verified sources (`confidence="high"`).

#### Q10: How does the source snippet highlighting (`_extract_and_highlight_match`) work?
**Answer:**
- Instead of dumping the raw 1,000-character chunk without context, `_extract_and_highlight_match(chunk_text, answer_text)` in `core/rag_engine.py`:
  1. Strips citation brackets (`[S1]`) from the LLM's answer and lowercases it.
  2. Splits the source chunk into individual sentences using regex lookbehind `(?<=[.!?])\s+`.
  3. Extracts significant words (`len(w) > 4`) from each sentence and counts how many appear in the LLM's answer.
  4. If a sentence has `>= 2` keyword matches, it wraps those matched keywords in `**word**` using case-insensitive word-boundary regex (`\b(word)\b`), which the React frontend (`renderHighlightedSnippet`) converts into yellow `<mark className="snippet-highlight">` tags.
  5. If no individual sentence meets the threshold (e.g., paraphrased summary), it gracefully falls back to showing the full chunk text.

---

### Performance, Caching & Docker Engineering

#### Q11: What caused the 2.5-second latency spike on `/health` and `/chat` initially, and how did you fix it?
**Answer:**
- In `core/vector_store.py`, `get_collection()` called `_get_embedding_function()`, which instantiated a new `SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")` on every call.
- Because `/health` calls `count_documents()` -> `get_collection()`, every health check and chat request was re-initializing the PyTorch `SentenceTransformer` model and opening a new `chromadb.PersistentClient` from disk.
- By decorating `_get_embedding_function()` with `@lru_cache(maxsize=1)` and `_get_cached_client(persist_path)` with `@lru_cache(maxsize=4)`, the embedding model and Chroma client are initialized once per worker process and reused in memory—reducing `/health` latency from ~2.5s to <15ms.

#### Q12: How did you structure Docker so code and `.env` changes don't require rebuilding the 3.5GB backend image?
**Answer:**
- **Dependency Layer Caching with `uv`**: `Dockerfile.backend` copies `ghcr.io/astral-sh/uv`, copies `requirements.txt` *before* application code, and uses `RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system -r requirements.txt`.
- **Live Code Bind Mounts**: `docker-compose.yml` mounts `./backend:/app/backend` and `./core:/app/core` into the backend container and runs `uvicorn --reload --reload-dir /app/backend --reload-dir /app/core`.
- **Frontend HMR with Isolated `node_modules`**: Mounts `./frontend:/app` alongside a named volume `frontend_node_modules:/app/node_modules` and configures Vite with `usePolling: true`.
- **Live `.env` Reload**: Mounts `./.env:/app/.env:ro` and calls `load_dotenv(override=True)` inside `load_config()`.

---

### Scaling & Production Follow-Up Questions ("What would you improve next?")

#### Q13: What are the limitations of pure vector (dense) search here, and how would you improve retrieval accuracy?
**Answer:**
- Pure dense embeddings (`all-MiniLM-L6-v2`) capture semantic similarity well but can miss exact keyword, acronym, or ID matches (e.g., an exact job requisition code or version number).
- To improve retrieval in v2, I would implement **Hybrid Search + Cross-Encoder Reranking**:
  1. Combine **BM25 sparse keyword search** with **ChromaDB dense cosine search** using **Reciprocal Rank Fusion (RRF)** to retrieve the top 15 candidates.
  2. Pass those 15 candidates through a lightweight **Cross-Encoder reranker** (e.g., `ms-marco-MiniLM-L-6-v2`) to score `(query, chunk)` pairs jointly and pass the top 5 to the LLM.

#### Q14: Why didn't you stream tokens via Server-Sent Events (SSE) or WebSockets in `/chat`, and how would you add it?
**Answer:**
- Currently `/chat` returns a single JSON payload because post-processing (`_extract_and_highlight_match` and citation validation) inspects the complete generated answer before mapping and highlighting the cited source snippets.
- To add streaming while keeping citation highlighting:
  - Switch `POST /chat` to FastAPI's `StreamingResponse` (`text/event-stream`), set `"stream": True` in `OllamaProvider`, and stream `event: token` chunks as they arrive.
  - Once generation finishes (`done: True`), run the citation parser and sentence highlighter on the accumulated text and emit a final `event: sources` JSON frame containing the verified source cards and confidence score.

#### Q15: How would you make this multi-user ready?
**Answer:**
- Currently all uploads share a single ChromaDB collection (`ai_knowledge_assistant`).
- To support multiple users or workspaces:
  1. Add JWT / OAuth authentication middleware in FastAPI.
  2. Scope documents either by **per-tenant ChromaDB collections** (`f"kb_{user_id}"`) or by storing `user_id` / `workspace_id` in chunk metadata and passing `where={"workspace_id": current_workspace}` on every `collection.query()` and `collection.get()` call.
  3. Move synchronous PDF OCR/ingestion on large files to a background task queue (`FastAPI BackgroundTasks` or `Celery + Redis`) with status polling.
