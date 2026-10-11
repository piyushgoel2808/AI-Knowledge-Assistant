# End-to-End Data Flow

This document traces the complete lifecycle of data through the **AI Knowledge Assistant** across both primary pipelines:
1. **Document Ingestion & Vector Indexing Pipeline**
2. **Conversational RAG Query, Citation & Highlighting Pipeline**

---

## 1. Document Ingestion & Vector Indexing Pipeline

When a user uploads `.pdf`, `.docx`, or `.txt` files (or clicks **Load Sample PDFs**), the system executes a 6-stage pipeline that transforms raw files into 384-dimensional cosine-indexed vectors in local ChromaDB.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as React Frontend (:5173)
    participant API as FastAPI Backend (:8000)
    participant FS as Local Disk (data/uploads/)
    participant Ingest as Ingestion Engine (core/ingestion.py)
    participant Embed as SentenceTransformer (all-MiniLM-L6-v2)
    participant Chroma as ChromaDB (data/chroma/)

    User->>UI: Select .pdf / .docx / .txt files & click Upload
    UI->>API: POST /documents/upload (multipart/form-data)
    API->>FS: Stream & save raw files to data/uploads/{filename}
    API->>Ingest: ingest_file(target_path, config)

    alt File is .pdf
        Ingest->>Ingest: Extract text per page via PyMuPDF
        Ingest->>Ingest: Extract tables via page.find_tables()
        opt Page has no text (Scanned PDF)
            Ingest->>Ingest: Render 2x PNG pixmap & run pytesseract OCR
        end
    else File is .docx
        Ingest->>Ingest: Group paragraphs by Heading styles & extract tables
    else File is .txt
        Ingest->>Ingest: Read UTF-8 text (with Latin-1 fallback)
    end

    Ingest->>Ingest: Split into 1000-char chunks (150 overlap) via RecursiveCharacterTextSplitter
    Ingest-->>API: Return list[Document] with rich metadata

    API->>Chroma: Delete prior chunks where source == filename (deduplication)
    API->>Embed: Compute 384-dim embeddings for new chunks
    Embed-->>Chroma: Upsert (ids, embeddings, documents, metadatas)
    Chroma-->>API: Confirm stored chunk count & document summary
    API-->>UI: 200 OK {message, files, chunks_indexed, stored_chunks, documents}
    UI-->>User: Update Indexed Documents list & telemetry counters
```

### Ingestion Decision Flow

```mermaid
flowchart TD
    Start["Incoming File Upload"] --> CheckExt{"Extension Supported? (.pdf, .docx, .txt)"}
    CheckExt -->|"No"| Skip["Skip Unsupported File"]
    CheckExt -->|"Yes"| Save["Save to data/uploads/"]

    Save --> RouteType{"File Type"}
    RouteType -->|".pdf"| PDFOpen["Open PDF with PyMuPDF"]
    PDFOpen --> PDFText["Extract Native Text & Tables (page.find_tables)"]
    PDFText --> HasText{"Page Text Empty?"}
    HasText -->|"Yes (Scanned)"| OCR["Render 2x Pixmap -> Tesseract OCR"]
    HasText -->|"No"| BuildPDFChunk["Create SourceChunk (page, section)"]
    OCR --> BuildPDFChunk

    RouteType -->|".docx"| DOCXOpen["Open with python-docx"]
    DOCXOpen --> DOCXParse["Group Paragraphs by Heading + Convert Tables to Markdown"]
    DOCXParse --> BuildDOCXChunk["Create SourceChunk (section)"]

    RouteType -->|".txt"| TXTOpen["Read Text (UTF-8 / Latin-1 Fallback)"]
    TXTOpen --> BuildTXTChunk["Create SourceChunk"]

    BuildPDFChunk --> Split["RecursiveCharacterTextSplitter (size=1000, overlap=150)"]
    BuildDOCXChunk --> Split
    BuildTXTChunk --> Split

    Split --> Dedup["Delete Existing Chunks for Same Filename in ChromaDB"]
    Dedup --> Upsert["Embed with all-MiniLM-L6-v2 & Upsert into ChromaDB"]
```

---

## 2. Conversational Question Answering (RAG) Pipeline

When a user submits a question in the chat workspace, the backend retrieves the top-$K$ semantically matching chunks, constructs a strict citation-enforcing prompt, invokes `gemma4:e4b` on Ollama (or Gemini/Groq), and highlights the exact supporting sentences in the cited source chunks.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as React Frontend (:5173)
    participant API as FastAPI Backend (:8000)
    participant RAG as RAG Engine (core/rag_engine.py)
    participant VS as Vector Store (core/vector_store.py)
    participant LLM as Ollama (gemma4:e4b @ :11434)

    User->>UI: Enter question & press Send
    UI->>API: POST /chat {question, history, top_k}
    API->>RAG: answer_question(question, chat_history, top_k)
    RAG->>VS: count_documents()

    alt Total Indexed Chunks == 0
        VS-->>RAG: 0 chunks
        RAG-->>API: Return grounded refusal ("No documents are indexed yet...")
        API-->>UI: 200 OK {answer, sources: [], confidence: "low", refusal: true}
    else Chunks Available (> 0)
        RAG->>VS: search_documents(query, top_k=min(top_k, total_chunks))
        VS-->>RAG: Return top-K RetrievedChunk objects with cosine distances
        RAG->>RAG: Build System Prompt + Context Block ([S1], [S2], ...) + Recent History
        RAG->>LLM: POST /api/generate (model="gemma4:e4b", think=false, temperature=0.1)
        LLM-->>RAG: Generated answer with inline citations (e.g. "... [S1], [S2].")
        RAG->>RAG: Check _is_refusal_response() & parse [S1..Sn] citations
        RAG->>RAG: Run _extract_and_highlight_match() on cited chunks
        RAG-->>API: Return {answer, sources, confidence, refusal, model, provider}
        API-->>UI: 200 OK JSON response
        UI-->>User: Render formatted answer, [S1] pills & highlighted source cards
    end
```

---

## 3. Post-Processing: Citation Parsing & Sentence Highlighting

A key differentiator of the RAG engine (`core/rag_engine.py`) is how it verifies and highlights source evidence after the LLM generates an answer:

```mermaid
flowchart LR
    RawAns["LLM Answer Text"] --> StripThink["Strip <think> Tags"]
    StripThink --> CheckRefusal{"Matches Refusal Phrase?"}

    CheckRefusal -->|"Yes"| RefusalOut["refusal = true, confidence = low, sources = []"]
    CheckRefusal -->|"No"| ParseCites["Regex Extract [S1], [S2], [S1, S2]"]

    ParseCites --> HasCites{"Explicit Citations Found?"}
    HasCites -->|"Yes"| MapCited["Select Cited Chunks (confidence = high)"]
    HasCites -->|"No"| MapFallback["Select Top-2 Chunks (confidence = medium)"]

    MapCited --> Highlight["Split Chunk into Sentences & Bold Overlapping Keywords (>4 chars)"]
    MapFallback --> Highlight
    Highlight --> FinalOut["Return Grounded Answer + Highlighted Source Cards"]
```
