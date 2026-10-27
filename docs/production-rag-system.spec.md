# Feature Specification: Production RAG Knowledge Assistant with Strict Attribution & Large-File Scalability

**Specification ID:** `SPEC-RAG-2026-V2`  
**Status:** Approved  
**Author:** AI Knowledge Assistant Core Team  
**Target:** Local Mac OS / Docker / Apple Silicon / Hybrid Cloud

---

## 1. Problem Statement

Enterprise knowledge bases consist of heterogeneous, complex documents (employee handbooks, financial reports, technical documentation, and compliance policies) stored as PDFs, DOCX, and TXT files. Conventional naive RAG pipelines suffer from six critical architectural failure modes:

1. **Large-File Ingestion Starvation:** Synchronous HTTP document ingestion crashes with `504 Gateway Timeout` or OOM exceptions when processing files $\ge 20\text{MB}$ or $\ge 100\text{ pages}$.
2. **Layout & Table Corruption:** Naive text extractors destroy multi-column reading order and flatten tabular data into unusable space-delimited text, losing cell-to-header semantics.
3. **The Chunking Granularity Dilemma:** Fixed-size chunking ($1000\text{ chars}$) forces an engineering compromise: small chunks optimize vector retrieval but starve the LLM of context, while large chunks dilute vector embeddings with multiple topics.
4. **Lexical Blindness in Pure Vector Search:** Dense embeddings fail on exact alphanumeric policy numbers, acronyms, and section references (e.g., *"Policy #HR-409B"*, *"Form 1099-NEC"*).
5. **Conversational Pronoun Disconnect:** Multi-turn follow-up questions (*"Does it apply to contractors?"*) fail during retrieval because naive systems search the raw user string without resolving pronouns against previous turns.
6. **Attribution Fabrication & Disagreement Blindness:** LLMs hallucinate citation tags (`[S1]`) for facts absent from the cited snippet, and silently average or arbitrarily pick between conflicting multi-document statements.

---

## 2. Solution Overview

The **Production AI Knowledge Assistant** implements an end-to-end, deterministic RAG system engineered for correctness, transparency, and scalability:

* **Decoupled Asynchronous Ingestion:** Sub-50ms HTTP acceptance returning a `job_id`, executed by a background streaming worker with backpressure (10 pages per batch) and live progress tracking.
* **Layout-Aware Parsing:** PyMuPDF block geometry analysis preserving multi-column reading order and converting tables into structured Markdown tables.
* **Hierarchical Parent-Child Chunking:** Leaf child chunks ($250\text{ tokens}$) indexed in ChromaDB for high vector recall; rich parent sections ($1200\text{ tokens}$) stored in SQLite and injected into the LLM context.
* **Two-Stage Hybrid Search & Reranking:** Dense vector search (`all-MiniLM-L6-v2`) fused with sparse lexical search (`rank_bm25`) via Reciprocal Rank Fusion ($k=60$), followed by precision cross-attention reranking (`bge-reranker-base`).
* **Query Condensation Engine:** Standalone query rewriter resolving conversational pronouns into fully qualified search queries before hitting the index.
* **Two-Tier Attribution & Disagreement Engine:** Post-generation NLI sentence-to-snippet entailment verification, deterministic refusal when rerank scores fall below threshold ($\tau < 0.30$), and explicit multi-document discrepancy reporting.
* **Hands-Free Voice Layer:** Browser Web Speech API integration for STT/TTS with strict retention of on-screen attribution cards.

---

## 3. Extensive User Stories & Acceptance Criteria

### US-001: Large File Ingestion with Real-Time Progress
**As a** compliance officer uploading a 350-page employee handbook ($45\text{MB}$),  
**I want** the system to ingest the document asynchronously with live progress feedback,  
**So that** my browser connection does not time out and I can track the ingestion stages.

* **EARS Functional Requirements:**
  * While an upload request is received, when the file is valid, the system shall return an ingestion `job_id` within $50\text{ms}$ with status `queued`.
  * While ingestion is running, when the worker completes a 10-page batch, the system shall update progress percentage, completed pages, and active stage (`parsing` | `chunking` | `embedding`).
  * While polling `GET /documents/tasks/{job_id}`, when processing finishes, the system shall return status `completed` and total indexed chunks.
* **Acceptance Criteria (Given-When-Then):**
  * **AC-001.1:** Given a 350-page PDF handbook, When the user uploads it via the web interface, Then the API responds with HTTP 202 and a `job_id` within $50\text{ms}$.
  * **AC-001.2:** Given an active ingestion job, When the UI polls the task endpoint, Then it renders a real-time progress bar transitioning smoothly from 0% to 100% without gateway timeouts.

---

### US-002: Structured Table & Multi-Column Question Answering
**As an** operations analyst reviewing health benefits documentation,  
**I want** to query tabular deductibles and multi-column policy clauses,  
**So that** I receive exact numerical answers without column bleed or cell misalignment.

* **EARS Functional Requirements:**
  * When a PDF page contains multi-column layouts, the system shall sort text blocks by column bounding box coordinates before chunking.
  * When a PDF page contains structured tables, the system shall convert cell matrices into Markdown tables with headers preserved.
  * The system shall retain Markdown table formatting within parent chunks fed to the LLM.
* **Acceptance Criteria:**
  * **AC-002.1:** Given a 2-column page with distinct left and right topics, When text is extracted, Then sentences from the right column are never interleaved into the left column.
  * **AC-002.2:** Given an uploaded table of medical deductibles, When the user asks "What is the deductible for out-of-network specialists?", Then the assistant returns the exact numerical amount and cites the row and column in the source snippet.

---

### US-003: Conversational Multi-Turn Follow-Up Resolution
**As a** team manager querying company policies in an active chat session,  
**I want** the assistant to understand follow-up questions referencing previous answers,  
**So that** I don't have to retype full context in every message.

* **EARS Functional Requirements:**
  * While conversation history exists, when a new user turn is submitted, the system shall rewrite the question into a standalone query containing all referenced entities.
  * The system shall execute vector and BM25 retrieval using the rewritten query, while supplying original conversation turns to the generation prompt.
* **Acceptance Criteria:**
  * **AC-003.1:** Given a previous turn discussing "Parental Leave for Software Engineers", When the user asks "Does it also apply to contractors?", Then the query rewriter outputs "Does parental leave apply to independent contractors for Software Engineers?".
  * **AC-003.2:** Given the rewritten query, When retrieval executes, Then chunks discussing contractor leave are retrieved rather than general loan/visa extension clauses.

---

### US-004: Verifiable Source Attribution & Snippet Grounding
**As an** internal auditor verifying legal policy compliance,  
**I want** every factual claim in an answer to be linked to a verifiable source snippet,  
**So that** I can immediately audit the evidence behind any recommendation.

* **EARS Functional Requirements:**
  * The system shall append deterministic citation markers (`[S1]`, `[S2]`) directly after every asserted factual statement.
  * When citations are emitted, the system shall provide document filename, 1-based page number, section heading, similarity score, and verbatim snippet.
  * When an answer is drafted, the system shall verify that each cited snippet entails the claim; where entailment fails, the citation shall be flagged as unverified.
* **Acceptance Criteria:**
  * **AC-004.1:** Given an assistant answer asserting "Employees receive 18 days annual leave [S1]", When the user clicks citation `[S1]`, Then a modal displays the document name, page 14, Section "Leave Policy", and the exact supporting sentence.
  * **AC-004.2:** Given a claim where the cited snippet mentions the document but does not contain the fact, When entailment verification runs, Then the system marks confidence as "unverified" and alerts the user.

---

### US-005: Multi-Document Disagreement & Synthesis
**As an** HR director comparing regional employee handbooks ("Handbook_US.pdf" and "Handbook_UK.pdf"),  
**I want** the assistant to highlight where policies contradict each other,  
**So that** I am not misled by an arbitrary single-document selection.

* **EARS Functional Requirements:**
  * Where retrieved chunks from different documents present conflicting figures or rules for the same topic, the system shall synthesize both positions and explicitly report the discrepancy.
  * The system shall not pick one source arbitrarily or average numerical discrepancies.
* **Acceptance Criteria:**
  * **AC-005.1:** Given Doc A states probation is 3 months and Doc B states probation is 6 months, When the user asks "How long is the probation period?", Then the assistant answers: "Sources disagree: Handbook_US.pdf (Page 4) specifies 3 months, whereas Handbook_UK.pdf (Page 6) specifies 6 months."

---

### US-006: Deterministic Abstention on Unknown Questions
**As an** enterprise user asking about topics not covered in uploaded files,  
**I want** the system to clearly state that the information is unavailable,  
**So that** I never receive fabricated or hallucinated answers.

* **EARS Functional Requirements:**
  * When the top cross-encoder reranking score for a query is below threshold $\tau = 0.30$, the system shall bypass the generation model and return an immediate deterministic refusal.
  * The system shall return `"confidence": "low"` and `"refusal": true`.
* **Acceptance Criteria:**
  * **AC-006.1:** Given an index containing only employment handbooks, When the user asks "What is the Q3 corporate revenue?", Then the assistant responds: "I could not find the answer in the uploaded documents." within $150\text{ms}$.

---

### US-007: Hands-Free Voice Interaction Layer
**As an** executive reviewing documents while multitasking,  
**I want** to ask questions using speech and hear responses read aloud,  
**So that** I can interact hands-free while maintaining access to visual citations.

* **EARS Functional Requirements:**
  * Where voice input is enabled, the system shall transcribe microphone audio using the browser Web Speech API and populate the chat input.
  * Where voice output is enabled, the system shall synthesize the text response into audio while displaying source attribution cards on screen.
* **Acceptance Criteria:**
  * **AC-007.1:** Given the microphone button is toggled on, When the user speaks "Summarize the leave policy", Then the speech is transcribed into the input field and sent.
  * **AC-007.2:** Given voice output is active, When the answer completes, Then the assistant speaks the response while the UI highlights citations on screen.

---

## 4. Implementation Decisions

```
+-----------------------------------------------------------------------------------------------+
|                                    TECHNICAL COMPONENT ARCHITECTURE                           |
+----------------------+-----------------------------+------------------------------------------+
| Component            | Technology Choice           | Engineering Rationale                    |
+----------------------+-----------------------------+------------------------------------------+
| Background Worker    | FastAPI BackgroundTasks     | Lightweight in-process worker, zero Redis|
|                      | + SQLite Task Registry      | dependency for local single-node deploy. |
+----------------------+-----------------------------+------------------------------------------+
| Parser Engine        | PyMuPDF (fitz)              | 10x faster than PyPDF; supports block    |
|                      | + Markdown Table Extractor  | bounding boxes, reading order, and OCR.  |
+----------------------+-----------------------------+------------------------------------------+
| Chunking Engine      | Hierarchical Parent-Child   | Solves context-vs-granularity dilemma:   |
|                      | (Child: 250t / Parent: 1200t| Child for vector ANN; Parent for LLM.    |
+----------------------+-----------------------------+------------------------------------------+
| Dual-Index Store     | ChromaDB (Dense Cosine)     | High-recall semantic retrieval combined  |
|                      | + rank_bm25 (Inverted Index)| with exact lexical keyword matching.     |
+----------------------+-----------------------------+------------------------------------------+
| Candidate Fusion     | Reciprocal Rank Fusion      | Unifies rank lists across disparate      |
|                      | (RRF, constant k=60)        | score distributions without calibration. |
+----------------------+-----------------------------+------------------------------------------+
| Reranker             | BAAI/bge-reranker-base      | Cross-attention between query & passage  |
|                      | (Cross-Encoder)             | eliminates false-positive cosine matches.|
+----------------------+-----------------------------+------------------------------------------+
| Query Rewriter       | Fast LLM Context Condenser  | Resolves pronouns & ellipsis before      |
|                      | (Few-shot template)         | vector retrieval runs.                   |
+----------------------+-----------------------------+------------------------------------------+
| Claim Verifier       | Sentence-Level NLI Judge    | Detects hallucinated citations where the |
|                      | (Snippet -> Claim Entail)   | snippet doesn't support the assertion.   |
+----------------------+-----------------------------+------------------------------------------+
| Inference Engine     | Local Ollama (gemma4:e4b)   | Zero cloud cost, 100% private data,      |
|                      | via Apple Silicon Metal GPU | 38 tokens/second sustained speed.        |
+----------------------+-----------------------------+------------------------------------------+
```

---

## 5. Testing Decisions & Evaluation Gates

1. **Unit Test Suite (`tests/unit/`):**
   * Multi-column bounding-box reading order verification.
   * Markdown table extraction fidelity across merged and empty cells.
   * Hierarchical chunk parent ID foreign-key integrity in SQLite docstore.
   * Reciprocal Rank Fusion mathematical accuracy with tied rankings.
   * Query condensation few-shot template formatting.

2. **Integration Test Suite (`tests/integration/`):**
   * Asynchronous task lifecycle: `POST /documents/upload` $\rightarrow$ poll `GET /documents/tasks/{job_id}` until `completed`.
   * Multi-turn query retrieval: verify rewritten query matches golden reference on 3-turn dialogues.
   * Contradiction detection: ingest two conflicting policy documents and verify response contrasts both.
   * Large document stress test: ingest a 250-page PDF with memory capped at $< 300\text{MB}$ RAM.

3. **Quantitative RAG Evaluation Metrics:**
   * **Retrieval Mean Reciprocal Rank (MRR@5):** Must achieve $\text{MRR@5} \ge 0.85$ on a 25-question golden benchmark.
   * **Attribution Entailment Score:** $\ge 95\%$ of emitted citations must achieve an `Entailment` verdict against the snippet.
   * **Refusal Precision:** $100\%$ refusal rate on 15 adversarial unanswerable questions.

---

## 6. Out of Scope

To preserve engineering focus, maintain trivial operating costs, and satisfy the assignment guidelines, the following are explicitly out of scope:

1. **Multi-Tenant User Authentication & Billing:** No OAuth2, JWT user sessions, Stripe integration, or tenant database sharding.
2. **Model Fine-Tuning:** No LoRA/RLHF training of foundation models; all intelligence is achieved via retrieval and grounded prompt engineering.
3. **Proprietary Cloud OCR APIs:** No Google Cloud Document AI, AWS Textract, or Azure Form Recognizer; runs entirely locally using open-source PyMuPDF and Tesseract.
4. **Distributed Cluster Orchestration:** No multi-node Kubernetes, Ray, or Celery/RabbitMQ clusters; system runs in clean Docker Compose on a single developer workstation.
