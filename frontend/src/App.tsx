import { useEffect, useMemo, useRef, useState } from 'react'
import './styles.css'
import {
  askQuestion,
  clearDocuments,
  deleteDocument,
  getHealth,
  getTaskStatus,
  loadSampleDocuments,
  uploadDocuments,
  type ChatMessage,
  type ChatResponse,
  type ConflictingSource,
  type Disagreement,
  type HealthStatus,
  type IndexedDocument,
  type Source,
} from './api'

type ViewMode = 'home' | 'workspace' | 'architecture'

type ChatItem = ChatMessage & {
  sources?: Source[]
  confidence?: string
  refusal?: boolean
  model?: string
  timestamp?: string
  standalone_query?: string
  has_disagreements?: boolean
  disagreements?: Disagreement[]
}

const SAMPLE_PROMPTS = [
  'What are the key responsibilities and technical skills required for the Software Developer role?',
  'Summarize the qualifications and duties for the Data Quality Assurance position.',
  'Compare the required experience between the Software Developer and Data Quality Assurance roles.',
  'What educational background is mentioned across the uploaded job descriptions?',
]

function renderHighlightedSnippet(snippet: string) {
  const parts = snippet.split(/(\*\*.*?\*\*)/g)
  return parts.map((part, idx) => {
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      return (
        <mark key={idx} className="snippet-highlight">
          {part.slice(2, -2)}
        </mark>
      )
    }
    return <span key={idx}>{part}</span>
  })
}

function renderFormattedAnswer(content: string) {
  const lines = content.split('\n')
  return lines.map((line, lineIdx) => {
    const trimmed = line.trim()
    if (!trimmed) {
      return <div key={lineIdx} className="answer-spacer" />
    }
    const isBullet = /^[-*•]\s+/.test(trimmed)
    const cleanLine = isBullet ? trimmed.replace(/^[-*•]\s+/, '') : line

    // Highlight inline citations like [S1], [S2], [S1, S2] and **bold** text
    const tokens = cleanLine.split(/(\[S?\d+(?:,\s*S?\d+)*\]|\*\*.*?\*\*)/g)
    const renderedTokens = tokens.map((token, tIdx) => {
      if (/^\[S?\d+(?:,\s*S?\d+)*\]$/.test(token)) {
        return (
          <span key={tIdx} className="citation-pill">
            {token}
          </span>
        )
      }
      if (token.startsWith('**') && token.endsWith('**') && token.length > 4) {
        return <strong key={tIdx}>{token.slice(2, -2)}</strong>
      }
      return <span key={tIdx}>{token}</span>
    })

    if (isBullet) {
      return (
        <div key={lineIdx} className="answer-bullet">
          <span className="bullet-dot">•</span>
          <div>{renderedTokens}</div>
        </div>
      )
    }

    return (
      <p key={lineIdx} className="answer-paragraph">
        {renderedTokens}
      </p>
    )
  })
}

export default function App() {
  const [view, setView] = useState<ViewMode>('home')
  const [status, setStatus] = useState<HealthStatus | null>(null)
  const [files, setFiles] = useState<File[]>([])
  const [history, setHistory] = useState<ChatItem[]>([])
  const [question, setQuestion] = useState('')
  const [topK, setTopK] = useState<number>(5)
  const [busy, setBusy] = useState(false)
  const [uploadTask, setUploadTask] = useState<{
    jobId: string
    percent: number
    stage: string
  } | null>(null)
  const [toast, setToast] = useState<{ text: string; type: 'info' | 'error' | 'success' } | null>(
    null,
  )
  const [isListening, setIsListening] = useState(false)
  const [speakingIndex, setSpeakingIndex] = useState<number | null>(null)
  const chatEndRef = useRef<HTMLDivElement | null>(null)
  const recognitionRef = useRef<any>(null)

  function toggleSpeechRecognition() {
    const SpeechRecognitionClass =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SpeechRecognitionClass) {
      setToast({
        text: 'Speech recognition is not supported in this browser.',
        type: 'info',
      })
      return
    }

    if (isListening) {
      if (recognitionRef.current) {
        recognitionRef.current.stop()
      }
      setIsListening(false)
      return
    }

    try {
      const recognition = new SpeechRecognitionClass()
      recognition.continuous = false
      recognition.interimResults = false
      recognition.lang = 'en-US'

      recognition.onstart = () => {
        setIsListening(true)
      }

      recognition.onresult = (event: any) => {
        const transcript = event.results?.[0]?.[0]?.transcript
        if (transcript) {
          setQuestion((prev) => (prev ? `${prev} ${transcript}` : transcript))
        }
      }

      recognition.onerror = () => {
        setIsListening(false)
      }

      recognition.onend = () => {
        setIsListening(false)
      }

      recognitionRef.current = recognition
      recognition.start()
    } catch {
      setIsListening(false)
      setToast({
        text: 'Could not access microphone.',
        type: 'error',
      })
    }
  }

  function toggleSpeak(index: number, text: string) {
    if (!('speechSynthesis' in window)) {
      setToast({
        text: 'Text-to-speech is not supported in this browser.',
        type: 'info',
      })
      return
    }

    if (speakingIndex === index) {
      window.speechSynthesis.cancel()
      setSpeakingIndex(null)
      return
    }

    window.speechSynthesis.cancel()
    const cleanText = text
      .replace(/\[S?\d+(?:,\s*S?\d+)*\]/g, '')
      .replace(/\*\*/g, '')
      .replace(/^[-*•]\s+/gm, '')
      .trim()

    const utterance = new SpeechSynthesisUtterance(cleanText)
    utterance.onend = () => setSpeakingIndex(null)
    utterance.onerror = () => setSpeakingIndex(null)
    setSpeakingIndex(index)
    window.speechSynthesis.speak(utterance)
  }

  async function refreshHealth() {
    try {
      const data = await getHealth()
      setStatus(data)
    } catch (error: unknown) {
      setToast({
        text: error instanceof Error ? error.message : 'Unable to connect to backend API',
        type: 'error',
      })
    }
  }

  useEffect(() => {
    refreshHealth()
  }, [])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [history, busy])

  const indexedDocs: IndexedDocument[] = useMemo(() => status?.documents ?? [], [status])
  const storedChunks = status?.stored_chunks ?? 0
  const activeModel = status?.model || 'gemma4:e4b'
  const activeProvider = status?.provider || 'ollama'

  function handleFileSelection(event: React.ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(event.target.files ?? [])
    setFiles(selected)
  }

  async function handleUpload() {
    if (!files.length) {
      setToast({ text: 'Choose one or more PDF, DOCX, or TXT files first.', type: 'info' })
      return
    }
    setBusy(true)
    setToast(null)
    try {
      const initRes = await uploadDocuments(files)
      setFiles([])
      setUploadTask({ jobId: initRes.job_id, percent: 5, stage: 'Queued for processing...' })

      let finished = false
      while (!finished) {
        await new Promise((r) => setTimeout(r, 400))
        const task = await getTaskStatus(initRes.job_id)
        setUploadTask({
          jobId: task.job_id,
          percent: task.progress_percent,
          stage: `Stage: ${task.stage.toUpperCase()} (${task.progress_percent}%)`,
        })

        if (task.status === 'completed') {
          finished = true
          await refreshHealth()
          setToast({
            text: `Documents processed and indexed successfully (${task.stored_chunks ?? 0} total chunks).`,
            type: 'success',
          })
          setView('workspace')
        } else if (task.status === 'failed') {
          finished = true
          throw new Error(task.error || 'Document ingestion failed')
        }
      }
    } catch (error: unknown) {
      setToast({
        text: error instanceof Error ? error.message : 'Upload failed',
        type: 'error',
      })
    } finally {
      setBusy(false)
      setUploadTask(null)
    }
  }

  async function handleLoadSampleDocs(switchToWorkspace = true) {
    setBusy(true)
    setToast(null)
    try {
      const result = await loadSampleDocuments()
      await refreshHealth()
      setToast({
        text: `${result.message} Indexed ${result.chunks_indexed} chunks.`,
        type: 'success',
      })
      if (switchToWorkspace) {
        setView('workspace')
      }
    } catch (error: unknown) {
      setToast({
        text: error instanceof Error ? error.message : 'Failed to load sample documents',
        type: 'error',
      })
    } finally {
      setBusy(false)
    }
  }

  async function handleDeleteDoc(sourceName: string) {
    setBusy(true)
    setToast(null)
    try {
      const result = await deleteDocument(sourceName)
      await refreshHealth()
      setToast({ text: result.message, type: 'info' })
    } catch (error: unknown) {
      setToast({
        text: error instanceof Error ? error.message : 'Failed to remove document',
        type: 'error',
      })
    } finally {
      setBusy(false)
    }
  }

  async function handleClear() {
    setBusy(true)
    setToast(null)
    try {
      await clearDocuments()
      setHistory([])
      setFiles([])
      await refreshHealth()
      setToast({ text: 'Vector store and uploaded documents cleared.', type: 'info' })
    } catch (error: unknown) {
      setToast({
        text: error instanceof Error ? error.message : 'Clear failed',
        type: 'error',
      })
    } finally {
      setBusy(false)
    }
  }

  async function submitQuestion(rawQuestion: string) {
    const trimmed = rawQuestion.trim()
    if (!trimmed || busy) return

    if (view !== 'workspace') {
      setView('workspace')
    }

    setBusy(true)
    setToast(null)

    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    const userMessage: ChatItem = { role: 'user', content: trimmed, timestamp: now }
    const priorHistory: ChatMessage[] = history.map((h) => ({
      role: h.role,
      content: h.content,
    }))
    const nextHistory = [...history, userMessage]
    setHistory(nextHistory)
    setQuestion('')

    try {
      const result = await askQuestion(trimmed, priorHistory, topK)
      const replyTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      setHistory([
        ...nextHistory,
        {
          role: 'assistant',
          content: result.answer,
          sources: result.sources,
          confidence: result.confidence,
          refusal: result.refusal,
          model: result.model || activeModel,
          timestamp: replyTime,
          standalone_query: result.standalone_query,
          has_disagreements: result.has_disagreements,
          disagreements: result.disagreements,
        },
      ])
      if (result.refusal) {
        setToast({
          text: 'Grounded refusal: The answer was not found in the uploaded documents.',
          type: 'info',
        })
      }
    } catch (error: unknown) {
      setToast({
        text: error instanceof Error ? error.message : 'Chat request failed',
        type: 'error',
      })
    } finally {
      setBusy(false)
    }
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submitQuestion(question)
    }
  }

  return (
    <div className="app-root">
      {/* Top Navigation Header */}
      <header className="navbar">
        <div className="navbar-brand" onClick={() => setView('home')} role="button" tabIndex={0}>
          <div className="brand-logo">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>
          <div>
            <span className="brand-name">AI Knowledge Assistant</span>
            <span className="brand-tag">Local RAG Studio · Apple Silicon Optimized</span>
          </div>
        </div>

        <nav className="navbar-tabs">
          <button
            className={`nav-tab ${view === 'home' ? 'active' : ''}`}
            onClick={() => setView('home')}
          >
            Home
          </button>
          <button
            className={`nav-tab ${view === 'workspace' ? 'active' : ''}`}
            onClick={() => setView('workspace')}
          >
            RAG Workspace
            {storedChunks > 0 && <span className="tab-badge">{storedChunks}</span>}
          </button>
          <button
            className={`nav-tab ${view === 'architecture' ? 'active' : ''}`}
            onClick={() => setView('architecture')}
          >
            System Architecture
          </button>
        </nav>

        <div className="navbar-status">
          <div className="status-pill">
            <span className={`status-dot ${status?.status === 'ok' ? 'online' : 'offline'}`} />
            <span>
              <strong>{activeProvider.toUpperCase()}</strong> · <code>{activeModel}</code>
            </span>
          </div>
          <div className="status-pill subtle">
            <span>
              {indexedDocs.length} doc{indexedDocs.length === 1 ? '' : 's'} · {storedChunks} chunks
            </span>
          </div>
        </div>
      </header>

      {/* Toast Notification Bar */}
      {toast && (
        <div className={`toast-banner ${toast.type}`}>
          <span>{toast.text}</span>
          <button className="toast-close" onClick={() => setToast(null)}>
            ×
          </button>
        </div>
      )}

      {/* VIEW 1: HOMEPAGE */}
      {view === 'home' && (
        <main className="home-container">
          <section className="hero-section">
            <div className="hero-badge">
              <span className="pulse-dot" />
              Powered by Local Ollama <code>{activeModel}</code> &amp; ChromaDB Vector Search
            </div>
            <h1 className="hero-title">
              Grounded Document Intelligence,{' '}
              <span className="gradient-text">Running Locally on Your Mac.</span>
            </h1>
            <p className="hero-subtitle">
              Upload PDFs (with automatic table &amp; Tesseract OCR extraction), Word documents, and
              text files. Ask complex questions and receive zero-hallucination answers backed by
              sentence-level highlighted citations.
            </p>

            <div className="hero-actions">
              <button className="btn-primary btn-lg" onClick={() => setView('workspace')}>
                Launch RAG Workspace →
              </button>
              <button
                className="btn-secondary btn-lg"
                onClick={() => handleLoadSampleDocs(true)}
                disabled={busy}
              >
                {busy ? 'Indexing Sample Docs...' : '⚡ Load Sample Docs & Try Now'}
              </button>
            </div>

            {/* Live Telemetry Strip */}
            <div className="telemetry-grid">
              <div className="telemetry-card">
                <span className="telemetry-label">Active LLM Engine</span>
                <strong className="telemetry-value">{activeModel}</strong>
                <span className="telemetry-meta">Provider: {activeProvider.toUpperCase()}</span>
              </div>
              <div className="telemetry-card">
                <span className="telemetry-label">Embedding Model</span>
                <strong className="telemetry-value">all-MiniLM-L6-v2</strong>
                <span className="telemetry-meta">384-dim Cosine HNSW</span>
              </div>
              <div className="telemetry-card">
                <span className="telemetry-label">Indexed Documents</span>
                <strong className="telemetry-value">{indexedDocs.length} Files</strong>
                <span className="telemetry-meta">PDF · DOCX · TXT</span>
              </div>
              <div className="telemetry-card">
                <span className="telemetry-label">Vector Knowledge Base</span>
                <strong className="telemetry-value">{storedChunks} Chunks</strong>
                <span className="telemetry-meta">Persistent ChromaDB</span>
              </div>
            </div>
          </section>

          {/* How It Works - 4 Stage Pipeline */}
          <section className="home-section">
            <div className="section-header">
              <span className="eyebrow">End-to-End Pipeline</span>
              <h2>How the Retrieval-Augmented Generation Engine Works</h2>
              <p className="muted">
                Designed for strict factual grounding—every claim is traced back to the exact page,
                section, and sentence in your uploaded files.
              </p>
            </div>

            <div className="pipeline-grid">
              <div className="pipeline-card">
                <div className="step-number">01</div>
                <h3>Multi-Format Ingestion</h3>
                <p>
                  Parses native PDFs and tables via <code>PyMuPDF</code>, falls back to{' '}
                  <code>Tesseract OCR</code> for scanned pages, and extracts headings &amp; tables
                  from <code>.docx</code> and <code>.txt</code> files.
                </p>
              </div>
              <div className="pipeline-card">
                <div className="step-number">02</div>
                <h3>Semantic Chunking &amp; Indexing</h3>
                <p>
                  Splits documents into 1,000-character overlapping windows (150-char overlap) and
                  encodes 384-dimensional vectors into local persistent <code>ChromaDB</code>.
                </p>
              </div>
              <div className="pipeline-card">
                <div className="step-number">03</div>
                <h3>Top-K Cosine Retrieval</h3>
                <p>
                  Embeds your question in real time and retrieves the top-{topK} most semantically
                  relevant chunks along with source filename, page number, and section metadata.
                </p>
              </div>
              <div className="pipeline-card">
                <div className="step-number">04</div>
                <h3>Cited Generation &amp; Highlighting</h3>
                <p>
                  <code>{activeModel}</code> synthesizes a strictly grounded answer with{' '}
                  <code>[S1]</code> citations, while the post-processor highlights matching
                  keywords in source snippets.
                </p>
              </div>
            </div>
          </section>

          {/* Quick Start Sample Prompts */}
          <section className="home-section">
            <div className="section-header">
              <span className="eyebrow">Interactive Demo</span>
              <h2>Try Asking a Grounded Question</h2>
              <p className="muted">
                Click any prompt below to jump straight into the workspace (make sure you have
                uploaded documents or loaded the bundled sample PDFs first).
              </p>
            </div>
            <div className="prompt-showcase-grid">
              {SAMPLE_PROMPTS.map((sample, idx) => (
                <button
                  key={idx}
                  className="prompt-showcase-card"
                  onClick={async () => {
                    if (storedChunks === 0) {
                      await handleLoadSampleDocs(true)
                    } else {
                      setView('workspace')
                    }
                    setQuestion(sample)
                  }}
                >
                  <span className="prompt-icon">💬</span>
                  <span className="prompt-text">{sample}</span>
                  <span className="prompt-arrow">Ask →</span>
                </button>
              ))}
            </div>
          </section>
        </main>
      )}

      {/* VIEW 2: RAG WORKSPACE */}
      {view === 'workspace' && (
        <div className="workspace-shell">
          <aside className="sidebar">
            <section className="panel">
              <div className="panel-header">
                <h2>Knowledge Base</h2>
                <span className="badge">{storedChunks} chunks</span>
              </div>
              <p className="panel-desc">
                Upload PDF, DOCX, or TXT documents to index them into local ChromaDB storage.
              </p>

              <label className="dropzone">
                <input
                  type="file"
                  multiple
                  accept=".pdf,.docx,.txt"
                  onChange={handleFileSelection}
                />
                <div className="dropzone-icon">📄</div>
                <div className="dropzone-title">
                  {files.length
                    ? `${files.length} file(s) ready to upload`
                    : 'Click to select PDF, DOCX, or TXT'}
                </div>
                <div className="dropzone-sub">
                  Supports native text, tables, and scanned PDF OCR
                </div>
              </label>

              {files.length > 0 && (
                <div className="selected-files-list">
                  {files.map((f, i) => (
                    <div key={i} className="selected-file-pill">
                      <span>{f.name}</span>
                      <small>{(f.size / 1024).toFixed(1)} KB</small>
                    </div>
                  ))}
                </div>
              )}

              <div className="button-stack">
                <button className="btn-primary full-width" onClick={handleUpload} disabled={busy}>
                  {busy ? (uploadTask ? 'Processing Ingestion...' : 'Uploading...') : 'Upload & Index Files'}
                </button>
                {uploadTask && (
                  <div style={{ marginTop: '10px', padding: '10px 12px', background: '#0f172a', borderRadius: '8px', border: '1px solid #1e293b' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#94a3b8', marginBottom: '6px' }}>
                      <span style={{ fontWeight: 500 }}>{uploadTask.stage}</span>
                      <span style={{ fontWeight: 600, color: '#38bdf8' }}>{uploadTask.percent}%</span>
                    </div>
                    <div style={{ height: '6px', background: '#1e293b', borderRadius: '3px', overflow: 'hidden' }}>
                      <div style={{ width: `${uploadTask.percent}%`, height: '100%', background: 'linear-gradient(90deg, #38bdf8, #818cf8)', transition: 'width 0.3s ease' }} />
                    </div>
                  </div>
                )}
                <div className="button-row-split">
                  <button
                    className="btn-secondary"
                    onClick={() => handleLoadSampleDocs(false)}
                    disabled={busy}
                    title="Index the bundled Sample_doc PDFs"
                  >
                    ⚡ Load Sample PDFs
                  </button>
                  <button
                    className="btn-danger-outline"
                    onClick={handleClear}
                    disabled={busy || storedChunks === 0}
                  >
                    Clear All
                  </button>
                </div>
              </div>
            </section>

            {/* Indexed Documents Inventory */}
            <section className="panel">
              <div className="panel-header">
                <h2>Indexed Documents</h2>
                <span className="badge-subtle">{indexedDocs.length} files</span>
              </div>

              {indexedDocs.length === 0 ? (
                <div className="empty-docs">
                  <p>No documents indexed yet.</p>
                  <small>Upload files above or click &ldquo;Load Sample PDFs&rdquo;.</small>
                </div>
              ) : (
                <div className="doc-list">
                  {indexedDocs.map((doc) => (
                    <div key={doc.source} className="doc-item">
                      <div className="doc-item-main">
                        <span className="doc-type-badge">{doc.source_type}</span>
                        <div className="doc-info">
                          <strong title={doc.source}>{doc.source}</strong>
                          <span>
                            {doc.chunks} chunks
                            {doc.page_count ? ` · ${doc.page_count} pg` : ''}
                            {doc.total_words ? ` · ${doc.total_words} words` : ''}
                          </span>
                        </div>
                      </div>
                      <button
                        className="doc-delete-btn"
                        title={`Remove ${doc.source}`}
                        onClick={() => handleDeleteDoc(doc.source)}
                        disabled={busy}
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </section>

            {/* Retrieval Settings */}
            <section className="panel">
              <div className="panel-header">
                <h2>Retrieval Tuning</h2>
                <span className="badge-subtle">Top-{topK}</span>
              </div>
              <label className="slider-label">
                <span>Retrieved Chunks per Query (`top_k`): {topK}</span>
                <input
                  type="range"
                  min={2}
                  max={10}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                />
              </label>
              <div className="model-summary-box">
                <div>
                  <span>LLM Provider</span>
                  <strong>{activeProvider.toUpperCase()}</strong>
                </div>
                <div>
                  <span>Active Model</span>
                  <strong>{activeModel}</strong>
                </div>
              </div>
            </section>
          </aside>

          {/* Main Chat Panel */}
          <main className="chat-workspace">
            <div className="chat-header">
              <div>
                <h2>Grounded Document Q&amp;A</h2>
                <p className="muted">
                  Answers are strictly synthesized from your {storedChunks} indexed chunks using{' '}
                  <code>{activeModel}</code>.
                </p>
              </div>
              <div className="chat-header-actions">
                {history.length > 0 && (
                  <button
                    className="btn-secondary btn-sm"
                    onClick={() => {
                      setHistory([])
                      setToast(null)
                    }}
                    disabled={busy}
                  >
                    Reset Conversation
                  </button>
                )}
              </div>
            </div>

            <div className="chat-thread">
              {history.length === 0 ? (
                <div className="chat-empty-state">
                  <div className="empty-icon">🧠</div>
                  <h3>Ask Anything About Your Documents</h3>
                  <p>
                    Every answer includes inline source markers (e.g. <code>[S1]</code>) and
                    expandable source cards with keyword-highlighted excerpts.
                  </p>
                  <div className="quick-chips">
                    {SAMPLE_PROMPTS.map((prompt, i) => (
                      <button
                        key={i}
                        className="quick-chip"
                        onClick={() => submitQuestion(prompt)}
                        disabled={busy}
                      >
                        {prompt}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                history.map((item, index) => (
                  <article key={`${item.role}-${index}`} className={`chat-bubble ${item.role}`}>
                    <div className="bubble-header">
                      <span className="bubble-role">
                        {item.role === 'user' ? 'You' : `Assistant (${item.model || activeModel})`}
                      </span>
                      <div className="bubble-badges">
                        {item.role === 'assistant' && (
                          <button
                            className={`btn-speak ${speakingIndex === index ? 'speaking' : ''}`}
                            title={speakingIndex === index ? 'Stop audio' : 'Read answer aloud'}
                            onClick={() => toggleSpeak(index, item.content)}
                          >
                            {speakingIndex === index ? '⏹ Stop' : '🔊 Listen'}
                          </button>
                        )}
                        {item.role === 'assistant' && item.confidence && (
                          <span className={`confidence-badge ${item.confidence}`}>
                            {item.refusal
                              ? 'Unverified / Refusal'
                              : `${item.confidence.toUpperCase()} CONFIDENCE`}
                          </span>
                        )}
                        {item.timestamp && <span className="bubble-time">{item.timestamp}</span>}
                      </div>
                    </div>

                    <div className="bubble-body">
                      {item.role === 'assistant' && item.standalone_query && (
                        <div className="standalone-query-pill">
                          <span>🔍 Standalone Retrieval Query:</span>
                          <em>&ldquo;{item.standalone_query}&rdquo;</em>
                        </div>
                      )}
                      {item.role === 'assistant' && item.has_disagreements && item.disagreements && (
                        <div className="disagreement-box">
                          <div className="disagreement-header">
                            <span>⚠️</span>
                            <span>Multi-Document Discrepancy Detected</span>
                          </div>
                          {item.disagreements.map((d, dIdx) => (
                            <div key={dIdx}>
                              <p className="disagreement-summary">{d.summary}</p>
                              <div className="disagreement-sources">
                                {d.conflicting_sources.map((cs, csIdx) => (
                                  <div key={csIdx}>
                                    <strong>{cs.source}</strong> (Page {cs.page ?? 'n/a'}): &ldquo;{cs.statement}&rdquo;
                                  </div>
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                      {item.role === 'assistant' ? (
                        renderFormattedAnswer(item.content)
                      ) : (
                        <p>{item.content}</p>
                      )}
                    </div>

                    {item.sources && item.sources.length > 0 && (
                      <details className="sources-drawer" open={index === history.length - 1}>
                        <summary>
                          <span>
                            📚 Verified Source Excerpts ({item.sources.length} cited chunk
                            {item.sources.length === 1 ? '' : 's'})
                          </span>
                        </summary>
                        <div className="source-cards-grid">
                          {item.sources.map((src, sIdx) => (
                            <div key={`${src.chunk_id ?? sIdx}`} className="source-card">
                              <div className="source-card-header">
                                <div className="source-title-group">
                                  <span className="cite-tag">[S{src.cite_index ?? sIdx + 1}]</span>
                                  <strong>{src.source}</strong>
                                </div>
                                <div className="source-meta-pills">
                                  {src.verified != null && (
                                    <span
                                      className={`meta-pill ${
                                        src.verified ? 'verified-pill' : 'unverified-pill'
                                      }`}
                                    >
                                      {src.verified ? '✓ Verified' : '⚠️ Unverified'}
                                    </span>
                                  )}
                                  {src.page != null && (
                                    <span className="meta-pill">Page {src.page}</span>
                                  )}
                                  {src.section && (
                                    <span className="meta-pill">{src.section}</span>
                                  )}
                                  {src.rerank_score != null && (
                                    <span className="meta-pill rerank-pill">
                                      Rerank {Math.round(src.rerank_score * 100)}%
                                    </span>
                                  )}
                                  {src.similarity != null && (
                                    <span className="meta-pill match-pill">
                                      {Math.round(src.similarity * 100)}% match
                                    </span>
                                  )}
                                </div>
                              </div>
                              <div className="source-snippet">
                                {renderHighlightedSnippet(src.snippet)}
                              </div>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}
                  </article>
                ))
              )}

              {busy && (
                <div className="chat-bubble assistant loading-bubble">
                  <div className="typing-indicator">
                    <span />
                    <span />
                    <span />
                  </div>
                  <span>
                    Retrieving chunks &amp; generating grounded response with{' '}
                    <code>{activeModel}</code>...
                  </span>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <div className="composer-bar">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  storedChunks > 0
                    ? 'Ask a question about your uploaded documents... (Press Enter to send, Shift+Enter for new line)'
                    : 'Upload or load sample documents first, or type a question...'
                }
                rows={3}
                disabled={busy}
              />
              <div className="composer-actions">
                <span className="composer-hint">
                  Strict Grounding Mode · Model: <code>{activeModel}</code> · Top-{topK} chunks
                </span>
                <div style={{ display: 'flex', gap: '0.6rem', alignItems: 'center' }}>
                  <button
                    type="button"
                    className={`btn-voice ${isListening ? 'listening' : ''}`}
                    onClick={toggleSpeechRecognition}
                    disabled={busy}
                    title={isListening ? 'Stop listening' : 'Start voice input (Speech to Text)'}
                  >
                    <span>🎙️</span>
                    <span>{isListening ? 'Listening...' : 'Voice'}</span>
                  </button>
                  <button
                    className="btn-primary"
                    onClick={() => submitQuestion(question)}
                    disabled={busy || !question.trim()}
                  >
                    {busy ? 'Generating...' : 'Send Question →'}
                  </button>
                </div>
              </div>
            </div>
          </main>
        </div>
      )}

      {/* VIEW 3: SYSTEM ARCHITECTURE OVERVIEW */}
      {view === 'architecture' && (
        <main className="home-container">
          <section className="home-section">
            <div className="section-header">
              <span className="eyebrow">System Blueprint</span>
              <h2>Architecture, Data Flow &amp; Storage Schema</h2>
              <p className="muted">
                Complete technical overview of the AI Knowledge Assistant stack running on your Mac.
              </p>
            </div>

            <div className="arch-grid">
              <div className="arch-card">
                <h3>1. Frontend Layer (`React 18 + TypeScript + Vite`)</h3>
                <ul>
                  <li>
                    <strong>Views:</strong> Interactive Homepage, Split-Pane RAG Studio, and System
                    Architecture Blueprint.
                  </li>
                  <li>
                    <strong>State &amp; API Client (`src/api.ts`):</strong> Typed REST calls to
                    FastAPI (<code>/health</code>, <code>/documents</code>,{' '}
                    <code>/documents/upload</code>, <code>/documents/sample</code>,{' '}
                    <code>/chat</code>).
                  </li>
                  <li>
                    <strong>Citation &amp; Highlighting Renderer:</strong> Parses inline{' '}
                    <code>[S1]</code> markers and renders <code>**highlighted**</code> sentence
                    matches inside source verification drawers.
                  </li>
                </ul>
              </div>

              <div className="arch-card">
                <h3>2. API Gateway (`FastAPI + Uvicorn`)</h3>
                <ul>
                  <li>
                    <strong>Endpoints (`backend/main.py`):</strong> Handles multipart document
                    uploads, sample PDF ingestion, per-document deletion, collection reset, and
                    conversational RAG queries.
                  </li>
                  <li>
                    <strong>Validation:</strong> Pydantic request models (<code>ChatRequest</code>{' '}
                    with configurable <code>top_k</code> and conversation history).
                  </li>
                </ul>
              </div>

              <div className="arch-card">
                <h3>3. Ingestion &amp; Vector Store (`core/ingestion.py`, `core/vector_store.py`)</h3>
                <ul>
                  <li>
                    <strong>Multi-Modal Parsing:</strong> <code>PyMuPDF</code> text + table
                    extraction, <code>pytesseract</code> OCR fallback for scanned PDFs,{' '}
                    <code>python-docx</code> section/table parser.
                  </li>
                  <li>
                    <strong>Vector Indexing:</strong> Singleton-cached{' '}
                    <code>SentenceTransformerEmbeddingFunction(&quot;all-MiniLM-L6-v2&quot;)</code>{' '}
                    with persistent <code>ChromaDB</code> HNSW cosine index and automatic per-file
                    deduplication.
                  </li>
                </ul>
              </div>

              <div className="arch-card">
                <h3>4. RAG Engine &amp; LLM Adapters (`core/rag_engine.py`, `core/ai_service.py`)</h3>
                <ul>
                  <li>
                    <strong>Apple Silicon Ollama Default:</strong> Uses local{' '}
                    <code>gemma4:e4b</code> with HTTP connection pooling,{' '}
                    <code>keep_alive=&quot;30m&quot;</code> in Unified Memory, and low-latency
                    non-thinking inference.
                  </li>
                  <li>
                    <strong>Multi-Provider Adapter:</strong> Seamlessly switchable between{' '}
                    <code>Ollama</code>, <code>Google Gemini</code>, and <code>Groq</code> via{' '}
                    <code>.env</code>.
                  </li>
                </ul>
              </div>
            </div>
          </section>
        </main>
      )}
    </div>
  )
}
