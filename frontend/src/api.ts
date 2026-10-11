export type Source = {
  source: string
  page?: number | null
  section?: string | null
  snippet: string
  chunk_id?: string
  distance?: number | null
  similarity?: number | null
  cite_index?: number
}

export type IndexedDocument = {
  source: string
  source_type: string
  chunks: number
  page_count?: number | null
  section_count?: number
  total_words?: number
}

export type HealthStatus = {
  status: string
  provider: string
  model?: string
  stored_chunks: number
  document_count?: number
  documents?: IndexedDocument[]
}

export type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
}

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

async function parseJsonResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.text()
    try {
      const parsed = JSON.parse(body) as { detail?: string }
      if (parsed.detail) {
        throw new Error(parsed.detail)
      }
    } catch (err) {
      if (err instanceof Error && err.message !== body) {
        throw err
      }
    }
    throw new Error(body || `Request failed with status ${response.status}`)
  }
  return response.json() as Promise<T>
}

export async function getHealth(): Promise<HealthStatus> {
  return parseJsonResponse<HealthStatus>(await fetch(`${API_URL}/health`))
}

export async function uploadDocuments(files: File[]) {
  const formData = new FormData()
  files.forEach((file) => formData.append('files', file))

  return parseJsonResponse<{
    message: string
    files: string[]
    chunks_indexed: number
    stored_chunks: number
    documents?: IndexedDocument[]
  }>(await fetch(`${API_URL}/documents/upload`, { method: 'POST', body: formData }))
}

export async function loadSampleDocuments() {
  return parseJsonResponse<{
    message: string
    files: string[]
    chunks_indexed: number
    stored_chunks: number
    documents?: IndexedDocument[]
  }>(await fetch(`${API_URL}/documents/sample`, { method: 'POST' }))
}

export async function deleteDocument(sourceName: string) {
  return parseJsonResponse<{
    message: string
    removed_chunks: number
    stored_chunks: number
    documents?: IndexedDocument[]
  }>(
    await fetch(`${API_URL}/documents/${encodeURIComponent(sourceName)}`, {
      method: 'DELETE',
    }),
  )
}

export async function clearDocuments() {
  return parseJsonResponse<{
    message: string
    stored_chunks?: number
    documents?: IndexedDocument[]
  }>(await fetch(`${API_URL}/documents/clear`, { method: 'POST' }))
}

export async function askQuestion(question: string, history: ChatMessage[], topK?: number) {
  return parseJsonResponse<{
    answer: string
    sources: Source[]
    confidence: string
    refusal: boolean
    model?: string
    provider?: string
  }>(
    await fetch(`${API_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, history, top_k: topK }),
    }),
  )
}
