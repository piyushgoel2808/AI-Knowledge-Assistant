"""Document ingestion, table extraction, OCR fallback, and chunking utilities."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pymupdf
import pytesseract
from PIL import Image
from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import AppConfig, load_config


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@dataclass(frozen=True)
class SourceChunk:
    """Lightweight representation of an extracted text chunk."""

    text: str
    metadata: dict[str, object]


def is_supported_file(file_path: str | Path) -> bool:
    """Return True when the file extension is one we can ingest."""

    return Path(file_path).suffix.lower() in SUPPORTED_EXTENSIONS


def load_documents(file_paths: Sequence[str | Path], config: AppConfig | None = None) -> list[Document]:
    """Load and split multiple files into LangChain documents."""

    chunks: list[Document] = []
    for file_path in file_paths:
        chunks.extend(ingest_file(file_path, config=config))
    return chunks


def ingest_file(file_path: str | Path, config: AppConfig | None = None) -> list[Document]:
    """Load a supported file and split it into metadata-rich chunks."""

    path = Path(file_path)
    if not is_supported_file(path):
        raise ValueError(f"Unsupported file type: {path.suffix}")

    resolved_config = config or load_config()
    raw_chunks = _extract_source_chunks(path)
    documents: list[Document] = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=resolved_config.chunk_size,
        chunk_overlap=resolved_config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for raw_chunk in raw_chunks:
        split_texts = splitter.split_text(raw_chunk.text)
        if not split_texts:
            continue

        for index, split_text in enumerate(split_texts, start=1):
            cleaned_text = split_text.strip()
            if not cleaned_text:
                continue
            metadata = dict(raw_chunk.metadata)
            metadata["chunk_index"] = index
            metadata["chunk_count"] = len(split_texts)
            metadata["char_count"] = len(cleaned_text)
            metadata["word_count"] = len(cleaned_text.split())
            documents.append(Document(page_content=cleaned_text, metadata=metadata))

    return documents


def _extract_source_chunks(path: Path) -> list[SourceChunk]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_chunks(path)
    if suffix == ".docx":
        return _extract_docx_chunks(path)
    if suffix == ".txt":
        return _extract_txt_chunks(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _extract_pdf_chunks(path: Path) -> list[SourceChunk]:
    chunks: list[SourceChunk] = []

    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            # Extract structured tables if present on the PDF page
            table_blocks: list[str] = []
            try:
                finder = page.find_tables()
                for t_idx, table in enumerate(getattr(finder, "tables", []), start=1):
                    extracted = table.extract()
                    rows = []
                    for row in extracted:
                        if not row:
                            continue
                        cleaned_cells = [str(cell or "").replace("\n", " ").strip() for cell in row]
                        if any(cleaned_cells):
                            rows.append(" | ".join(cleaned_cells))
                    if len(rows) >= 2:
                        table_blocks.append(f"[Table {t_idx}]\n" + "\n".join(rows))
            except Exception:
                # Fallback gracefully if table detection is unsupported on a given page
                pass

            # --- OCR FALLBACK LOGIC ---
            # If standard text extraction returns nothing, the page is likely a scanned image
            if not text:
                try:
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
                    image_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(image_data))
                    text = pytesseract.image_to_string(image).strip()
                except Exception:
                    # Do not crash ingestion if Tesseract is not installed on host
                    text = ""

            if table_blocks:
                combined_tables = "\n\n".join(table_blocks)
                text = f"{text}\n\n{combined_tables}".strip() if text else combined_tables

            if not text:
                continue

            metadata = {
                "source": path.name,
                "file_path": str(path),
                "page": page_number,
                "section": _guess_section_from_text(text),
                "source_type": "pdf",
            }
            chunks.append(SourceChunk(text=text, metadata=metadata))

    return chunks


def _extract_docx_chunks(path: Path) -> list[SourceChunk]:
    doc = DocxDocument(path)
    chunks: list[SourceChunk] = []
    current_section = "Document"
    paragraph_buffer: list[str] = []

    def flush_buffer(section_name: str) -> None:
        nonlocal paragraph_buffer
        text = "\n".join(line for line in paragraph_buffer if line.strip()).strip()
        if text:
            chunks.append(
                SourceChunk(
                    text=text,
                    metadata={
                        "source": path.name,
                        "file_path": str(path),
                        "page": None,
                        "section": section_name,
                        "source_type": "docx",
                    },
                )
            )
        paragraph_buffer = []

    # --- PARAGRAPH EXTRACTION ---
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue

        style_name = getattr(paragraph.style, "name", "") or ""
        if style_name.lower().startswith("heading"):
            flush_buffer(current_section)
            current_section = text
            continue

        paragraph_buffer.append(text)

    flush_buffer(current_section)

    # --- TABLE EXTRACTION LOGIC ---
    for i, table in enumerate(doc.tables):
        table_lines = []
        for row in table.rows:
            row_data = [cell.text.replace("\n", " ").strip() for cell in row.cells]
            table_lines.append(" | ".join(row_data))

        if table_lines:
            markdown_table = "\n".join(table_lines)
            chunks.append(
                SourceChunk(
                    text=markdown_table,
                    metadata={
                        "source": path.name,
                        "file_path": str(path),
                        "page": None,
                        "section": f"Table {i + 1}",
                        "source_type": "docx",
                    },
                )
            )

    return chunks


def _extract_txt_chunks(path: Path) -> list[SourceChunk]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1", errors="replace").strip()

    if not text:
        return []

    return [
        SourceChunk(
            text=text,
            metadata={
                "source": path.name,
                "file_path": str(path),
                "page": None,
                "section": _guess_section_from_text(text),
                "source_type": "txt",
            },
        )
    ]


def _guess_section_from_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "Document"

    first_line = lines[0]
    if len(first_line) <= 120 and not first_line.endswith("."):
        return first_line
    return "Document"


def collect_supported_files(paths: Iterable[str | Path]) -> list[Path]:
    """Filter and normalize a set of file paths to supported inputs."""

    supported_paths: list[Path] = []
    for file_path in paths:
        path = Path(file_path)
        if path.is_file() and is_supported_file(path):
            supported_paths.append(path)
    return supported_paths