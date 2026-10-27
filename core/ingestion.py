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
from core.hierarchical_chunker import HierarchicalChunker, ParentDocStore


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
    """Load a supported file and split it into metadata-rich child chunks while persisting parent contexts."""

    path = Path(file_path)
    if not is_supported_file(path):
        raise ValueError(f"Unsupported file type: {path.suffix}")

    resolved_config = config or load_config()
    raw_chunks = _extract_source_chunks(path)
    documents: list[Document] = []

    # Use Hierarchical Chunker (Parents ~1200t, Children ~250-300t)
    chunker = HierarchicalChunker(
        parent_chunk_size=1500,
        parent_overlap=150,
        child_chunk_size=resolved_config.chunk_size,
        child_overlap=resolved_config.chunk_overlap,
    )
    doc_store = ParentDocStore()

    for raw_chunk in raw_chunks:
        parents, children = chunker.split_text_hierarchical(raw_chunk.text, base_metadata=raw_chunk.metadata)
        if parents:
            doc_store.save_parents(parents)
        documents.extend(children)

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


def format_markdown_table(rows: list[list[Any]]) -> str:
    """Format a 2D matrix of extracted table cells into GitHub-Flavored Markdown table syntax."""
    if not rows or len(rows) < 2:
        return ""

    cleaned_matrix: list[list[str]] = []
    for row in rows:
        cleaned_matrix.append([str(cell or "").replace("\n", " ").strip() for cell in row])

    col_count = max(len(r) for r in cleaned_matrix)
    if col_count == 0:
        return ""

    # Normalize row lengths
    normalized: list[list[str]] = []
    for r in cleaned_matrix:
        normalized.append(r + [""] * (col_count - len(r)))

    header_cells = normalized[0]
    header_line = "| " + " | ".join(header_cells) + " |"
    separator_line = "| " + " | ".join(["---"] * col_count) + " |"

    data_lines = []
    for r in normalized[1:]:
        data_lines.append("| " + " | ".join(r) + " |")

    return "\n".join([header_line, separator_line] + data_lines)


def prune_headers_footers(
    blocks: list[tuple],
    page_height: float,
    top_margin_ratio: float = 0.08,
    bottom_margin_ratio: float = 0.08,
) -> list[tuple]:
    """Prune running headers and footers appearing in page margin bands."""
    top_threshold = page_height * top_margin_ratio
    bottom_threshold = page_height * (1.0 - bottom_margin_ratio)

    body_blocks = []
    for b in blocks:
        # b = (x0, y0, x1, y1, text, block_no, block_type)
        y0, y1 = b[1], b[3]
        if y1 <= top_threshold:
            # Header block
            continue
        if y0 >= bottom_threshold:
            # Footer block
            continue
        body_blocks.append(b)

    return body_blocks


def sort_blocks_by_column(blocks: list[tuple], page_width: float) -> str:
    """Sort text blocks by geometric column layout (left-to-right columns, top-to-bottom)."""
    text_blocks = [b for b in blocks if len(b) >= 5 and (len(b) < 7 or b[6] == 0) and b[4].strip()]
    if not text_blocks:
        return ""

    mid_x = page_width * 0.5
    left_col = [b for b in text_blocks if b[0] < mid_x]
    right_col = [b for b in text_blocks if b[0] >= mid_x]

    # If substantial text exists on both halves, treat as multi-column
    is_multi_column = len(left_col) >= 2 and len(right_col) >= 2

    if is_multi_column:
        left_col.sort(key=lambda b: (b[1], b[0]))
        right_col.sort(key=lambda b: (b[1], b[0]))
        ordered_blocks = left_col + right_col
    else:
        text_blocks.sort(key=lambda b: (b[1], b[0]))
        ordered_blocks = text_blocks

    return "\n\n".join(b[4].strip() for b in ordered_blocks if b[4].strip())


def _extract_pdf_chunks(path: Path) -> list[SourceChunk]:
    chunks: list[SourceChunk] = []

    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_rect = page.rect
            raw_blocks = page.get_text("blocks")
            pruned_blocks = prune_headers_footers(raw_blocks, page_height=page_rect.height)
            text = sort_blocks_by_column(pruned_blocks, page_width=page_rect.width).strip()

            # Extract structured tables and format as GitHub-Flavored Markdown
            table_blocks: list[str] = []
            try:
                finder = page.find_tables()
                for t_idx, table in enumerate(getattr(finder, "tables", []), start=1):
                    extracted = table.extract()
                    md_table = format_markdown_table(extracted)
                    if md_table:
                        table_blocks.append(f"[Table {t_idx}]\n{md_table}")
            except Exception:
                pass

            # --- OCR FALLBACK LOGIC ---
            if not text:
                try:
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
                    image_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(image_data))
                    text = pytesseract.image_to_string(image).strip()
                except Exception:
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