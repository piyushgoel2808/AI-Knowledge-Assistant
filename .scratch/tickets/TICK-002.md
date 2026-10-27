# TICK-002: Layout-Aware Reading Order & Markdown Table Parser

- **Status:** COMPLETED & VERIFIED
- **User Story:** US-002 (Structured Table & Multi-Column Question Answering)
- **Dependencies:** None

## Problem
PyPDF stripped page layout, flattening multi-column pages across columns and converting structured tables into unaligned whitespace soup, losing row/column relationships.

## Solution Implemented
1. Upgraded PyMuPDF (fitz) parser in `core/ingestion.py`.
2. Added `sort_blocks_by_column`: clusters text blocks by horizontal coordinates before reading order sorting to prevent cross-column bleed.
3. Added `format_markdown_table`: extracts table bounding boxes and renders true GitHub-Flavored Markdown tables (`| Col 1 | Col 2 |`).
4. Added `prune_headers_footers`: strips recurring running headers and page numbers based on margin thresholds.

## Files Touched
- `core/ingestion.py` (Updated)
- `tests/unit/test_layout_parser.py` (Created)

## Verification
`PYTHONPATH=. .venv/bin/pytest -v tests/unit/test_layout_parser.py` (3 passed).
