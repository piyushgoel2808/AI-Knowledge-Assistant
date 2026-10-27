"""Unit tests for Ticket 02: Layout-Aware Reading Order & Markdown Table Parser."""

from __future__ import annotations

import pytest
from core.ingestion import format_markdown_table, prune_headers_footers, sort_blocks_by_column


def test_markdown_table_formatting():
    """format_markdown_table should generate valid GitHub-Flavored Markdown tables with header separators."""
    raw_table = [
        ["Plan Tier", "Annual Deductible", "Specialist Co-pay"],
        ["Standard", "$1,000", "$40"],
        ["Enhanced", "$500", "$20"],
    ]

    md_table = format_markdown_table(raw_table)

    lines = md_table.strip().split("\n")
    assert len(lines) == 4, f"Expected 4 lines (header, separator, 2 rows), got: {md_table}"
    assert lines[0] == "| Plan Tier | Annual Deductible | Specialist Co-pay |"
    assert lines[1] == "| --- | --- | --- |"
    assert lines[2] == "| Standard | $1,000 | $40 |"
    assert lines[3] == "| Enhanced | $500 | $20 |"


def test_column_sorting_preserves_reading_order():
    """sort_blocks_by_column should read all left-column blocks before right-column blocks."""
    # (x0, y0, x1, y1, text, block_no, block_type)
    blocks = [
        (50.0, 50.0, 250.0, 100.0, "Left column para 1.\n", 0, 0),
        (320.0, 50.0, 520.0, 100.0, "Right column para 1.\n", 1, 0),
        (50.0, 120.0, 250.0, 180.0, "Left column para 2.\n", 2, 0),
        (320.0, 120.0, 520.0, 180.0, "Right column para 2.\n", 3, 0),
    ]
    page_width = 600.0

    sorted_text = sort_blocks_by_column(blocks, page_width=page_width)

    # Left column must precede right column
    idx_left_1 = sorted_text.find("Left column para 1.")
    idx_left_2 = sorted_text.find("Left column para 2.")
    idx_right_1 = sorted_text.find("Right column para 1.")
    idx_right_2 = sorted_text.find("Right column para 2.")

    assert idx_left_1 != -1 and idx_left_2 != -1 and idx_right_1 != -1 and idx_right_2 != -1
    assert idx_left_1 < idx_left_2 < idx_right_1 < idx_right_2, (
        f"Columns were interleaved! Sequence was: {sorted_text}"
    )


def test_prune_headers_footers():
    """prune_headers_footers should remove running headers and footers from page blocks."""
    # Running header at y0=20 (top 5% of 800h page), body at y0=150, footer at y0=780 (bottom 5%)
    blocks = [
        (50.0, 20.0, 550.0, 35.0, "CONFIDENTIAL - ACME CORP 2026\n", 0, 0),
        (50.0, 150.0, 550.0, 300.0, "This is the actual policy content.\n", 1, 0),
        (50.0, 780.0, 550.0, 795.0, "Page 14 of 500\n", 2, 0),
    ]
    page_height = 800.0

    cleaned = prune_headers_footers(blocks, page_height=page_height)

    texts = [b[4].strip() for b in cleaned]
    assert "CONFIDENTIAL - ACME CORP 2026" not in texts
    assert "Page 14 of 500" not in texts
    assert "This is the actual policy content." in texts
