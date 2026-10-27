"""Unit tests for Ticket 03: Hierarchical Parent-Child Chunking & Dual Storage Architecture."""

from __future__ import annotations

import pytest
from core.hierarchical_chunker import (
    HierarchicalChunker,
    ParentDocStore,
)


def test_hierarchical_chunker_splits_into_parent_and_child_chunks():
    """HierarchicalChunker should produce linked parent (~1200 tokens) and child (~250 tokens) chunks."""
    # Generate ~2000 words text
    sample_section = (
        "Leave Policy Overview. Employees in good standing are entitled to multiple forms of leave. "
        "Annual leave is accrued monthly at a rate of 1.5 days up to a maximum of 18 days per calendar year. "
        "Probationary employees must complete 6 months before taking paid vacation leave. "
        "Sick leave requires a medical certificate if extending beyond 3 consecutive days. "
        "Parental leave provides 16 weeks of paid leave for primary caregivers and 4 weeks for secondary caregivers. "
    ) * 10

    metadata = {"source": "Employee_Handbook.pdf", "page": 4, "section": "Leave Policy"}
    chunker = HierarchicalChunker(parent_chunk_size=1500, child_chunk_size=300, child_overlap=50)

    parents, children = chunker.split_text_hierarchical(sample_section, base_metadata=metadata)

    assert len(parents) >= 1, "Must generate at least one parent chunk"
    assert len(children) > len(parents), "Children count must exceed parent count"

    parent_ids = {p.id for p in parents}
    for child in children:
        assert "parent_id" in child.metadata, "Child chunk must reference parent_id"
        assert child.metadata["parent_id"] in parent_ids, (
            f"Child parent_id {child.metadata['parent_id']} not found in generated parents"
        )
        assert len(child.page_content) <= 450, "Child chunks must remain granular for high vector similarity"


def test_parent_docstore_crud(tmp_path):
    """ParentDocStore should store and retrieve parent chunks by parent_id from SQLite."""
    db_path = tmp_path / "test_docstore.db"
    store = ParentDocStore(db_path=db_path)

    chunker = HierarchicalChunker(parent_chunk_size=1000, child_chunk_size=250)
    parents, _ = chunker.split_text_hierarchical(
        "Important parent section content.", base_metadata={"source": "doc.pdf", "page": 1}
    )

    store.save_parents(parents)

    # Verify retrieval
    for p in parents:
        retrieved = store.get_parent(p.id)
        assert retrieved is not None
        assert retrieved.content == p.content
        assert retrieved.metadata["source"] == "doc.pdf"

    # Nonexistent parent returns None
    assert store.get_parent("nonexistent_parent_id") is None
