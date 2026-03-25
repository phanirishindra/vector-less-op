"""Tests for Markdown chunker."""
import pytest
from vnull.indexer.chunker import MarkdownChunker


def test_chunk_by_headers():
    md = """# Title

Intro paragraph.

## Section 1

Section 1 content here.

## Section 2

Section 2 content here.
"""
    chunker = MarkdownChunker(min_chunk_size=10)
    chunks = chunker.chunk(md)
    assert len(chunks) >= 2
    assert any("Section 1" in c.header for c in chunks)


def test_chunk_extracts_bookends():
    md = """# Test

First sentence here. Middle content. Last sentence here.
"""
    chunker = MarkdownChunker(min_chunk_size=10)
    chunks = chunker.chunk(md)
    assert chunks[0].first_sentence
    assert chunks[0].last_sentence


def test_chunk_generates_ids():
    md = "# Header\n\nContent here."
    chunker = MarkdownChunker(min_chunk_size=5)
    chunks = chunker.chunk(md)
    assert all(c.chunk_id for c in chunks)
