"""Tests for BM25 fallback."""
import pytest
from datetime import datetime, timezone
from vnull.retrieval.bm25_fallback import BM25Fallback
from vnull.indexer.toc_builder import TableOfContents, ToCEntry


def test_bm25_search():
    toc = TableOfContents(
        entries=[
            ToCEntry(
                chunk_id="auth1",
                dense_signpost="[Authentication] + [OAuth, JWT] + [How to auth?]",
                first_sentence="Authentication uses OAuth2.",
                last_sentence="Tokens expire after 1 hour.",
                raw_markdown="# Auth\n\nAuthentication uses OAuth2 and JWT tokens.",
                header="Auth",
                header_level=1,
            ),
            ToCEntry(
                chunk_id="db1",
                dense_signpost="[Database] + [PostgreSQL, indexes] + [How to query?]",
                first_sentence="Database uses PostgreSQL.",
                last_sentence="Indexes improve performance.",
                raw_markdown="# Database\n\nPostgreSQL with indexes.",
                header="Database",
                header_level=1,
            ),
        ],
        created_at=datetime.now(timezone.utc),
        source_name="test",
    )
    
    fallback = BM25Fallback(toc)
    result = fallback.search("OAuth authentication")
    
    assert result.top_entry is not None
    assert result.top_entry.chunk_id == "auth1"
