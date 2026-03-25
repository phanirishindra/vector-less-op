"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field
from typing import Literal


class QueryRequest(BaseModel):
    """Query request."""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    stream: bool = Field(default=False, description="Enable streaming response")


class QueryResponse(BaseModel):
    """Query response."""
    query: str
    answer: str
    layer_used: Literal["sieve", "router", "explorer", "bm25_fallback"]
    chunks_used: int
    success: bool


class IndexRequest(BaseModel):
    """Index request."""
    source_name: str = Field(..., min_length=1, max_length=100)
    markdown_dir: str | None = None


class IndexResponse(BaseModel):
    """Index response."""
    source_name: str
    chunks_indexed: int
    signposts_generated: int
    index_path: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    llm_connected: bool
    index_loaded: bool
    index_entries: int


class CrawlRequest(BaseModel):
    """Crawl request."""
    url: str = Field(..., description="Seed URL to crawl")
    max_depth: int = Field(default=2, ge=1, le=5)
    max_pages: int = Field(default=50, ge=1, le=500)
    render_js: bool = Field(default=False)
