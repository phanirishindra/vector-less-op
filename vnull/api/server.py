"""FastAPI server for Zero-Null RAG."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from vnull.api.schemas import QueryRequest, QueryResponse, IndexRequest, IndexResponse, HealthResponse
from vnull.core.config import settings
from vnull.core.logging import get_logger, configure_logging
from vnull.core.llm_client import LLMClient
from vnull.indexer.toc_builder import ToCBuilder
from vnull.retrieval.orchestrator import RetrievalOrchestrator

logger = get_logger(__name__)

_orchestrator: RetrievalOrchestrator | None = None
_llm: LLMClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _orchestrator, _llm
    configure_logging()
    logger.info("Starting Zero-Null RAG API")
    
    _llm = LLMClient()
    
    toc_files = list(settings.index_dir.glob("*.json"))
    if toc_files:
        builder = ToCBuilder()
        tocs = [builder.load(f) for f in toc_files]
        if len(tocs) == 1:
            toc = tocs[0]
        else:
            toc = builder.merge(tocs, "merged")
        _orchestrator = RetrievalOrchestrator(toc=toc, llm_client=_llm)
        logger.info(f"Loaded {len(toc.entries)} chunks from {len(toc_files)} index files")
    else:
        logger.warning("No index files found. Run indexing first.")
    
    yield
    
    if _llm:
        await _llm._async_client.close()
    logger.info("Shutting down")


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Zero-Null Vectorless RAG",
        description="Memory-efficient RAG without vector embeddings",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/health", response_model=HealthResponse)
    async def health():
        llm_ok = await _llm.health_check() if _llm else False
        return HealthResponse(
            status="healthy" if llm_ok else "degraded",
            llm_connected=llm_ok,
            index_loaded=_orchestrator is not None,
            index_entries=len(_orchestrator.toc.entries) if _orchestrator else 0,
        )
    
    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest):
        if not _orchestrator:
            raise HTTPException(status_code=503, detail="Index not loaded")
        
        result = await _orchestrator.retrieve(request.query)
        
        return QueryResponse(
            query=result.query,
            answer=result.answer,
            layer_used=result.layer_used,
            chunks_used=result.exploration_result.chunks_explored if result.exploration_result else 0,
            success=result.success,
        )
    
    @app.post("/query/stream")
    async def query_stream(request: QueryRequest):
        if not _orchestrator:
            raise HTTPException(status_code=503, detail="Index not loaded")
        
        async def generate() -> AsyncGenerator[str, None]:
            async for token in _orchestrator.stream_retrieve(request.query):
                yield token
        
        return StreamingResponse(generate(), media_type="text/plain")
    
    @app.get("/index/stats")
    async def index_stats():
        if not _orchestrator:
            raise HTTPException(status_code=503, detail="Index not loaded")
        
        toc = _orchestrator.toc
        return {
            "source_name": toc.source_name,
            "entry_count": len(toc.entries),
            "created_at": toc.created_at.isoformat(),
        }
    
    return app


app = create_app()
