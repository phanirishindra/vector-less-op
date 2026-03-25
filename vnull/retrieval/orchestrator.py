"""Multi-Layer Retrieval Orchestrator.

Orchestrates the complete Zero-Null retrieval pipeline:
- Layer 1: DeepSieve (Query Deconstruction)
- Layer 2: ToC Routing (Signpost Matching)
- Layer 3: Iterative Exploration (Multi-path + MCTS-lite)
- Layer 4: BM25 Fallback (Lexical Search)
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Literal

from vnull.core.llm_client import LLMClient
from vnull.core.logging import get_logger
from vnull.indexer.toc_builder import TableOfContents, ToCBuilder
from vnull.retrieval.deep_sieve import DeepSieve, SieveResult
from vnull.retrieval.toc_router import ToCRouter, RouteResult
from vnull.retrieval.explorer import IterativeExplorer, ExplorationResult
from vnull.retrieval.bm25_fallback import BM25Fallback, BM25Result

logger = get_logger(__name__)

FALLBACK_SYSTEM = """You are a precise answer system. Answer the query using ONLY the provided context.

Rules:
1. Use ONLY information from the context
2. If the context doesn't contain the answer, say so
3. Be concise and direct
4. Do not make up information"""


@dataclass
class RetrievalResult:
    """Complete retrieval result."""
    query: str
    answer: str
    layer_used: Literal["sieve", "router", "explorer", "bm25_fallback"]
    sieve_result: SieveResult | None = None
    route_result: RouteResult | None = None
    exploration_result: ExplorationResult | None = None
    bm25_result: BM25Result | None = None
    
    @property
    def success(self) -> bool:
        return bool(self.answer and "could not find" not in self.answer.lower())


class RetrievalOrchestrator:
    """Orchestrate multi-layer retrieval pipeline.
    
    Cascading fallback ensures an answer:
    1. DeepSieve analyzes/expands query
    2. ToC Router finds relevant chunks
    3. Explorer extracts and synthesizes
    4. BM25 Fallback if routing fails
    
    Example:
        >>> orch = RetrievalOrchestrator(toc)
        >>> result = await orch.retrieve("How does auth work?")
        >>> print(result.answer)
    """
    
    def __init__(
        self,
        toc: TableOfContents | None = None,
        llm_client: LLMClient | None = None,
        toc_path: str | None = None,
    ) -> None:
        self.llm = llm_client or LLMClient()
        
        if toc:
            self.toc = toc
        elif toc_path:
            from pathlib import Path
            self.toc = ToCBuilder().load(Path(toc_path))
        else:
            raise ValueError("Must provide toc or toc_path")
        
        self.sieve = DeepSieve(self.llm)
        self.router = ToCRouter(self.toc, self.llm)
        self.explorer = IterativeExplorer(self.toc, self.llm)
        self.bm25 = BM25Fallback(self.toc)
    
    async def _fallback_answer(self, query: str, entry) -> str:
        """Generate answer from BM25 result."""
        prompt = f"""Query: {query}

Context:
{entry.raw_markdown[:3000]}

Answer the query using only the context above:"""
        
        response = await self.llm.complete(
            prompt=prompt,
            system_prompt=FALLBACK_SYSTEM,
            temperature=0.1,
        )
        return response.content.strip()
    
    async def retrieve(self, query: str) -> RetrievalResult:
        """Execute full retrieval pipeline."""
        logger.info(f"Starting retrieval: {query[:50]}...")
        
        sieve_result = await self.sieve.analyze(query)
        queries = sieve_result.queries
        
        all_chunk_ids = []
        route_result = None
        
        for q in queries:
            result = await self.router.route(q)
            all_chunk_ids.extend(result.chunk_ids)
            if route_result is None:
                route_result = result
        
        all_chunk_ids = list(dict.fromkeys(all_chunk_ids))
        
        if all_chunk_ids:
            exploration = await self.explorer.explore(query, all_chunk_ids)
            
            return RetrievalResult(
                query=query,
                answer=exploration.synthesis,
                layer_used="explorer",
                sieve_result=sieve_result,
                route_result=route_result,
                exploration_result=exploration,
            )
        
        logger.info("Router returned no results, falling back to BM25")
        bm25_result = self.bm25.search(query)
        
        if bm25_result.top_entry:
            answer = await self._fallback_answer(query, bm25_result.top_entry)
            
            return RetrievalResult(
                query=query,
                answer=answer,
                layer_used="bm25_fallback",
                sieve_result=sieve_result,
                bm25_result=bm25_result,
            )
        
        return RetrievalResult(
            query=query,
            answer="I could not find relevant information to answer this query.",
            layer_used="bm25_fallback",
            sieve_result=sieve_result,
        )
    
    async def stream_retrieve(self, query: str) -> AsyncGenerator[str, None]:
        """Stream retrieval with hidden <think> tags."""
        result = await self.retrieve(query)
        
        for char in result.answer:
            yield char
