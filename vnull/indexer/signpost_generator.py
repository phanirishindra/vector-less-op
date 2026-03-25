"""Dense Conceptual Signpost generation using LLM.

Generates compact signposts for each chunk:
[Core Theme] + [Key Entities] + [Questions Answered]

NO VECTOR EMBEDDINGS - this is the vectorless approach.
"""

import asyncio
import re
from dataclasses import dataclass

from vnull.core.config import settings
from vnull.core.llm_client import LLMClient
from vnull.core.logging import get_logger
from vnull.indexer.chunker import MarkdownChunk

logger = get_logger(__name__)


SIGNPOST_SYSTEM_PROMPT = """You are a precise indexing system. Generate a DENSE SIGNPOST for the given text chunk.

Format your response EXACTLY as:
[Core Theme] + [Key Entities] + [Questions Answered]

Rules:
1. Core Theme: 2-5 words capturing the main topic
2. Key Entities: Important names, terms, concepts (comma-separated)
3. Questions Answered: What questions does this text answer? (comma-separated)
4. MAXIMUM 30 tokens total
5. Be specific and precise - this is for retrieval matching
6. Output ONLY the signpost line, nothing else

Example outputs:
- [API Authentication] + [OAuth2, JWT, tokens] + [How to authenticate?, What are token types?]
- [Database Indexing] + [B-tree, PostgreSQL, query optimization] + [How to speed up queries?, When to use indexes?]
- [Error Handling] + [try-catch, exceptions, logging] + [How to handle errors?, Best practices for exceptions?]"""


@dataclass
class Signpost:
    """Dense conceptual signpost for a chunk."""
    chunk_id: str
    signpost: str
    core_theme: str
    key_entities: list[str]
    questions_answered: list[str]
    token_count: int
    
    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "signpost": self.signpost,
            "core_theme": self.core_theme,
            "key_entities": self.key_entities,
            "questions_answered": self.questions_answered,
            "token_count": self.token_count,
        }


class SignpostGenerator:
    """Generate dense conceptual signposts for chunks.
    
    This is the core of the vectorless approach - instead of
    embedding vectors, we generate semantic signposts that
    can be matched by the LLM during retrieval.
    
    Example:
        >>> generator = SignpostGenerator()
        >>> signpost = await generator.generate(chunk)
        >>> print(signpost.signpost)
        [API Auth] + [OAuth2, JWT] + [How to authenticate?]
    """
    
    SIGNPOST_PATTERN = re.compile(
        r"\[([^\]]+)\]\s*\+\s*\[([^\]]+)\]\s*\+\s*\[([^\]]+)\]"
    )
    
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize generator.
        
        Args:
            llm_client: LLM client instance.
            max_tokens: Maximum tokens for signpost.
        """
        self.llm = llm_client or LLMClient()
        self.max_tokens = max_tokens or settings.max_signpost_tokens
    
    def _parse_signpost(self, raw: str) -> tuple[str, list[str], list[str]]:
        """Parse signpost into components."""
        match = self.SIGNPOST_PATTERN.search(raw)
        if match:
            core_theme = match.group(1).strip()
            entities = [e.strip() for e in match.group(2).split(",")]
            questions = [q.strip() for q in match.group(3).split(",")]
            return core_theme, entities, questions
        
        return raw.strip(), [], []
    
    def _count_tokens(self, text: str) -> int:
        """Approximate token count."""
        return len(text.split()) + len(text) // 4
    
    async def generate(self, chunk: MarkdownChunk) -> Signpost:
        """Generate signpost for a single chunk.
        
        Args:
            chunk: MarkdownChunk to process.
            
        Returns:
            Signpost object.
        """
        preview = chunk.content[:2000] if len(chunk.content) > 2000 else chunk.content
        
        prompt = f"""Generate a dense signpost for this text:

Header: {chunk.header}

Content:
{preview}

First sentence: {chunk.first_sentence}
Last sentence: {chunk.last_sentence}"""
        
        response = await self.llm.complete(
            prompt=prompt,
            system_prompt=SIGNPOST_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=50,
        )
        
        raw_signpost = response.content.strip()
        core_theme, entities, questions = self._parse_signpost(raw_signpost)
        
        if not self.SIGNPOST_PATTERN.search(raw_signpost):
            raw_signpost = f"[{core_theme}] + [{', '.join(entities) or chunk.header}] + [What is {chunk.header}?]"
        
        token_count = self._count_tokens(raw_signpost)
        
        logger.debug(
            "Signpost generated",
            chunk_id=chunk.chunk_id,
            signpost=raw_signpost[:100],
            tokens=token_count,
        )
        
        return Signpost(
            chunk_id=chunk.chunk_id,
            signpost=raw_signpost,
            core_theme=core_theme,
            key_entities=entities,
            questions_answered=questions,
            token_count=token_count,
        )
    
    async def generate_batch(
        self,
        chunks: list[MarkdownChunk],
        flush_cache: bool = True,
    ) -> list[Signpost]:
        """Generate signposts for multiple chunks.
        
        Args:
            chunks: List of chunks to process.
            flush_cache: Flush KV cache between chunks.
            
        Returns:
            List of Signpost objects.
        """
        signposts = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Generating signpost {i+1}/{len(chunks)}")
            
            signpost = await self.generate(chunk)
            signposts.append(signpost)
            
            if flush_cache and i < len(chunks) - 1:
                await self.llm.flush_kv_cache()
        
        logger.info(
            "Batch signpost generation complete",
            total=len(signposts),
            avg_tokens=sum(s.token_count for s in signposts) // len(signposts) if signposts else 0,
        )
        
        return signposts
