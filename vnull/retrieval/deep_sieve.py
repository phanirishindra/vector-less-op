"""DeepSieve - Query Deconstruction with <think> scratchpad.

Layer 1 of the retrieval pipeline. Analyzes user queries,
rewrites vague queries into specific sub-queries.
"""

import re
from dataclasses import dataclass
from collections.abc import AsyncGenerator

from vnull.core.llm_client import LLMClient
from vnull.core.logging import get_logger

logger = get_logger(__name__)

DEEP_SIEVE_SYSTEM = """You are a query analysis system. Your task is to analyze user queries and improve them for retrieval.

Use <think>...</think> tags for your reasoning process. This will be hidden from the user.

Process:
1. In <think> tags, analyze the query:
   - Is it specific or vague?
   - What information is actually being requested?
   - Are there implicit sub-questions?

2. After </think>, output your result as JSON:
   - If query is clear: {"action": "search", "queries": ["original query"]}
   - If query is vague: {"action": "expand", "queries": ["sub-query 1", "sub-query 2", "sub-query 3"]}
   - Maximum 3 sub-queries

Example:
User: "How does authentication work?"
<think>
This query is vague. "Authentication" could mean:
- API authentication methods
- User login flows
- Token management
I should expand into specific sub-queries.
</think>
{"action": "expand", "queries": ["What API authentication methods are available?", "How does user login work?", "How are authentication tokens managed?"]}"""


@dataclass
class SieveResult:
    """Result of query analysis."""
    original_query: str
    action: str
    queries: list[str]
    reasoning: str
    
    @property
    def is_expanded(self) -> bool:
        return self.action == "expand" and len(self.queries) > 1


class DeepSieve:
    """Query deconstruction and analysis.
    
    Uses LLM with <think> scratchpad to analyze queries
    and expand vague ones into specific sub-queries.
    
    Example:
        >>> sieve = DeepSieve()
        >>> result = await sieve.analyze("How does it work?")
        >>> print(result.queries)
        ['How does authentication work?', 'How does the API work?']
    """
    
    THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client or LLMClient()
    
    async def analyze(self, query: str) -> SieveResult:
        """Analyze and potentially expand a query."""
        prompt = f"Analyze this query: {query}"
        
        response = await self.llm.complete(
            prompt=prompt,
            system_prompt=DEEP_SIEVE_SYSTEM,
            temperature=0.3,
        )
        
        content = response.content
        
        think_match = self.THINK_PATTERN.search(content)
        reasoning = think_match.group(1).strip() if think_match else ""
        
        content_no_think = self.THINK_PATTERN.sub("", content).strip()
        
        try:
            import json
            json_match = re.search(r"\{.*\}", content_no_think, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                action = data.get("action", "search")
                queries = data.get("queries", [query])
            else:
                action = "search"
                queries = [query]
        except (json.JSONDecodeError, KeyError):
            action = "search"
            queries = [query]
        
        if not queries:
            queries = [query]
        
        logger.debug("Query analyzed", original=query, action=action, expanded_count=len(queries))
        
        return SieveResult(
            original_query=query,
            action=action,
            queries=queries,
            reasoning=reasoning,
        )
    
    async def stream_analyze(self, query: str) -> AsyncGenerator[str, None]:
        """Stream analysis with <think> tags hidden."""
        prompt = f"Analyze this query: {query}"
        
        async for token in self.llm.stream(
            prompt=prompt,
            system_prompt=DEEP_SIEVE_SYSTEM,
            temperature=0.3,
            hide_think_tags=True,
        ):
            yield token
