# Zero-Null Vectorless RAG - Complete Execution Trace

This document provides a comprehensive walkthrough of the entire pipeline execution, from crawling to querying.

## Project Structure (Verified)

```
v-less/
├── vnull/
│   ├── __init__.py
│   ├── cli.py                      # CLI commands (crawl, convert, index, query, serve, pipeline)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py               # Pydantic settings (LLM URL, tokens, paths)
│   │   ├── logging.py              # Structlog configuration
│   │   └── llm_client.py           # OpenAI SDK wrapper for llama.cpp
│   ├── crawler/
│   │   ├── __init__.py
│   │   ├── bloom_filter.py         # URL deduplication (mmh3 + bitarray)
│   │   ├── async_crawler.py        # aiohttp concurrent crawler
│   │   └── js_renderer.py          # Playwright stealth renderer
│   ├── parser/
│   │   ├── __init__.py
│   │   ├── dom_pruner.py           # BeautifulSoup boilerplate removal
│   │   ├── html_splitter.py        # Token-aware splitting (Qwen tokenizer)
│   │   └── markdown_converter.py   # LLM-driven HTML→Markdown
│   ├── indexer/
│   │   ├── __init__.py
│   │   ├── chunker.py              # Header-boundary chunking + bookends
│   │   ├── signpost_generator.py   # Dense signpost generation via LLM
│   │   └── toc_builder.py          # JSON ToC builder
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── deep_sieve.py           # Layer 1: Query deconstruction
│   │   ├── toc_router.py           # Layer 2: Signpost matching
│   │   ├── explorer.py             # Layer 3: Multi-path + MCTS-lite
│   │   ├── bm25_fallback.py        # Layer 4: Lexical fallback
│   │   └── orchestrator.py         # Multi-layer orchestrator
│   └── api/
│       ├── __init__.py
│       ├── server.py               # FastAPI + streaming
│       └── schemas.py              # Pydantic models
├── tests/
│   ├── conftest.py
│   ├── test_bloom_filter.py
│   ├── test_dom_pruner.py
│   ├── test_chunker.py
│   ├── test_toc_builder.py
│   └── test_bm25_fallback.py
├── scripts/setup.sh
├── data/                           # Runtime data
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Example Execution Trace

**Command:** `vnull pipeline https://docs.example.com --name my-docs --depth 2`

---

### STEP 1: CRAWLING (`vnull/crawler/`)

```
Input:  https://docs.example.com
Output: data/raw/*.html
```

**Execution Flow:**

```python
# 1. AsyncCrawler initializes
crawler = AsyncCrawler()
├── BloomFilter(expected_items=1_000_000, fp_rate=0.01)
│   └── Allocates ~1.2MB bitarray for URL deduplication
├── Semaphore(5)  # max_concurrent from settings
└── aiohttp.ClientSession with stealth headers

# 2. Crawl loop starts
seed_url = "https://docs.example.com"
bloom.add(seed_url)
queue.put(CrawlTask(url=seed_url, depth=0))

# 3. For each URL in queue:
while not queue.empty():
    batch = [queue.get() for _ in range(5)]  # Get up to 5 tasks
    
    results = await asyncio.gather(*[fetch_url(task) for task in batch])
    
    for result in results:
        # Extract links using regex: href=["']([^"'>]+)["']
        links = extract_links(result.content, result.url)
        
        for link in links:
            if link not in bloom and same_domain(link, seed_url):
                bloom.add(link)
                queue.put(CrawlTask(url=link, depth=result.depth+1))
        
        # Save with metadata header
        # <!-- URL: https://docs.example.com/page1 -->
        # <!-- Crawled: 2024-01-15T10:30:00Z -->
        save_result(result)  # → data/raw/a1b2c3d4e5f6.html
```

**Sample Output:**
```
data/raw/
├── 8f3a2b1c9d4e.html  (homepage)
├── 2c4d6e8f0a1b.html  (docs/getting-started)
├── 5e7f9a1b3c5d.html  (docs/api-reference)
└── ... (50 pages max)
```

---

### STEP 2: CONVERSION (`vnull/parser/`)

```
Input:  data/raw/*.html
Output: data/markdown/*.md
```

**Execution Flow:**

```python
# 1. For each HTML file:
for html_file in raw_dir.glob("*.html"):
    html = html_file.read_text()
    
    # 2. DOM Pruning (DOMPruner)
    pruner = DOMPruner(remove_nav=True, remove_footer=True)
    prune_result = pruner.prune(html)
    
    # Removes: <script>, <style>, <nav>, <footer>, <svg>, <aside>
    # Removes: elements with class/id matching: ad, banner, cookie, popup
    # Result: 60-80% size reduction typically
    
    pruned_html = prune_result.pruned_html
    # Original: 45,000 chars → Pruned: 12,000 chars
    
    # 3. Token Check (HTMLSplitter)
    splitter = HTMLSplitter(max_tokens=6000)
    token_count = count_tokens(pruned_html)  # Using Qwen tokenizer
    
    if token_count > 6000:
        # Split at structural boundaries: <section>, <article>, <h1>-<h6>
        chunks = splitter.split(pruned_html)
        # Returns: [HTMLChunk(content=..., token_count=3200), 
        #           HTMLChunk(content=..., token_count=2800)]
    else:
        chunks = [HTMLChunk(content=pruned_html, token_count=token_count)]
    
    # 4. LLM Conversion (MarkdownConverter)
    for chunk in chunks:
        response = await llm.complete(
            prompt=f"Convert this HTML to Markdown:\n\n{chunk.content}",
            system_prompt=SYSTEM_PROMPT,  # Rules for tables, code, links
            temperature=0.1
        )
        markdown_parts.append(response.content)
        
        # Flush KV cache between chunks to prevent OOM
        await llm.flush_kv_cache()
    
    markdown = "\n\n".join(markdown_parts)
    
    # 5. Save with source metadata
    # <!-- Source: https://docs.example.com/api-reference -->
    output_path.write_text(markdown)
```

**LLM Prompt Example:**
```
System: You are a precise HTML-to-Markdown converter...

User: Convert this HTML to Markdown:

<body>
<main>
<h1>API Reference</h1>
<p>This section covers the REST API endpoints.</p>
<h2>Authentication</h2>
<p>All requests require an <code>Authorization</code> header.</p>
<table>
<tr><th>Method</th><th>Endpoint</th></tr>
<tr><td>POST</td><td>/auth/token</td></tr>
</table>
</main>
</body>
```

**LLM Response:**
```markdown
# API Reference

This section covers the REST API endpoints.

## Authentication

All requests require an `Authorization` header.

| Method | Endpoint |
|--------|----------|
| POST | /auth/token |
```

---

### STEP 3: INDEXING (`vnull/indexer/`)

```
Input:  data/markdown/*.md
Output: data/index/my-docs.json
```

**Execution Flow:**

```python
# 1. Chunk Markdown at headers (MarkdownChunker)
chunker = MarkdownChunker(min_chunk_size=100)

for md_file in markdown_dir.glob("*.md"):
    chunks = chunker.chunk(md_file.read_text())
    
    # For each header boundary:
    # - Extract chunk_id (SHA256 hash of header+content)
    # - Extract first_sentence (bookend)
    # - Extract last_sentence (bookend)
    # - Track parent_id for hierarchy

all_chunks = [
    MarkdownChunk(
        chunk_id="a1b2c3d4e5f6",
        header="Authentication",
        header_level=2,
        content="## Authentication\n\nAll requests require...",
        first_sentence="All requests require an Authorization header.",
        last_sentence="Tokens expire after 1 hour.",
        parent_id="root_api_ref",
    ),
    # ... more chunks
]

# 2. Generate Dense Signposts (SignpostGenerator)
generator = SignpostGenerator()

for chunk in all_chunks:
    response = await llm.complete(
        prompt=f"""Generate a dense signpost for this text:
        
Header: {chunk.header}
Content: {chunk.content[:2000]}
First sentence: {chunk.first_sentence}
Last sentence: {chunk.last_sentence}""",
        system_prompt=SIGNPOST_SYSTEM,  # Format: [Theme] + [Entities] + [Questions]
        max_tokens=50
    )
    
    signpost = parse_signpost(response.content)
    # "[API Auth] + [OAuth2, JWT, tokens] + [How to authenticate?, Token expiry?]"
    
    await llm.flush_kv_cache()  # Prevent OOM

# 3. Build ToC JSON (ToCBuilder)
toc = TableOfContents(
    entries=[
        ToCEntry(
            chunk_id="a1b2c3d4e5f6",
            dense_signpost="[API Auth] + [OAuth2, JWT] + [How to authenticate?]",
            first_sentence="All requests require an Authorization header.",
            last_sentence="Tokens expire after 1 hour.",
            raw_markdown="## Authentication\n\nAll requests require...",
            header="Authentication",
            header_level=2,
            parent_id="root_api_ref",
            core_theme="API Auth",
            key_entities=["OAuth2", "JWT", "tokens"],
            questions_answered=["How to authenticate?", "Token expiry?"],
        ),
        # ... more entries
    ],
    source_name="my-docs",
    created_at=datetime.now(UTC),
)

# 4. Save to JSON
builder.save(toc, Path("data/index/my-docs.json"))
```

**Output JSON Structure:**
```json
{
  "version": "1.0",
  "source_name": "my-docs",
  "created_at": "2024-01-15T10:45:00Z",
  "entry_count": 47,
  "entries": [
    {
      "chunk_id": "a1b2c3d4e5f6",
      "dense_signpost": "[API Auth] + [OAuth2, JWT] + [How to authenticate?]",
      "first_sentence": "All requests require an Authorization header.",
      "last_sentence": "Tokens expire after 1 hour.",
      "header": "Authentication",
      "header_level": 2,
      "parent_id": "root_api_ref",
      "children_ids": ["child1", "child2"],
      "core_theme": "API Auth",
      "key_entities": ["OAuth2", "JWT", "tokens"],
      "questions_answered": ["How to authenticate?", "Token expiry?"],
      "raw_markdown": "## Authentication\n\nAll requests require..."
    }
  ]
}
```

---

### STEP 4: QUERY (`vnull/retrieval/`)

```
Input:  "How do I authenticate API requests?"
Output: Synthesized answer from relevant chunks
```

**Execution Flow (4-Layer Cascade):**

```python
orchestrator = RetrievalOrchestrator(toc=toc)
result = await orchestrator.retrieve("How do I authenticate API requests?")

# ═══════════════════════════════════════════════════════════════
# LAYER 1: DeepSieve (Query Deconstruction)
# ═══════════════════════════════════════════════════════════════

sieve = DeepSieve(llm)
sieve_result = await sieve.analyze(query)

# LLM uses <think> scratchpad (hidden from user):
"""
<think>
This query is about API authentication. It's fairly specific but could mean:
- OAuth2 flow
- API key usage  
- JWT tokens
I'll keep it as-is since it's clear enough.
</think>
{"action": "search", "queries": ["How do I authenticate API requests?"]}
"""

# Result: queries = ["How do I authenticate API requests?"]

# ═══════════════════════════════════════════════════════════════
# LAYER 2: ToC Router (Signpost Matching)
# ═══════════════════════════════════════════════════════════════

router = ToCRouter(toc, llm)

# Format signposts for LLM:
signposts_text = """
- a1b2c3d4e5f6: [API Auth] + [OAuth2, JWT] + [How to authenticate?] (Header: Authentication)
- b2c3d4e5f6a1: [Rate Limits] + [throttling, 429] + [What are rate limits?] (Header: Rate Limiting)
- c3d4e5f6a1b2: [Endpoints] + [REST, GET, POST] + [What endpoints exist?] (Header: API Endpoints)
...
"""

response = await llm.complete_json(
    prompt=f"Query: {query}\n\nTable of Contents:\n{signposts_text}\n\nReturn relevant chunk_ids:",
    system_prompt=ROUTER_SYSTEM
)

# LLM returns: ["a1b2c3d4e5f6", "d4e5f6a1b2c3"]
chunk_ids = ["a1b2c3d4e5f6", "d4e5f6a1b2c3"]

# ═══════════════════════════════════════════════════════════════
# LAYER 3: Iterative Explorer (Multi-path + MCTS-lite)
# ═══════════════════════════════════════════════════════════════

explorer = IterativeExplorer(toc, llm)

extractions = []
for chunk_id in chunk_ids:
    entry = toc.get_entry(chunk_id)
    
    response = await llm.complete(
        prompt=f"""Query: {query}
        
Chunk ID: {entry.chunk_id}
Header: {entry.header}
Parent ID: {entry.parent_id}

Content:
{entry.raw_markdown[:3000]}

Extract relevant facts:""",
        system_prompt=EXTRACT_SYSTEM
    )
    
    # If LLM needs more context, it can request parent:
    # {"action": "explore_parent", "target": "root_api_ref"}
    
    # Otherwise, returns extracted facts:
    facts = response.content
    extractions.append(ExtractionResult(chunk_id=chunk_id, facts=facts))
    
    await llm.flush_kv_cache()  # Prevent OOM between chunks

# Synthesis call:
facts_text = "\n\n".join([f"From {e.chunk_id}:\n{e.facts}" for e in extractions])

synthesis = await llm.complete(
    prompt=f"""Query: {query}

Extracted Facts:
{facts_text}

Synthesize a complete answer:""",
    system_prompt=SYNTHESIS_SYSTEM
)

answer = synthesis.content

# ═══════════════════════════════════════════════════════════════
# LAYER 4: BM25 Fallback (Only if Layer 2 returns [])
# ═══════════════════════════════════════════════════════════════

# If router returned no chunk_ids:
if not chunk_ids:
    bm25 = BM25Fallback(toc)
    bm25_result = bm25.search(query, top_k=5)
    
    # BM25 searches against:
    # - first_sentence (bookend)
    # - last_sentence (bookend)
    # - dense_signpost
    # - raw_markdown[:1000]
    
    if bm25_result.top_entry:
        answer = await llm.complete(
            prompt=f"Query: {query}\n\nContext:\n{bm25_result.top_entry.raw_markdown}\n\nAnswer:",
            system_prompt=FALLBACK_SYSTEM
        )
```

**Final Result:**
```python
RetrievalResult(
    query="How do I authenticate API requests?",
    answer="""To authenticate API requests, you need to include an Authorization 
header with a valid JWT token. Here's the process:

1. Obtain a token by calling POST /auth/token with your credentials
2. Include the token in subsequent requests: `Authorization: Bearer <token>`
3. Tokens expire after 1 hour and must be refreshed

For OAuth2 flows, you can also use the /oauth/authorize endpoint.""",
    layer_used="explorer",
    chunks_explored=2,
    success=True
)
```

---

## Memory Efficiency Features

| Feature | Implementation | Memory Saved |
|---------|---------------|--------------|
| Bloom Filter | 1.2MB for 1M URLs | vs. 50MB+ HashSet |
| KV Cache Flush | Between each LLM call | Prevents OOM |
| Token-aware Splitting | Max 6000 tokens/chunk | Fits in 4GB RAM |
| No Vector Embeddings | Dense signposts instead | Saves 500MB+ |
| Streaming Response | AsyncGenerator | Constant memory |

---

## API Endpoints

```bash
# Health check
GET /health
→ {"status": "healthy", "llm_connected": true, "index_loaded": true, "index_entries": 47}

# Query (blocking)
POST /query
{"query": "How do I authenticate?"}
→ {"answer": "...", "layer_used": "explorer", "chunks_used": 2, "success": true}

# Query (streaming)
POST /query/stream
{"query": "How do I authenticate?"}
→ Streams tokens as text/plain

# Index stats
GET /index/stats
→ {"source_name": "my-docs", "entry_count": 47, "created_at": "..."}
```

---

## Quick Start Commands

```bash
# Setup
git clone https://gitlab.com/organicsol-group/v-less.git
cd v-less
chmod +x scripts/setup.sh && ./scripts/setup.sh

# Start LLM server (separate terminal)
./llama-server -m qwen2.5-3b-instruct.gguf --port 8000

# Run full pipeline
source venv/bin/activate
vnull pipeline https://docs.example.com --name my-docs --depth 2

# Query
vnull query "How does authentication work?"

# Start API server
vnull serve --port 8080
```

---

This is the complete execution flow of the Zero-Null Vectorless RAG system. The key innovation is replacing vector embeddings with LLM-generated dense signposts, enabling semantic retrieval on systems with only 4-8GB RAM.
