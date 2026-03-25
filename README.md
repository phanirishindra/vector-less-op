# Zero-Null Vectorless RAG System

A high-performance, memory-efficient RAG system that operates without traditional vector embeddings. Designed for low-RAM environments (4-8GB) using local LLMs via llama.cpp.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ZERO-NULL VECTORLESS RAG                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   CRAWLER    │───▶│    PARSER    │───▶│   INDEXER    │                   │
│  │              │    │              │    │              │                   │
│  │ • Async HTTP │    │ • DOM Prune  │    │ • Signposts  │                   │
│  │ • Playwright │    │ • HTML Split │    │ • Bookends   │                   │
│  │ • Bloom Filt │    │ • LLM→MD     │    │ • ToC JSON   │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                 │                            │
│                                                 ▼                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      RETRIEVAL ORCHESTRATOR                          │   │
│  │                                                                      │   │
│  │  Layer 1: DeepSieve (Query Deconstruction + <think> scratchpad)     │   │
│  │  Layer 2: ToC Routing (Signpost matching via LLM)                   │   │
│  │  Layer 3: Iterative Exploration (Multi-path + MCTS-lite)            │   │
│  │  Layer 4: BM25 Fallback (Lexical search on bookends)                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                 │                            │
│                                                 ▼                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                           API SERVER                                 │   │
│  │                    FastAPI + Streaming Response                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Zero Vector Embeddings**: Uses Dense Conceptual Signposts instead of embeddings
- **Memory Efficient**: Designed for 4-8GB RAM environments
- **Local LLM**: Qwen 2.5B via llama.cpp (OpenAI-compatible API)
- **Aggressive DOM Pruning**: Strips boilerplate before LLM processing
- **Multi-Layer Retrieval**: Cascading fallback guarantees answers
- **Streaming Output**: Hides `<think>` reasoning from users

## Requirements

- Python 3.11+
- llama.cpp server running on port 8000
- Qwen 2.5B model loaded
- 4-8GB RAM minimum

## Installation

```bash
# Clone repository
git clone https://gitlab.com/organicsol-group/v-less.git
cd v-less

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .

# Copy environment template
cp .env.example .env
```

## Quick Start

```bash
# 1. Start llama.cpp server (in separate terminal)
./llama-server -m qwen2.5-3b-instruct.gguf --port 8000

# 2. Crawl a website
python -m vnull.cli crawl https://example.com --depth 2

# 3. Start the API server
python -m vnull.cli serve

# 4. Query the system
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?"}'
```

## Project Structure

```
v-less/
├── vnull/                    # Main package
│   ├── core/                 # Core utilities and config
│   ├── crawler/              # Async web crawling
│   ├── parser/               # DOM pruning & HTML→Markdown
│   ├── indexer/              # Signpost generation & ToC
│   ├── retrieval/            # Multi-layer retrieval orchestrator
│   └── api/                  # FastAPI server
├── data/                     # Runtime data storage
│   ├── raw/                  # Raw crawled HTML
│   ├── markdown/             # Converted Markdown
│   └── index/                # ToC JSON files
├── tests/                    # Test suite
└── scripts/                  # Utility scripts
```

## Configuration

Edit `.env` or set environment variables:

```bash
LLM_BASE_URL=http://127.0.0.1:8000/v1
LLM_API_KEY=sk-local
LLM_MODEL=qwen2.5-3b-instruct
MAX_TOKENS_PER_CHUNK=6000
DATA_DIR=./data
```

# Simple examples (no LLM required)
python -m examples.example_usage bloom/n
python -m examples.example_usage prune/n
python -m examples.example_usage chunk/n
python -m examples.example_usage bm25

# Full pipeline (requires llama.cpp on port 8000)
python -m examples.example_usage full



## License

MIT License
