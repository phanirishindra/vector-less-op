"""Zero-Null Vectorless RAG System.

A high-performance, memory-efficient RAG system that operates without
traditional vector embeddings. Designed for low-RAM environments (4-8GB)
using local LLMs via llama.cpp.
"""

__version__ = "0.1.0"
__author__ = "OrganicSol Group"

from vnull.core.config import settings

__all__ = ["settings", "__version__"]
