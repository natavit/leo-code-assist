"""Chunking utilities for document processing.

This module provides various chunking strategies for splitting documents into
manageable pieces for vector storage and retrieval.
"""
from .base import Chunk, Chunker, LangChainChunker, ChunkerFactory
from .leo_chunker import TreeSitterLeoChunker, LeoRegexChunker
from .config import DEFAULT_CONFIGS

# Register built-in chunkers
ChunkerFactory.register_chunker("langchain", LangChainChunker)

try:
    from .leo_chunker import TreeSitterLeoChunker, LeoRegexChunker
    # Register chunkers
    ChunkerFactory.register_chunker("tree_sitter_leo", TreeSitterLeoChunker)
    ChunkerFactory.register_chunker("leo_regex", LeoRegexChunker)
except ImportError:
    pass  # Tree-sitter dependencies not installed

__all__ = [
    "Chunk",
    "Chunker",
    "LangChainChunker",
    "TreeSitterLeoChunker",
    "LeoRegexChunker",
    "ChunkerFactory",
    "DEFAULT_CONFIGS"
]
