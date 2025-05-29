# """Retrieval module for the Leo RAG Assistant.

# This module provides document retrieval functionality with support for:
# - Basic document retrieval
# - Advanced retrieval with chunking and reranking
# - Configuration management for retrieval components
# """
# from typing import List, Dict, Any, Optional, Union
# from pathlib import Path

# # Document retriever for the RAG system
# from .retriever import DocumentRetriever

# # Advanced retrieval components
# from .advanced import (
#     AdvancedRetriever,
#     RetrievalConfig,
#     create_default_config,
#     save_default_config
# )

# # Re-export components for backward compatibility
# __all__ = [
#     'DocumentRetriever',
#     'AdvancedRetriever',
#     'RetrievalConfig',
#     'create_default_config',
#     'save_default_config'
# ]