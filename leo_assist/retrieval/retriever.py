"""Document retriever with advanced chunking, reranking, and search capabilities.

This module provides the DocumentRetriever class which handles document retrieval
with support for:
- Hybrid search (semantic + lexical)
- Advanced reranking with Gemini
- Configurable chunking strategies
- Filtering and pagination
"""
from typing import List, Dict, Any, Optional
from pathlib import Path

from leo_assist.core.vector_store import DocumentResult
from leo_assist.core.vector_store import VectorStore
from leo_assist.core.embedding.types.task_type import TaskType

from leo_assist.retrieval.advanced.retriever import AdvancedRetriever
from leo_assist.utils.settings import Settings
from leo_assist.utils.settings import RetrievalConfig
from leo_assist.utils.logger import logger

class DocumentRetriever:
    """Unified document retriever with advanced search and filtering capabilities.
    
    This class provides a comprehensive interface for document retrieval with support
    for hybrid search, reranking, and filtering. It's designed to work with the
    DocumentIndexingPipeline for a complete document management solution.
    """
    
    # Keywords that indicate a code-related query
    CODE_QUERY_KEYWORDS = {
        'code', 'sample', 'example', 'write', 'create', 'implement', 'function',
        'class', 'struct', 'app', 'application', 'script', 'program', 'functionality'
    }
    
    def _is_code_query(self, query: str) -> bool:
        """Determine if a query is likely to be code-related based on keywords.
        
        Args:
            query: The search query string
            
        Returns:
            True if the query appears to be code-related, False otherwise
        """
        # Convert query to lowercase and split into words
        words = set(word.lower() for word in query.split())
        
        # Check if any code-related keywords are present
        return bool(words & self.CODE_QUERY_KEYWORDS)
    
    def __init__(
        self, 
        vector_store: VectorStore,
        top_k: int,
        score_threshold: float,
        max_chars: int,
        retrieval_config: RetrievalConfig,
    ):
        """Initialize the document retriever.
        
        Args:
            vector_store: Vector store instance for similarity search
            top_k: Maximum number of results to return
            score_threshold: Minimum similarity score threshold (0-1)
            max_chars: Maximum number of characters to return in total
            reranker_config: Configuration for the reranker
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.max_chars = max_chars
        
        # Initialize the advanced retriever with configuration
        self._init_advanced_retriever(retrieval_config)
        
    @classmethod
    def from_settings(
        cls, 
        vector_store: VectorStore,
        settings: Settings,
    ) -> 'DocumentRetriever':
        """Create a DocumentRetriever instance from settings.
        
        Args:
            vector_store: Vector store instance for similarity search
            settings: Settings object containing configuration
            
        Returns:
            DocumentRetriever: Configured instance
            
        Example:
            ```python
            from leo_rag_assistant.core.vector_store import DocumentVectorStore
            from leo_rag_assistant.retrieval import DocumentRetriever
            from leo_rag_assistant.utils.config import settings
            
            # Create vector store
            vector_store = DocumentVectorStore.from_settings(settings)
            
            # Create retriever with settings
            retriever = DocumentRetriever.from_settings(
                vector_store=vector_store,
                settings=settings
            )
            
            # With overrides
            retriever = DocumentRetriever.from_settings(
                vector_store=vector_store,
                settings=settings,
                top_k=15,
                score_threshold=0.8
            )
            ```
        """
        return cls(
            vector_store=vector_store,
            top_k=settings.retrieval.top_k,
            score_threshold=settings.retrieval.score_threshold,
            max_chars=settings.retrieval.max_chars,
            retrieval_config=settings.retrieval
        )
    
    def _init_advanced_retriever(self, retrieval_config: RetrievalConfig) -> None:
        """Initialize the advanced retriever with configuration."""

        self.retriever = AdvancedRetriever(
            vector_store=self.vector_store,
            config=retrieval_config
        )
        logger.info("Initialized retriever with configuration from settings")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentResult]:
        """Search for documents similar to the query with advanced filtering.
        
        Args:
            query: The search query string
            top_k: Maximum number of results to return (overrides instance default)
            score_threshold: Minimum similarity score (0-1) for results
            filter: Optional filter dictionary to apply to the search
            
        Returns:
            List of matching DocumentResult objects sorted by relevance
        """
        # Use instance values if not overridden
        top_k = top_k or self.top_k
        score_threshold = score_threshold or self.score_threshold
        
        try:
            is_code_query = self._is_code_query(query)
            
            # Determine task type based on whether this is a code query
            query_task_type = TaskType.CODE_RETRIEVAL_QUERY if is_code_query else TaskType.RETRIEVAL_DOCUMENT
            
            # Apply any additional filters from kwargs
            if filter is not None:
                kwargs["filter"] = filter
            
            # Use the advanced retriever with appropriate task type
            results = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
                task_type=query_task_type,
            )
            
            # Convert results to DocumentChunk objects and apply max_chars limit
            # return self._apply_max_chars(results)
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise
            
    def get_document(self, doc_id: str) -> Optional[DocumentResult]:
        """Retrieve a single document by its ID.
        
        Args:
            doc_id: The unique identifier of the document
            
        Returns:
            DocumentResult if found, None otherwise
        """
        try:
            return self.vector_store.get_document(doc_id)
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {str(e)}")
            return None
    
    # Maintain backward compatibility
    def similarity_search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[DocumentResult]:
        """Alias for retrieve() for backward compatibility."""
        return self.retrieve(query, top_k=top_k)
    
    