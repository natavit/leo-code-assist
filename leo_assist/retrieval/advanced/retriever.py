"""Advanced retriever implementation with chunking and reranking support."""
from typing import List, Dict, Any, Optional
from leo_assist.core.vector_store import VectorStore
from leo_assist.retrieval.ranking.base import RerankerFactory
from leo_assist.utils.settings import RetrievalConfig
from leo_assist.core.embedding.types.task_type import TaskType
from leo_assist.core.vector_store import DocumentResult
from leo_assist.utils.logger import logger

class AdvancedRetriever:
    """Advanced retriever that handles document retrieval with chunking and reranking."""
    
    def __init__(
        self, 
        vector_store: VectorStore,
        config: RetrievalConfig,
    ):
        """Initialize the retriever.
        
        Args:
            vector_store: Vector store instance that implements similarity_search
            config: Retrieval configuration
        """
        self.vector_store = vector_store
        
        # Load configuration
        self.config = config
        
        # Initialize reranker
        self.reranker = RerankerFactory.create_reranker(self.config.reranker)
        
    def retrieve(
        self,
        query: str,
        top_k: int,
        score_threshold: Optional[float] = None,
        task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT,
    ) -> List[DocumentResult]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: The search query
            top_k: Maximum number of results to return
            score_threshold: Optional minimum score threshold for results
            
        Returns:
            List of relevant documents with scores
        """
        # First-pass retrieval
        results = self.vector_store.similarity_search(
            query=query,
            top_k=top_k * 2,  # Get more results for reranking
            task_type=task_type
        )

        logger.info(f"Retrieved {len(results)} documents")
        
        # Rerank the results
        if self.reranker is not None:
            docs = self.reranker.rerank(query, list(results))
            logger.info(f"Reranked {len(docs)} documents")
        else:
            docs = list(results)
        
        # Apply score threshold if provided
        if score_threshold is not None:
            docs = [doc for doc in docs if doc.score >= score_threshold]
        
        # Return top-k results
        return docs[:top_k]
