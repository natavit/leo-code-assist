"""Base classes and implementations for ranking strategies."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from collections import defaultdict

from leo_assist.utils.settings import RerankerConfig
from leo_assist.utils.settings import RerankerType
from leo_assist.core.vector_store import DocumentResult
from leo_assist.utils.logger import logger

class Reranker(ABC):
    """Abstract base class for rerankers."""
    
    @abstractmethod
    def rerank(
        self, 
        query: str, 
        docs: List[DocumentResult], 
    ) -> List[DocumentResult]:
        """Rerank the given chunks based on the query."""
        pass

class ParentDocumentReranker(Reranker):
    """Reranker that groups chunks by parent document and selects the best chunks."""
    
    def __init__(self, top_k_parents: int = 5, max_chunks_per_parent: int = 3):
        self.top_k_parents = top_k_parents
        self.max_chunks_per_parent = max_chunks_per_parent

    def rerank(
        self, 
        query: str, 
        docs: List[DocumentResult], 
    ) -> List[DocumentResult]:
        """
        Rerank chunks by grouping them by parent document and selecting the best chunks.
        
        Args:
            query: The search query (unused in this implementation but kept for interface compatibility)
            docs: List of document results to rerank
            
        Returns:
            Reranked list of document results
        """
        if not docs:
            return []
            
        # Group chunks by parent document
        parent_to_chunks: Dict[str, List[DocumentResult]] = defaultdict(list)
        
        for doc in docs:
            parent_id = doc.document.parent_id
            parent_to_chunks[parent_id].append(doc)

        logger.info(f"Grouped {len(docs)} chunks into {len(parent_to_chunks)} parents")
        
        # Sort parents by their highest scoring chunk
        sorted_parents = sorted(
            parent_to_chunks.items(),
            key=lambda x: max(doc.score for doc in x[1]),
            reverse=True
        )[:self.top_k_parents]
        
        logger.info(f"Selected {len(sorted_parents)} parents")
        logger.info([parent_id for parent_id, _ in sorted_parents])
        
        # Select top chunks from each parent
        results = []
        seen_chunk_ids = set()
        
        for parent_id, chunks in sorted_parents:
            # Sort chunks within parent by score (descending)
            sorted_chunks = sorted(
                chunks,
                key=lambda x: x.score,
                reverse=True
            )
            
            # Add chunks up to max_chunks_per_parent
            for chunk in sorted_chunks[:self.max_chunks_per_parent]:
                if chunk.document.id not in seen_chunk_ids:
                    results.append(chunk)
                    seen_chunk_ids.add(chunk.document.id)
        
        return results
    
    # def rerank(
    #     self, 
    #     query: str, 
    #     docs: List[DocumentResult], 
    #     **kwargs
    # ) -> List[DocumentResult]:
    #     """Group chunks by parent and select the best ones."""
    #     if not docs:
    #         return []
            
    #     # Group chunks by parent
    #     parent_chunks = defaultdict(list)
    #     parent_scores = defaultdict(float)
        
    #     for doc in docs:
    #         parent_id = doc.document.parent_id
    #         if not parent_id:
    #             parent_id = doc.document.id  # Use chunk ID as parent if no parent_id
                
    #         score = doc.score
    #         parent_chunks[parent_id].append(doc)
    #         parent_scores[parent_id] = max(parent_scores[parent_id], score)
        
    #     # Sort parents by their best chunk score
    #     sorted_parents = sorted(
    #         parent_scores.items(),
    #         key=lambda x: x[1],
    #         reverse=True
    #     )[:self.top_k_parents]
        
    #     # Select best chunks from top parents
    #     results = []
    #     seen_chunks = set()
        
    #     for parent_id, _ in sorted_parents:
    #         # Sort chunks within parent by score
    #         sorted_chunks = sorted(
    #             parent_chunks[parent_id],
    #             key=lambda x: x.score,
    #             reverse=True
    #         )
            
    #         # Add docs up to max_chunks_per_parent
    #         for doc in sorted_chunks[:self.max_chunks_per_parent]:
    #             doc_id = doc.document.id
    #             if doc_id not in seen_chunks:
    #                 results.append(doc)
    #                 seen_chunks.add(doc_id)
        
    #     return results

class HybridReranker(Reranker):
    """Reranker that combines semantic and lexical (BM25) scores."""
    
    def __init__(self, alpha: float = 0.5, bm25_k1: float = 1.2, bm25_b: float = 0.75):
        self.alpha = alpha  # Weight for semantic score (1-alpha for BM25)
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
    
    def rerank(
        self, 
        query: str, 
        docs: List[DocumentResult], 
        **kwargs
    ) -> List[DocumentResult]:
        """Rerank chunks using a combination of semantic and BM25 scores."""
        if not docs:
            return []
            
        # Extract texts and scores
        # texts = [doc.document.content for doc in docs]
        # semantic_scores = np.array([doc.score for doc in docs])
        
        # # Calculate BM25 scores
        # tokenized_corpus = [self._tokenize(text) for text in texts]
        # tokenized_query = self._tokenize(query)
        
        # bm25 = BM25Okapi(
        #     tokenized_corpus,
        #     k1=self.bm25_k1,
        #     b=self.bm25_b
        # )
        
        # bm25_scores = bm25.get_scores(tokenized_query)
        
        # # Normalize scores
        # if len(semantic_scores) > 1:
        #     semantic_scores = (semantic_scores - np.min(semantic_scores)) / \
        #                     (np.max(semantic_scores) - np.min(semantic_scores) + 1e-9)
        
        # if len(bm25_scores) > 1:
        #     bm25_scores = (bm25_scores - np.min(bm25_scores)) / \
        #                  (np.max(bm25_scores) - np.min(bm25_scores) + 1e-9)
        
        # # Combine scores
        # combined_scores = self.alpha * semantic_scores + (1 - self.alpha) * bm25_scores
        
        # # Sort chunks by combined score
        # sorted_indices = np.argsort(combined_scores)[::-1]
        
        # # Update scores in chunks
        # for i, idx in enumerate(sorted_indices):
        #     chunks[idx]['score'] = float(combined_scores[i])
        
        # return [chunks[i] for i in sorted_indices]
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenizer for BM25."""
        # This is a simple tokenizer - you might want to use a more sophisticated one
        return text.lower().split()

class RerankerFactory:
    """Factory for creating rerankers based on configuration."""
    
    @staticmethod
    def create_reranker(config: RerankerConfig) -> Reranker:
        """Create a reranker based on the configuration.
        
        Args:
            config: Configuration dictionary with 'name' and other parameters
            
        Returns:
            An instance of a Reranker
        """
        reranker_type = config.name
        
        if reranker_type == RerankerType.GEMINI:
            from .gemini_reranker import GeminiReranker
            return GeminiReranker(
                model_name=config.model_name,
                temperature=config.temperature,
                **{k: v for k, v in config.items() if k not in ['name', 'model_name', 'temperature']}
            )
        elif reranker_type == RerankerType.HYBRID:
            return HybridReranker(
                alpha=config.alpha,
                bm25_k1=config.bm25_k1,
                bm25_b=config.bm25_b
            )
        elif reranker_type == RerankerType.PARENT:
            return ParentDocumentReranker(
                top_k_parents=config.top_k_parents,
                max_chunks_per_parent=config.max_chunks_per_parent
            )
        else:
            raise ValueError(f"Unknown reranker type: {name}")
