"""Reranker implementation using Google's Gemini API."""
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from leo_assist.utils.logger import logger
from leo_assist.retrieval.ranking.base import Reranker

class GeminiReranker(Reranker):
    """Reranker implementation using Google's Gemini API."""
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-preview-05-06",
        temperature: float = 0.0,
        **kwargs
    ):
        """Initialize the Gemini reranker.
        
        Args:
            model_name: Name of the Gemini model to use
            temperature: Temperature for generation (0.0 for deterministic)
            **kwargs: Additional arguments for the Gemini client
        """
        self.model_name = model_name
        self.temperature = temperature
        self._client = None
        self._verify_google_auth()
    
    def _verify_google_auth(self):
        """Verify Google authentication is configured."""
        try:
            genai.configure()
            self._client = genai
            # Test the connection
            list(genai.list_models())
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Rerank documents based on relevance to the query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return (None for all)
            **kwargs: Additional arguments for the reranking
            
        Returns:
            List of reranked documents with updated scores
        """
        if not documents:
            return []
            
        try:
            # Prepare the documents for reranking
            doc_texts = [doc.get('content', doc.get('text', '')) for doc in documents]
            
            # Generate relevance scores using Gemini
            scores = []
            for doc_text in doc_texts:
                # Create a prompt for relevance scoring
                prompt = f"""
                Given the following query and document, rate the relevance of the document to the query 
                on a scale from 0.0 to 1.0, where 1.0 is most relevant. 
                Only respond with the score, nothing else.
                
                Query: {query}
                
                Document: {doc_text}
                
                Relevance score (0.0-1.0):
                """
                
                # Get the relevance score from Gemini
                response = self._client.generate_content(
                    model=f"models/{self.model_name}",
                    contents=prompt,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": 10,
                    },
                )
                
                try:
                    # Try to parse the score from the response
                    score = float(response.text.strip())
                    scores.append(score)
                except (ValueError, AttributeError):
                    # If parsing fails, use a default score
                    scores.append(0.0)
            
            # Update documents with scores
            for doc, score in zip(documents, scores):
                doc['score'] = float(score)
            
            # Sort by score in descending order
            documents.sort(key=lambda x: x.get('score', 0.0), reverse=True)
            
            # Return top-k documents if specified
            if top_k is not None:
                return documents[:top_k]
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in Gemini reranker: {str(e)}")
            # Return documents with default scores if there's an error
            for doc in documents:
                doc['score'] = doc.get('score', 0.0)
            return documents
