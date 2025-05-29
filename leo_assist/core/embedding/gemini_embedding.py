"""Gemini embedding function for document and code embeddings."""
from typing import List, Optional, Dict, Any
from chromadb.api.types import Documents as ChromaDocuments, EmbeddingFunction, Embeddings

from google.genai import types

from leo_assist.utils.genai_client import get_genai_client
from leo_assist.utils.logger import logger
from leo_assist.utils.settings import EmbeddingConfig
from leo_assist.core.embedding.types.task_type import TaskType

class GeminiEmbeddingFunction(EmbeddingFunction[ChromaDocuments]):
    """Embedding function using Google's Gemini API."""
    
    def __init__(
        self, 
        embedding_config: EmbeddingConfig,
        task_type: TaskType,
        google_cloud_project: str,
        google_cloud_location: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Gemini embedding function.
        
        Args:
            model_name: Name of the Gemini embedding model to use. If None, uses settings.embedding.model_name
            config: Optional configuration overrides
            document_task_type: Task type for document embeddings. If None, uses config.task_type
        """
        # Use provided config or fall back to settings
        self.config = embedding_config.model_copy()
        self.genai_client = get_genai_client(
            google_cloud_project=google_cloud_project,
            google_cloud_location=google_cloud_location,
        )

        if config:
            self.config = self.config.model_validate({**self.config.model_dump(), **config})
            
        # Set task types
        self.task_type = task_type
    
    def __call__(self, input: ChromaDocuments) -> Embeddings:
        """Generate embeddings for the input documents.
        
        This is the main entry point that ChromaDB will call. It expects a list of text strings
        and returns a list of embeddings (each embedding is a list of floats).
        
        Args:
            input: A list of text strings to generate embeddings for
            
        Returns:
            A list of embeddings, where each embedding is a list of floats
        """
        return self.embed_documents(input)
        
    def embed_with_task_type(self, documents: ChromaDocuments, task_type: TaskType) -> Embeddings:
        """Generate embeddings with a specific task type.
        
        This allows dynamically setting the task type for each embedding call.
        
        Args:
            documents: A list of text strings to generate embeddings for
            task_type: The task type to use for this embedding
            
        Returns:
            A list of embeddings, where each embedding is a list of floats
        """
            
        # Save the current task type
        original_task_type = self.task_type
        try:
            # Set the new task type
            self.task_type = task_type
            # Generate embeddings
            return self.embed_documents(documents)
        finally:
            # Restore the original task type
            self.task_type = original_task_type
        
    def embed_documents(self, documents: ChromaDocuments) -> Embeddings:
        """Generate embeddings for multiple documents.
        
        Args:
            documents: A list of text strings to generate embeddings for
            
        Returns:
            A list of embeddings, where each embedding is a list of floats
        """
        if not documents:
            return []
            
        try:
            # Generate embeddings using the new client
            embeddings: List[List[float]] = []
            
            # Process in batches
            for i in range(0, len(documents), self.config.batch_size):
                # logger.info(f"Generated embeddings for batch {i // self.config.batch_size + 1} of {len(documents) // self.config.batch_size}")
                batch = documents[i:i + self.config.batch_size]
                
                # Create embedding request using the global client
                response = self.genai_client.models.embed_content(
                    model=self.config.model_name,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=self.task_type.value,
                        title=""  # Optional: Add title for better retrieval
                    )
                )
                
                # Extract embeddings from the response
                batch_embeddings = response.embeddings
                batch_embedding_values = [embedding.values for embedding in batch_embeddings]
                
                embeddings.extend(batch_embedding_values)
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    @property
    def dim(self) -> int:
        """Get the dimension of the embeddings.
        
        Returns:
            int: The dimension of the embeddings from the config
        """
        return self.config.dimension
