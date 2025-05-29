"""Interface definitions for vector store implementations.

This module provides the core interfaces and data structures for working with
vector stores in a type-safe and implementation-agnostic way.
"""
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Dict, List, Optional, Protocol, Any
from pydantic import BaseModel, Field
from leo_assist.core.embedding.types.task_type import TaskType


class DocumentType(str, Enum):
    """Enumeration of supported document types."""
    DOCUMENT = "document"
    CODE = "code"


class DocumentSource(str, Enum):
    """Sources of documents."""
    GITHUB = "github"
    WEB = "web"
    LOCAL = "local"


class Document(BaseModel):
    """A document in the knowledge base."""
    id: str
    title: str
    content: str
    type: DocumentType
    source: DocumentSource
    url: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(   
        default_factory=dict,
        description="Additional metadata about the document"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class DocumentChunk(BaseModel):
    """A chunk of a document with metadata."""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the document chunk"
    )
    parent_id: str
    chunk_index: int
    embedding: Optional[List[float]] = None
    
    class Config:
        arbitrary_types_allowed = True


class DocumentResult(BaseModel):
    """A document with content and metadata, including search score."""
    document: DocumentChunk
    score: float


class VectorStore(Protocol):
    """Protocol defining the interface for vector store implementations.
    
    This protocol defines the minimum interface that vector store implementations
    must provide to be compatible with our system.
    """
    
    def similarity_search(
        self,
        query: str,
        task_type: TaskType,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[DocumentResult]:
        """Search for similar documents.
        
        Args:
            query: The query string
            task_type: Task type for the query embedding
            top_k: Number of results to return
            filter: Optional filter to apply to the search
                
        Returns:
            List of matching documents, sorted by relevance
            
        Raises:
            ValueError: If task_type is invalid
        """
        ...
    
    def add_documents(
        self,
        document_chunks: List[DocumentChunk],
        task_type: TaskType,
        batch_size: int
    ) -> List[str]:
        """Add documents to the store.
        
        Args:
            document_chunks: List of document chunks to add
            task_type: Task type for the document embeddings
            batch_size: Number of documents to process in each batch
                
        Returns:
            List of document IDs for the added documents
            
        Raises:
            ValueError: If no documents provided
        """
        ...
    
    def delete_documents(
        self,
        doc_ids: List[str]
    ) -> bool:
        """Delete documents by their IDs.
        
        Args:
            doc_ids: List of document IDs to delete
                
        Returns:
            True if deletion was successful, False otherwise
        """
        ...
    
    def get_document(
        self,
        doc_id: str
    ) -> Optional[DocumentResult]:
        """Get a document by its ID.
        
        Args:
            doc_id: The document ID to retrieve
                
        Returns:
            The document if found, None otherwise
        """
        ...
    
    def clear(self) -> bool:
        """Clear all documents from the store.
        
        Returns:
            True if the operation was successful, False otherwise
        """
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        ...
