"""Enhanced document indexer with advanced chunking and metadata handling."""
import logging
from typing import List, Dict, Any

from pydantic import BaseModel

from leo_assist.core.vector_store import Document, DocumentChunk
from leo_assist.core.chunking.manager import ChunkerManager
from leo_assist.core.vector_store import VectorStore
from leo_assist.utils.logger import logger

class ProcessedChunk(BaseModel):
    """Type definition for a processed document chunk ready for vector storage.
    
    Attributes:
        id: Unique identifier for the chunk (format: "{document_id}_chunk_{index}")
        content: The actual text content of the chunk
        metadata: Dictionary containing chunk metadata including:
            - chunk_index: Position of this chunk in the document
            - total_chunks: Total number of chunks in the document
            - is_complete: Whether this is a complete chunk or truncated
            - chunk_type: Type of the chunk (e.g., 'text', 'code', etc.)
            - Additional document-specific metadata
    """
    id: str
    content: str
    metadata: Dict[str, Any]  # Contains both chunk metadata and document metadata

class DocumentIndexer:
    """Document indexer with configurable chunking and metadata handling."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        chunker_manager: ChunkerManager,
    ):
        """Initialize the document indexer.
        
        Args:
            vector_store: Vector store instance to use
            chunker_manager: ChunkerManager instance to use for document chunking
        """
        self.vector_store = vector_store
        self.chunker_manager = chunker_manager
    
    def index_documents(
        self, 
        documents: List[Document],
        clear_existing: bool = False
    ) -> Dict[str, Any]:
        """Index a list of documents with chunking and metadata.
        
        Args:
            documents: List of Document objects to index
            clear_existing: Whether to clear the vector store before indexing
            
        Returns:
            Dictionary with indexing statistics
        """
        if not documents:
            logger.warning("No documents to index")
            return {"status": "skipped", "reason": "No documents provided"}
        
        try:
            # Clear existing data if requested
            if clear_existing:
                logger.info("Clearing existing index...")
                self.vector_store.clear()
            
            # Process documents
            total_chunks = 0
            for i in range(0, len(documents)):
                logger.info(f"Processing document {i + 1} of {len(documents)} / {documents[i].id}")
                doc = documents[i]
                chunks = []
                
                # Process each document using _process_document
                processed_chunks = self._process_document(doc)
                logger.info(f"Generated {len(processed_chunks)} chunks for document: {doc.id}") 
                
                chunks.extend([
                    DocumentChunk(
                        id=chunk.id,
                        content=chunk.content,
                        metadata=chunk.metadata,
                        parent_id=doc.id,
                        chunk_index=chunk.metadata["chunk_index"]
                    ) for chunk in processed_chunks
                ])
                
                # Add chunks to vector store
                if chunks:
                    self.vector_store.add_documents(chunks)
                    total_chunks += len(chunks)
                
                logger.debug(f"Processed document {i + 1}, added {len(chunks)} chunks")
            
            # Get final stats
            stats = self.vector_store.get_stats()
            logger.info(
                f"Indexing complete. Added {total_chunks} chunks from {len(documents)} documents. "
                f"Total documents in index: {stats.get('document_count', 0)}"
            )
            
            return {
                "status": "success",
                "document_count": len(documents),
                "chunk_count": total_chunks,
                "vector_store_stats": stats
            }
            
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _process_document(self, document: Document) -> List[ProcessedChunk]:
        """Process a single document into chunks with metadata.
        
        Args:
            document: Document to process
            
        Returns:
            List of chunk dictionaries ready for indexing
            
        Note:
            This is an internal method used by index_documents to process individual documents.
        """
        # Get source path for chunker selection
        source_path = document.url
        
        # Prepare metadata for chunking
        metadata = {
            **document.metadata,
            "document_id": document.id,
            "document_type": document.type.value
        }
        
        # Generate chunks using the chunker manager
        doc_chunks = self.chunker_manager.chunk_document(
            text=document.content,
            file_path=source_path,
            metadata=metadata
        )
        
        # Convert chunks to the format expected by the vector store
        processed_chunks: List[ProcessedChunk] = []
        for i, chunk in enumerate(doc_chunks):
            chunk_metadata = chunk.metadata.copy() if hasattr(chunk, 'metadata') else {}
            chunk_metadata.update({
                "chunk_index": i,
                "total_chunks": len(doc_chunks),
                "is_complete": chunk.is_complete,
                "chunk_type": chunk.chunk_type
            })

            processed_chunks.append(ProcessedChunk(
                id=f"{document.id}_chunk_{i}",
                content=chunk.content,
                metadata=chunk_metadata
            ))
            
        return processed_chunks
