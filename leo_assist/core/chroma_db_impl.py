"""ChromaDB-based vector store implementation.

This module provides a ChromaDB implementation of the VectorStore interface.
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import Documents as ChromaDocuments
from typing import Optional

from leo_assist.core.chunking.manager import ChunkerManager
from leo_assist.utils.settings import Settings, EmbeddingConfig
from leo_assist.utils.logger import logger
from leo_assist.core.vector_store import DocumentChunk, DocumentResult
from leo_assist.core.vector_store import VectorStore, DocumentType
from leo_assist.core.embedding.gemini_embedding import GeminiEmbeddingFunction
from leo_assist.core.embedding.types.task_type import TaskType

class ChromaDBVectorStore(VectorStore):
    """ChromaDB implementation of the VectorStore interface.
    
    This class supports storing different types of documents (e.g., code and regular documents)
    in separate collections for better organization and search optimization.
    """
    
    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
        embedding_config: EmbeddingConfig,
        chunker_manager: ChunkerManager,
        document_type: DocumentType,
        google_cloud_project: str,
        google_cloud_location: str,
    ):
        """Initialize the document vector store.
        
        Args:
            persist_directory: Directory to persist the vector store
            collection_name: Base name of the collection to use (will be suffixed with document type)
            embedding_config: EmbeddingConfig object containing embedding configuration
            chunker_manager: ChunkerManager instance to use for dynamic chunking
            document_type: Type of documents being stored ('document' or 'code')
        """
        self.persist_directory = str(persist_directory)
        self.document_type = document_type
        # Append document type to collection name
        self.collection_name = f"{collection_name}_{self.document_type.value}"
        self.embedding_config = embedding_config
        self.chunker_manager = chunker_manager
        
        # Initialize client and collection with type hints
        self.client: Optional[ClientAPI] = None
        self.collection: Optional[chromadb.Collection] = None
        
        # Initialize embedding function
        self.embedding_function = GeminiEmbeddingFunction(
            embedding_config=self.embedding_config,
            task_type=self.embedding_config.task_type,
            google_cloud_project=google_cloud_project,
            google_cloud_location=google_cloud_location,
        )
        
        # Initialize Chroma client and collection
        self._init_chroma()
    
    def get_chunking_config(self, file_path: Optional[str] = None) -> Dict[str, int]:
        """Get chunking configuration for a specific file type.
        
        Args:
            file_path: Path to the file (used to determine file type)
            
        Returns:
            Dictionary with 'chunk_size' and 'overlap' keys
        """
        if file_path:
            # Get chunker for the file type
            chunker = self.chunker_manager.get_chunker_for_file(file_path)
            try:
                return {
                    "chunk_size": chunker.chunk_size,
                    "overlap": chunker.overlap
                }
            except AttributeError:
                # Fall through to default config if attributes are missing
                pass
        
        # Return defaults from chunker manager's default config
        default_config = self.chunker_manager.default_config
        return {
            "chunk_size": default_config.get("chunk_size", 1000),
            "overlap": default_config.get("overlap", 200)
        }
    
    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        chunker_manager: ChunkerManager,
        document_type: Union[DocumentType, str] = DocumentType.DOCUMENT,
    ) -> 'ChromaDBVectorStore':
        """Create a ChromaDBVectorStore instance from settings.
        
        Args:
            settings: Settings object containing all configuration
            chunker_manager: ChunkerManager instance to use for dynamic chunking
            document_type: Type of documents being stored ('document' or 'code')
                
        Returns:
            ChromaDBVectorStore: Configured instance
            
        Example:
            >>> from src.utils.config import Settings
            >>> from src.core.chunking.manager import ChunkerManager
            >>> settings = Settings()
            >>> chunker_manager = ChunkerManager()
            >>> # For documents
            >>> doc_store = DocumentVectorStore.from_settings(settings, chunker_manager, document_type='document')
            >>> # For code
            >>> code_store = DocumentVectorStore.from_settings(settings, chunker_manager, document_type='code')
        """
        return cls(
            persist_directory=settings.vector_store_path,
            collection_name=settings.chroma.collection_name,
            embedding_config=settings.embedding,
            chunker_manager=chunker_manager,
            document_type=document_type,
            google_cloud_project=settings.google_cloud_project,
            google_cloud_location=settings.google_cloud_location,
        )
    
    def _init_chroma(self) -> None:
        """Initialize the Chroma client and collection using instance settings.
        
        This method is called only once during instance initialization.
        It checks if the database already exists before creating a new one.
        """
            
        try:
            # Check if database already exists
            db_exists = os.path.exists(os.path.join(self.persist_directory, "chroma.sqlite3"))
            
            # Create client with instance settings
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            if db_exists:
                logger.info(
                    f"Connected to existing Chroma vector store at {self.persist_directory} "
                    f"with collection '{self.collection_name}'"
                )
            else:
                logger.info(
                    f"Created new Chroma vector store at {self.persist_directory} "
                    f"with collection '{self.collection_name}'"
                )
            
        except Exception as e:
            logger.error(f"Error initializing Chroma client: {str(e)}")
            raise
    
    def _convert_to_chroma_documents(self, chunks: List[DocumentChunk]) -> Tuple[List[str], ChromaDocuments, List[Dict[str, Any]]]:
        """Convert a list of DocumentChunks to ChromaDB format.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            Tuple of (ids, contents, metadatas)
        """
        ids = []
        contents = []
        metadatas = []
        
        for chunk in chunks:
            # Use the chunk's ID or generate one if not provided
            chunk_id = chunk.id or f"{chunk.parent_id}_chunk_{chunk.chunk_index}"
            ids.append(chunk_id)
            contents.append(chunk.content)
            
            # Start with the chunk's metadata or an empty dict
            metadata = chunk.metadata or {}
            
            # Add chunk-specific metadata
            metadata.update({
                "id": chunk_id,
                "parent_id": chunk.parent_id,
                "chunk_index": chunk.chunk_index,
                "is_chunk": True  # Mark this as a chunk for filtering if needed
            })
            
            # If the chunk has an embedding, it will be used by ChromaDB
            if chunk.embedding is not None:
                metadata["has_embedding"] = True
            
            metadatas.append(metadata)
            
        return ids, contents, metadatas
    
    def add_documents(
        self, 
        document_chunks: List[DocumentChunk],
        task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT,
        batch_size: int = 10
    ) -> List[str]:
        """Add document chunks to the vector store.
        
        Args:
            documents: List of DocumentChunk objects to add
            task_type: Task type for the document embeddings. If None, uses the default from config.
                     Must be a TaskType enum value.
            batch_size: Number of chunks to process in each batch (default: 10)
                
        Returns:
            List of chunk IDs that were added
            
        Raises:
            ValueError: If no document chunks provided
        """
        if not document_chunks:
            raise ValueError("No document chunks provided to add to vector store")
            
        # Verify all items are DocumentChunk instances
        if not all(isinstance(doc, DocumentChunk) for doc in document_chunks):
            raise TypeError("All documents must be instances of DocumentChunk")
            
        try:
            document_ids: List[str] = []
            logger.info(f"Adding {len(document_chunks)} documents to vector store...")
            
            # Process documents in batches
            for i in range(0, len(document_chunks), batch_size):
                batch = document_chunks[i:i + batch_size]
                
                # Convert documents to Chroma format
                ids, contents, metadatas = self._convert_to_chroma_documents(batch)
                
                # Generate embeddings for all documents in the batch with the specified task type
                embeddings = self.embedding_function.embed_with_task_type(contents, task_type)
                
                # Add to collection with the specified task type
                if ids:
                    self.collection.upsert(
                        ids=ids,
                        documents=contents,
                        metadatas=metadatas,
                        embeddings=embeddings
                    )
                    document_ids.extend(ids)
                    logger.debug(f"Added batch of {len(ids)} documents to vector store")
            
            logger.info(f"Successfully added {len(document_ids)} documents to vector store")
            return document_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def _convert_chroma_result_to_document(
        self, 
        result: Dict[str, Any], 
        index: int = 0
    ) -> List[DocumentResult]:
        """Convert a single ChromaDB result to a list of DocumentResult objects.
        
        Args:
            result: The ChromaDB query result
            index: The index of the query result to process
            
        Returns:
            List of DocumentResult objects
        """
        chunks: List[DocumentResult] = []
        
        # Check if we have any results
        if not result.get("ids") or not result["ids"]:
            return chunks
            
        # Get the number of results for this index
        num_results = len(result["ids"][index]) if index < len(result["ids"]) else 0
        
        for i in range(num_results):
            try:
                chunk_id = result["ids"][index][i]
                content = result["documents"][index][i]
                metadata = result["metadatas"][index][i] if result.get("metadatas") and result["metadatas"] else {}
                distance = result.get("distances", [None])[index][i] if result.get("distances") else None
                
                # Calculate score (1 - distance) since smaller distances are better
                score = 1.0 - float(distance) if distance is not None else 0.0
                
                # Ensure metadata is a dictionary
                if not isinstance(metadata, dict):
                    metadata = {}
                
                # Get document ID and chunk index from metadata or generate defaults
                parent_id = metadata.get("parent_id", "unknown_document")
                chunk_index = metadata.get("chunk_index", 0)
                
                # Create document chunk
                chunk = DocumentResult(
                    document=DocumentChunk(
                        id=chunk_id,
                        content=content,
                        metadata=metadata,
                        parent_id=parent_id,
                        chunk_index=chunk_index
                    ),
                    score=score
                )
                
                chunks.append(chunk)
                
            except (IndexError, KeyError, TypeError) as e:
                logger.warning(f"Error processing document chunk at index {i}: {str(e)}")
                continue
                
        return chunks
    
    def similarity_search(
        self,
        query: str,
        task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentResult]:
        """Search for similar documents to the query.
        
        Args:
            query: Search query
            task_type: Task type for the query embedding. If None, uses the default from config.
                     Must be a TaskType enum value.
            top_k: Number of results to return
            filter: Optional filter to apply to the search
            
        Returns:
            List of DocumentResult objects matching the query
            
        Raises:
            ValueError: If task_type is invalid
            TypeError: If task_type is provided but not a TaskType enum
        """
        try:
            # Generate query embedding with the specified task type
            query_embedding = self.embedding_function.embed_with_task_type(
                [query],
                task_type=task_type
            )[0]
            
            # Perform the search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter,
            )
            
            # Convert results to Document objects
            return self._convert_chroma_result_to_document(results)
            
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[DocumentResult]:
        """Get a document chunk by ID.
        
        Args:
            doc_id: Document chunk ID
            
        Returns:
            DocumentResult if found, None otherwise
        """
        try:
            result = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])
            if not result["ids"]:
                return None
                
            metadata = result["metadatas"][0] if result["metadatas"] else {}
            document_id = metadata.get("document_id", "unknown_document")
            chunk_index = metadata.get("chunk_index", 0)
            
            return DocumentResult(
                document=DocumentChunk(
                    id=result["ids"][0],
                    content=result["documents"][0],
                    metadata=metadata,
                    document_id=document_id,
                    chunk_index=chunk_index
                ),
                score=metadata.get("score", 0.0)
            )
        except Exception as e:
            logger.error(f"Error getting document chunk {doc_id}: {str(e)}")
            return None
    
    def delete_documents(self, doc_ids: List[str], **kwargs) -> bool:
        """Delete documents by their IDs.
        
        Args:
            doc_ids: List of document IDs to delete
            **kwargs: Additional arguments for the vector store
            
        Returns:
            True if successful, False otherwise
        """
        if not doc_ids:
            logger.warning("No document IDs provided for deletion")
            return False
            
        try:
            self.collection.delete(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents from vector store")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False
    
    def clear(self, **kwargs) -> bool:
        """Clear the entire vector store.
        
        Args:
            **kwargs: Additional arguments for the vector store
            
        Returns:
            True if successful, False otherwise
            
        Note:
            After clearing, you should create a new instance of DocumentVectorStore.
            The current instance should not be used after calling clear().
        """
        try:
            if not self.client:
                logger.warning("No client initialized, nothing to clear")
                return False
                
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Cleared vector store collection: {self.collection_name}")
            
            # Mark as uninitialized to prevent further operations
            self._initialized = False
            self.collection = None
            
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting vector store stats: {str(e)}")
            return {
                "error": str(e)
            }
