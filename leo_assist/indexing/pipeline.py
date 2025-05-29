"""Document indexing pipeline with chunking and metadata handling.

This module provides functionality for indexing documents into a vector store with
configurable chunking strategies. It's designed to work with the DocumentRetriever
class for document retrieval operations.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, cast

from leo_assist.utils.logger import logger
from leo_assist.core.vector_store import Document
from leo_assist.core.vector_store import VectorStore
from leo_assist.core.chroma_db_impl import ChromaDBVectorStore
from leo_assist.core.chunking.manager import ChunkerManager
from leo_assist.indexing.indexer import DocumentIndexer

class DocumentIndexingPipeline:
    """Pipeline for indexing documents into a vector store with advanced chunking.
    
    This class handles the document indexing process including chunking and storing
    documents in the vector store. For document retrieval, use the DocumentRetriever
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        indexer: DocumentIndexer
    ):
        """Initialize the indexing pipeline.
        
        Args:
            vector_store: Vector store instance to use for storage
            indexer: The indexer instance to use for document processing
        """
        self.vector_store = vector_store
        self.indexer = indexer
    
    def run(
        self, 
        documents: List[Document],
        clear_existing: bool = False
    ) -> Dict[str, Any]:
        """Index a list of documents with advanced chunking.
        
        Args:
            documents: List of Document objects to index
            clear_existing: Whether to clear the vector store before indexing
            
        Returns:
            Dictionary with indexing statistics
        """
        if not documents:
            logger.warning("No documents to index")
            return {"status": "skipped", "reason": "No documents provided"}
        
        logger.info(f"Indexing {len(documents)} documents with chunking...")
        return self.indexer.index_documents(documents, clear_existing=clear_existing)
    
    def clear_index(self) -> bool:
        """Clear the entire index.
        
        Returns:
            bool: True if successful, False otherwise
        """
        return self.vector_store.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index.
        
        Returns:
            Dictionary with statistics
        """
        return self.vector_store.get_stats()

def create_default_indexing_pipeline(
    vector_store: Optional[VectorStore] = None,
    chunker_manager: Optional[ChunkerManager] = None
) -> DocumentIndexingPipeline:
    """Create a default document indexing pipeline with sensible defaults.
    
    Args:
        vector_store: Optional vector store instance. If not provided, a default one will be created.
        chunker_manager: Optional ChunkerManager instance. If not provided, a new one will be created.
        
    Returns:
        Configured DocumentIndexingPipeline instance
    """
    
    # Initialize ChunkerManager if not provided
    if chunker_manager is None:
        chunker_manager = ChunkerManager()
    
    # Create or use provided vector store
    if vector_store is None:
        vector_store = ChromaDBVectorStore.from_settings(settings, chunker_manager=chunker_manager)
    
    # Create the indexer
    indexer = DocumentIndexer(
        vector_store=vector_store,
        chunker_manager=chunker_manager
    )
    
    return DocumentIndexingPipeline(
        vector_store=vector_store,
        indexer=indexer
    )

def run_indexing_pipeline(
    input_file: Optional[Union[str, Path]] = None,
    input_dir: Optional[Union[str, Path]] = None,
    clear_existing: bool = False,
    vector_store: Optional[VectorStore] = None,
    chunker_manager: Optional[ChunkerManager] = None,
    indexer: Optional[Any] = None
) -> Dict[str, Any]:
    """Run the document indexing pipeline with the given inputs.
    
    Args:
        input_file: Path to a single file to index
        input_dir: Path to a directory of files to index
        output_dir: Directory to save processed files (optional)
        clear_existing: Whether to clear the index before adding new documents
        config_path: Path to configuration file (deprecated)
        vector_store: Optional pre-configured vector store
        chunker_manager: Optional pre-configured chunker manager
        indexer: Optional pre-configured indexer instance
        
    Returns:
        Dictionary with indexing statistics
    """
    
    # Create or use the provided indexer
    if indexer is not None:
        # Use the provided indexer
        pipeline = DocumentIndexingPipeline(
            vector_store=vector_store or indexer.vector_store,
            indexer=indexer
        )
    else:
        # Create a default pipeline
        pipeline = create_default_indexing_pipeline(
            vector_store=vector_store,
            chunker_manager=chunker_manager
        )
    
    try:
        # Process inputs
        if input_file:
            from ..ingestion import process_single_file
            documents = process_single_file(input_file)
        elif input_dir:
            from ..ingestion import process_directory
            documents = process_directory(input_dir)
        else:
            # If no specific input, run the full ingestion pipeline
            logger.info("No specific input provided, running full ingestion pipeline...")
            documents = ingestion_pipeline.run()
        
        if not documents:
            logger.warning("No documents to index")
            return {"status": "skipped", "reason": "No documents to index"}
        
        # Index documents
        logger.info(f"Indexing {len(documents)} documents...")
        result = pipeline.run(
            documents=documents,
            clear_existing=clear_existing
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}", exc_info=True)
        return {"status": "error", "reason": str(e)}

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    
    # Import settings after path setup
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from utils.config import settings
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Leo RAG Assistant - Document Indexing")
    parser.add_argument(
        "--input", 
        type=str,
        help="Input file to index (supports various formats based on file extension)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing files to index"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save processed files (optional)"
    )
    parser.add_argument(
        "--clear", 
        action="store_true",
        help="Clear existing index before adding new documents"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (deprecated, use environment variables)",
        default=None
    )
    # Chunker configuration
    parser.add_argument(
        "--chunker-config",
        type=str,
        default=None,
        help="Path to chunker config JSON file"
    )

    # Parse arguments
    args = parser.parse_args()
    
    # Prepare chunker config if provided
    chunker_config = None
    if args.chunker_config:
        try:
            import yaml
            with open(args.chunker_config, 'r') as f:
                chunker_config = yaml.safe_load(f)
                if not isinstance(chunker_config, dict):
                    raise ValueError("Chunker config must be a valid YAML dictionary")
        except Exception as e:
            logger.error(f"Invalid chunker config: {str(e)}")
            sys.exit(1)
    
    # Validate input arguments
    if not args.input and not args.input_dir:
        logger.warning("No input specified. Will run full ingestion pipeline if configured.")
    
    # Run the pipeline with settings
    try:
        # Create ChunkerManager with provided or default config
        chunker_manager = ChunkerManager()
        
        # Create a vector store from settings with the chunker manager
        vector_store = DocumentVectorStore.from_settings(settings, chunker_manager=chunker_manager)
        
        result = run_indexing_pipeline(
            input_file=args.input,
            input_dir=args.input_dir,
            clear_existing=args.clear,
            vector_store=vector_store,
            chunker_config=chunker_config,
            chunker_manager=chunker_manager
        )
        
        # Print results
        print("\nIndexing complete!")
        print(f"Status: {result.get('status', 'unknown')}")
        if 'reason' in result:
            print(f"Reason: {result['reason']}")
        if 'documents_indexed' in result:
            print(f"Documents indexed: {result['documents_indexed']}")
        if 'chunks_created' in result:
            print(f"Chunks created: {result['chunks_created']}")
            
        # Exit with appropriate status code
        sys.exit(0 if result.get('status') != 'error' else 1)
        
    except Exception as e:
        logger.error(f"Fatal error during indexing: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)
