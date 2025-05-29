"""
Script to run the document indexing pipeline.

This script performs the following steps:
1. Initializes the vector store with the specified chunking strategy
2. Runs the document ingestion pipeline to load and preprocess documents
3. Indexes the documents in the vector store for efficient similarity search

Usage:
    python scripts/run_indexing.py

Environment Variables:
    DATA_DIR: Directory containing documents to index (from settings)
    CHROMA_DB_PATH: Path to store the vector database (from settings)
    CHUNK_SIZE: Size of document chunks (from settings)
    CHUNK_OVERLAP: Overlap between chunks (from settings)

Raises:
    Exception: If there's an error during the indexing process
"""

import sys
from pathlib import Path

import os

ROOT_DIR = Path(os.path.dirname(os.path.abspath(Path(__file__).parent)))
AGENT_DIR = ROOT_DIR / "leo_assist"

sys.path.insert(0, str(ROOT_DIR))

from leo_assist.utils.logger import logger
from leo_assist.utils.settings import Settings
from leo_assist.ingestion.pipeline import create_default_pipeline
from leo_assist.indexing.pipeline import create_default_indexing_pipeline
from leo_assist.core.chunking.manager import ChunkerManager
from leo_assist.core.chroma_db_impl import ChromaDBVectorStore

try:
    settings = Settings(_env_file=f"{AGENT_DIR}/.env")
    chunker_manager = ChunkerManager()
        
    # Create a vector store from settings with the chunker manager
    vector_store = ChromaDBVectorStore.from_settings(settings, chunker_manager=chunker_manager)

    ingestion_pipeline = create_default_pipeline(data_dir=settings.data_dir)
    documents = ingestion_pipeline.run()

    indexing_pipeline = create_default_indexing_pipeline(
        vector_store=vector_store,
        chunker_manager=chunker_manager
    )

    indexing_pipeline.run(documents)

    logger.info("Indexing completed successfully")
    
except Exception as e:
    logger.error(f"Error in document indexing: {str(e)}", exc_info=True)
    raise