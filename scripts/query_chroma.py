"""
Script to perform similarity search in a ChromaDB vector store.

This script demonstrates how to query the ChromaDB vector store for documents
similar to a given query string. It's useful for testing document retrieval
functionality and verifying the contents of the vector store.

Usage:
    python scripts/query_chroma.py

Environment Variables:
    CHROMA_DB_PATH: Path to the ChromaDB storage (from settings)
    CHUNK_SIZE: Document chunk size (from settings)
    CHUNK_OVERLAP: Overlap between chunks (from settings)

Example:
    The script performs a similarity search for "help me write a basic lottery app"
    and returns the most relevant document chunks from the vector store.

Raises:
    Exception: If there's an error during the query process
"""

import sys
from pathlib import Path
import os

ROOT_DIR = Path(os.path.dirname(os.path.abspath(Path(__file__).parent)))
AGENT_DIR = ROOT_DIR / "leo_assist"

sys.path.insert(0, str(ROOT_DIR))

from leo_assist.utils.logger import logger
from leo_assist.utils.settings import Settings
from leo_assist.core.chunking.manager import ChunkerManager
from leo_assist.core.chroma_db_impl import ChromaDBVectorStore
from leo_assist.core.embedding.types.task_type import TaskType

try:
    settings = Settings(_env_file=f"{AGENT_DIR}/.env")
    chunker_manager = ChunkerManager()

    # Create a vector store from settings with the chunker manager
    vector_store = ChromaDBVectorStore.from_settings(
        settings, chunker_manager=chunker_manager
    )

    docs = vector_store.similarity_search(
        "help me write a basic lottery app", task_type=TaskType.CODE_RETRIEVAL_QUERY
    )

    for doc in docs:
        logger.info(doc.model_dump_json(indent=2))
except Exception as e:
    logger.error(f"Error in query ChromaDB: {str(e)}", exc_info=True)
    raise
