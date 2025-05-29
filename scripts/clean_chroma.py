"""
Script to clean the ChromaDB vector store.

This script is designed to clear all data from the configured ChromaDB instance.
It's typically used for maintenance purposes or when you want to reset the vector store.

Usage:
    python scripts/clean_chroma.py

Environment Variables:
    CHROMA_DB_PATH: Path to the ChromaDB storage (from settings)
    CHUNK_SIZE: Document chunk size (from settings)
    CHUNK_OVERLAP: Overlap between chunks (from settings)

Raises:
    Exception: If there's an error during the cleaning process
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

try:
    settings = Settings(_env_file=f"{AGENT_DIR}/.env")
    chunker_manager = ChunkerManager()
    vector_store = ChromaDBVectorStore.from_settings(settings, chunker_manager=chunker_manager)
    vector_store.clear()
    logger.info("ChromaDB cleaned successfully")
except Exception as e:
    logger.error(f"Error in cleaning ChromaDB: {str(e)}", exc_info=True)
    raise