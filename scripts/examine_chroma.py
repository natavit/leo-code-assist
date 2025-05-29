#!/usr/bin/env python3
"""
Script to examine the structure and contents of a ChromaDB vector store.

This script connects to a ChromaDB instance and provides detailed information about
its collections, including metadata, document counts, and sample entries. It's useful
for debugging and understanding the current state of the vector store.

Usage:
    python scripts/examine_chroma.py

Environment Variables:
    VECTOR_STORE_PATH: Path to the ChromaDB storage (from settings)

Outputs:
    - List of all collections in the database
    - Document count for each collection
    - Metadata for each collection
    - Sample entries from each collection
"""

import sys
from pathlib import Path
import chromadb
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from leo_assist.utils.settings import Settings
from leo_assist.utils.logger import logger

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    
settings = Settings(_env_file=f"{ROOT_DIR}/.env")

try:
    client = chromadb.PersistentClient(path=str(settings.vector_store_path))

    collections = client.list_collections()

    logger.info(f"Found {len(collections)} collections:")

    for i, collection in enumerate(collections):
        logger.info(f"\nCollection {i+1}: {collection.name}")
        logger.info(f"Count: {collection.count()}")

        metadata = collection.metadata
        logger.info(f"Metadata: {metadata}")

        sample = collection.peek(limit=1)
        logger.info(f"Sample: {sample}")
except Exception as e:
    logger.error(f"Error listing collections: {e}")
