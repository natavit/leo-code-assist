"""
Script to run the document ingestion pipeline.

This script loads and preprocesses documents from the configured data directory.
It's typically used as the first step before indexing documents in the vector store.

Usage:
    python scripts/run_ingestion.py

Environment Variables:
    DATA_DIR: Directory containing documents to ingest (from settings)
    LOG_LEVEL: Logging level (default: INFO)

Raises:
    Exception: If there's an error during the ingestion process
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

try:
    settings = Settings(_env_file=f"{AGENT_DIR}/.env")
    pipeline = create_default_pipeline(data_dir=settings.data_dir)
    documents = pipeline.run()

    logger.info("Ingestion completed successfully")
except Exception as e:
    logger.error(f"Error in document ingestion: {str(e)}", exc_info=True)
    raise