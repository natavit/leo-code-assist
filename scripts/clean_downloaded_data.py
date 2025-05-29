"""
Script to clean the downloaded data directory.

This script removes the entire data directory specified in the settings,
which typically contains downloaded or processed files that can be regenerated.

Usage:
    python scripts/clean_downloaded_data.py

Environment Variables:
    DATA_DIR: Path to the data directory (from settings)

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


try:
    settings = Settings(_env_file=f"{AGENT_DIR}/.env")
    data_dir = settings.data_dir

    if data_dir.exists():
        import shutil
        shutil.rmtree(data_dir)

    logger.info("Data directory cleaned successfully")
except Exception as e:
    logger.error(f"Error in cleaning data directory: {str(e)}", exc_info=True)
    raise