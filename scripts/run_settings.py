"""
Script to display the current application configuration.

This script loads and displays the application settings. It's useful for debugging
and verifying that the configuration is loaded correctly from environment variables
and .env files.

Usage:
    python scripts/run_settings.py

Outputs:
    - Full application settings in JSON format

Environment Variables:
    All settings are loaded from environment variables or .env file
    See leo_assist/utils/settings.py for the complete list of configurable options
"""

import sys
from pathlib import Path
import os

ROOT_DIR = Path(os.path.dirname(os.path.abspath(Path(__file__).parent)))
AGENT_DIR = ROOT_DIR / "leo_assist"

sys.path.insert(0, str(ROOT_DIR))

from leo_assist.utils.logger import logger
from leo_assist.utils.settings import Settings

settings = Settings(_env_file=f"{AGENT_DIR}/.env")

logger.info("Current settings:")
logger.info(settings.model_dump_json(indent=2))
