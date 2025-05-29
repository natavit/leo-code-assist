"""
Script to test the Gemini API client functionality.

This script demonstrates basic usage of the Gemini API client by sending a simple
prompt and printing the response. It's useful for verifying that the API client
is properly configured and can communicate with the Gemini API.

Usage:
    python scripts/run_genai_client.py

Environment Variables:
    GEMINI_API_KEY: Your Gemini API key (from settings)
    LLM_MODEL: The model to use for generation (from settings)

Example Output:
    The script will print the model's response to the prompt "Hello, how are you?"

Raises:
    Exception: If there's an error communicating with the Gemini API
"""

import sys
from pathlib import Path
import os

ROOT_DIR = Path(os.path.dirname(os.path.abspath(Path(__file__).parent)))
AGENT_DIR = ROOT_DIR / "leo_assist"

sys.path.insert(0, str(ROOT_DIR))

from leo_assist.utils.settings import Settings
from leo_assist.utils.logger import logger
from leo_assist.utils.genai_client import genai_client

settings = Settings(_env_file=f"{AGENT_DIR}/.env")

res = genai_client.models.generate_content(
    model=settings.llm.model,
    contents="Hello, how are you?"
)

logger.info(res.text)
