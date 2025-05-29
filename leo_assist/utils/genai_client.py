"""Global configuration for Google's Generative AI client.

This module provides a singleton instance of the Google Generative AI client
that uses the default Google Cloud credentials from the environment.
"""

from google import genai
from leo_assist.utils.logger import logger


def get_genai_client(google_cloud_project: str, google_cloud_location: str):
    """Get the global Gemini client instance.

    Returns:
        Configured Google Generative AI client

    Raises:
        ValueError: If authentication is not properly configured
    """
    try:
        client = genai.Client(
            vertexai=True,
            project=google_cloud_project,
            location=google_cloud_location,
        )
        return client
    except Exception as e:
        logger.error(
            f"Failed to initialize Gemini client. "
            f"Ensure GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION are set correctly. Error: {str(e)}"
        )
        raise
