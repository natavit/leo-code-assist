"""Base downloader module.

This module contains the base downloader class that all other downloaders should inherit from.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging

from leo_assist.core.vector_store import Document

logger = logging.getLogger(__name__)

class BaseDownloader(ABC):
    """Base class for document downloaders."""
    
    def __init__(self):
        """Initialize the base downloader with default counters."""
        self.downloaded_files = 0
        self.skipped_files = 0
        self.failed_downloads = 0
    
    @abstractmethod
    def download(self) -> List[Document]:
        """Download documents from the source.
        
        Returns:
            List[Document]: List of downloaded documents
        """
        pass
    
    def get_stats(self) -> Dict[str, int]:
        """Get download statistics.
        
        Returns:
            Dict[str, int]: Dictionary with download statistics including:
                - downloaded: Number of successfully downloaded files
                - skipped: Number of skipped files
                - failed: Number of failed downloads
                - total: Total number of operations
        """
        return {
            "downloaded": self.downloaded_files,
            "skipped": self.skipped_files,
            "failed": self.failed_downloads,
            "total": self.downloaded_files + self.skipped_files + self.failed_downloads
        }
