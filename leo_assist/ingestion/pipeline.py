"""Main module for document ingestion pipeline.

This module provides the DocumentIngestionPipeline class which coordinates
the downloading and processing of documents from various sources.
"""
import time
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from leo_assist.utils.settings import Settings
from leo_assist.utils.logger import logger
from leo_assist.core.vector_store import Document
from leo_assist.ingestion.downloaders.base import BaseDownloader
from leo_assist.ingestion.downloaders.github_repo import GitHubRepoDownloader
from leo_assist.ingestion.downloaders.web_documentation import WebDocumentationDownloader

class DocumentIngestionPipeline:
    """Pipeline for ingesting documents from various sources."""
    
    def __init__(
        self, 
        data_dir: Path,
        config_path: Optional[Path] = None,
        max_workers: int = 4
    ):
        """Initialize the ingestion pipeline.
        
        Args:
            data_dir: Base directory for storing downloaded data
            config_path: Path to configuration file (deprecated)
            max_workers: Maximum number of worker threads for parallel downloads
        """
        self.data_dir = data_dir
        self.max_workers = max_workers
        self.downloaders: List[Tuple[str, BaseDownloader]] = []
        self.documents: List[Document] = []
        
        # Log deprecation warning for config_path
        if config_path is not None:
            logger.warning(
                "The 'config_path' parameter is deprecated and will be removed in a future version. "
                "Please use environment variables or the Settings class for configuration.",
            )
            
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_settings(
        cls,
        settings: Settings,
    ) -> 'DocumentIngestionPipeline':
        """Create a DocumentIngestionPipeline instance from settings.
        
        Args:
            settings: Application settings object
            data_dir: Optional base directory for storing downloaded data
            
        Returns:
            DocumentIngestionPipeline: Configured instance
        """
        data_dir = settings.data_dir
        return cls(
            data_dir=data_dir,
        )
    
    def add_github_source(
        self, 
        repo_url: str,
        branch: str = "main",
        file_extensions: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_retries: int = 3,
        retry_delay: int = 5
    ) -> 'DocumentIngestionPipeline':
        """Add a GitHub repository as a data source.
        
        Args:
            repo_url: URL of the GitHub repository
            branch: Branch to clone (default: main)
            file_extensions: List of file extensions to include (None for all files).
                          Example: ['.py', '.md']
            exclude_patterns: List of glob patterns to exclude. 
                           Example: ['*test*', '*/docs/*']
            
        Returns:
            Self for method chaining
        """
        output_dir = self.data_dir / "repos"
        output_dir.mkdir(exist_ok=True)
        md_output_dir = self.data_dir / "repos_md"
        md_output_dir.mkdir(exist_ok=True)
        
        downloader = GitHubRepoDownloader(
            repo_url=repo_url,
            output_dir=output_dir,
            json_output_dir=md_output_dir,
            branch=branch,
            file_extensions=file_extensions or ['.leo', '.md', '.txt'],
            exclude_patterns=exclude_patterns or [],
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        self.downloaders.append((f"GitHub: {repo_url}", downloader))
        return self
    
    def add_web_documentation(
        self,
        base_url: str,
        allowed_domains: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        max_pages: Optional[int] = 100,
        request_delay: Optional[float] = None,
        max_retries: int = 3,
        retry_delay: int = 5,
        user_agent: Optional[str] = None,
        timeout: int = 10
    ) -> 'DocumentIngestionPipeline':
        """Add a web documentation site as a data source.
        
        Args:
            base_url: Base URL of the documentation site
            allowed_domains: List of allowed domains (None for base domain only)
            exclude_paths: List of URL paths to exclude
            max_pages: Maximum number of pages to download
            
        Returns:
            Self for method chaining
        """
        output_dir = self.data_dir / "web_docs"
        output_dir.mkdir(exist_ok=True)
        
        downloader = WebDocumentationDownloader(
            base_url=base_url,
            output_dir=output_dir,
            allowed_domains=allowed_domains,
            exclude_paths=exclude_paths or [],
            max_pages=max_pages,
            request_delay=request_delay,
            max_retries=max_retries,
            retry_delay=retry_delay,
            user_agent=user_agent,
            timeout=timeout
        )
        self.downloaders.append((f"Web: {base_url}", downloader))
        return self
    
    def _process_downloader(
        self, 
        source_name: str, 
        downloader: BaseDownloader
    ) -> Tuple[str, List[Document], Dict[str, int]]:
        """Process a single downloader and return the results.
        
        Args:
            source_name: Name of the data source
            downloader: Downloader instance
            
        Returns:
            Tuple of (source_name, documents, stats)
        """
        try:
            logger.info(f"Processing source: {source_name}")
            start_time = time.time()
            
            # Download documents
            documents = downloader.download()
            
            # Get statistics
            stats = downloader.get_stats()
            stats['processing_time'] = time.time() - start_time
            
            logger.info(
                f"Completed {source_name} in {stats['processing_time']:.2f}s: "
                f"{stats['downloaded']} downloaded, "
                f"{stats['skipped']} skipped, "
                f"{stats['failed']} failed"
            )
            
            return source_name, documents, stats
            
        except Exception as e:
            logger.error(f"Error processing {source_name}: {str(e)}", exc_info=True)
            return source_name, [], {
                'downloaded': 0,
                'skipped': 0,
                'failed': 0,
                'error': str(e)
            }
    
    def run(self, parallel: bool = False) -> List[Document]:
        """Run the ingestion pipeline.
        
        Args:
            parallel: Whether to process sources in parallel
            
        Returns:
            List of Document objects
        """
        if not self.downloaders:
            logger.warning("No data sources configured. Add sources using add_* methods.")
            return []
        
        logger.info(f"Starting document ingestion pipeline with {len(self.downloaders)} sources...")
        start_time = time.time()
        
        if parallel and len(self.downloaders) > 1:
            # Process downloaders in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self._process_downloader, name, dl)
                    for name, dl in self.downloaders
                ]
                
                for future in concurrent.futures.as_completed(futures):
                    source_name, documents, stats = future.result()
                    self.documents.extend(documents)
        else:
            # Process downloaders sequentially
            for source_name, downloader in self.downloaders:
                _, documents, _ = self._process_downloader(source_name, downloader)
                self.documents.extend(documents)
        
        total_time = time.time() - start_time
        logger.info(
            f"Ingestion completed in {total_time:.2f}s. "
            f"Total documents: {len(self.documents)}"
        )
        
        return self.documents
    
    def get_documents(self) -> List[Document]:
        """Get all ingested documents.
        
        Returns:
            List of Document objects
        """
        return self.documents
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_documents": len(self.documents),
            "sources": [
                {"name": name, **downloader.get_stats()}
                for name, downloader in self.downloaders
            ]
        }

def create_default_pipeline(
    settings: Settings,
) -> 'DocumentIngestionPipeline':
    
    """Create a pre-configured ingestion pipeline with default sources.
    
    Args:
        data_dir: Optional base directory for storing downloaded data.
                 Defaults to settings.data_dir or 'data' if not set.
        max_workers: Maximum number of worker threads for parallel downloads.
                   Defaults to settings.llm.max_workers or 4.
    
    Returns:
        Configured DocumentIngestionPipeline instance
    """
    
    # Initialize pipeline
    pipeline = DocumentIngestionPipeline(
        data_dir=settings.data_dir,
        max_workers=1,
    )
    
    # Add default GitHub repositories from settings if available
    if settings.repositories:
        for repo in settings.repositories:
            pipeline.add_github_source(
                repo_url=repo.url,
                branch=repo.branch,
                file_extensions=repo.include or ['.leo', '.md', '.txt'],
                exclude_patterns=repo.exclude or [],
                max_retries=1,
                retry_delay=5
            )
    
    # Add web documentation sources from settings if available
    if settings.websites and settings.web_downloader:
        for site in settings.websites:
            pipeline.add_web_documentation(
                base_url=site.url,
                allowed_domains=[site.url.split('//')[-1].split('/')[0]] if not site.include_paths else None,
                exclude_paths=site.exclude_paths + ['/sitemap.xml', '/robots.txt'],
                max_pages=settings.web_downloader.max_pages,
                request_delay=1.0,
                max_retries=1,
                retry_delay=5,
                user_agent=settings.web_downloader.user_agent if settings.web_downloader else None
            )
    
    return pipeline

# if __name__ == "__main__":
#     import argparse
#     import sys
#     from pathlib import Path
    
#     # Add project root to path for imports
#     sys.path.insert(0, str(Path(__file__).parent.parent.parent))
#     from utils.config import settings
    
#     # Set up argument parsing
#     parser = argparse.ArgumentParser(description="Leo RAG Assistant - Document Ingestion")
#     parser.add_argument(
#         "--input", 
#         type=str,
#         help="Input file or directory to process (optional, uses default sources if not provided)"
#     )
#     parser.add_argument(
#         "--input-dir",
#         type=str,
#         help="Directory containing files to process"
#     )
#     parser.add_argument(
#         "--output-dir",
#         type=str,
#         help="Output directory for processed files (default: data/processed)"
#     )
#     parser.add_argument(
#         "--data-dir",
#         type=str,
#         help="Base directory for downloaded data (default: data)"
#     )
#     parser.add_argument(
#         "--clear", 
#         action="store_true",
#         help="Clear existing output directory before processing"
#     )
#     parser.add_argument(
#         "--config",
#         type=str,
#         help="Path to configuration file (deprecated, use environment variables)",
#         default=None
#     )
    
#     # Parse arguments
#     args = parser.parse_args()
    
#     # Configure logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#     )
    
#     try:
#         # Set up directories
#         data_dir = Path(args.data_dir) if args.data_dir else Path(getattr(settings, 'data_dir', 'data'))
#         output_dir = Path(args.output_dir) if args.output_dir else data_dir / "processed"
        
#         # Create and run the pipeline with settings
#         pipeline = create_default_pipeline(
#             data_dir=data_dir,
#             output_dir=output_dir
#         )
        
#         # Clear output directory if requested
#         if args.clear and output_dir.exists():
#             import shutil
#             shutil.rmtree(output_dir)
#             output_dir.mkdir(parents=True, exist_ok=True)
        
#         # Process input if provided
#         if args.input:
#             input_path = Path(args.input)
#             if input_path.is_file():
#                 documents = pipeline.process_file(input_path)
#             elif input_path.is_dir():
#                 documents = pipeline.process_directory(input_path)
#             else:
#                 raise ValueError(f"Input path does not exist: {args.input}")
            
#             # Save processed documents
#             output_dir.mkdir(parents=True, exist_ok=True)
            
#             for doc in documents:
#                 output_file = output_dir / f"{doc.id}.json"
#                 with open(output_file, 'w', encoding='utf-8') as f:
#                     import json
#                     json.dump(doc.dict(), f, indent=2)
            
#             logger.info(f"Processed {len(documents)} documents to {output_dir}")
#         else:
#             # Run with default sources from settings
#             pipeline.run()
#             logger.info("Ingestion completed successfully")
            
#     except Exception as e:
#         logger.error(f"Error in document ingestion: {str(e)}", exc_info=True)
#         raise
