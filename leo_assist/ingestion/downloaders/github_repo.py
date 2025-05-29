"""GitHub repository downloader.

This module provides functionality to download and process GitHub repositories.
"""
import os
import time
import fnmatch
from pathlib import Path
from typing import List, Optional, Set, FrozenSet

import git
from tqdm import tqdm

from leo_assist.core.vector_store import Document, DocumentSource, DocumentType
from leo_assist.utils.logger import logger
from leo_assist.ingestion.downloaders.base import BaseDownloader

# Set of file extensions that should be classified as code documents
CODE_FILE_EXTENSIONS: FrozenSet[str] = frozenset({
    '.leo', '.py', '.js', '.java', '.c', '.cpp', '.h', '.hpp'
})

class GitHubRepoDownloader(BaseDownloader):
    """Downloader for GitHub repositories."""
    
    def __init__(
        self, 
        repo_url: str, 
        output_dir: Path,
        json_output_dir: Path,
        branch: str = "main",
        file_extensions: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_retries: int = 3,
        retry_delay: int = 5
    ):
        """Initialize GitHub repository downloader.
        
        Args:
            repo_url: URL of the GitHub repository
            output_dir: Local directory to clone the repository to
            branch: Branch to clone (default: main)
            file_extensions: List of file extensions to include (None for all files)
            exclude_patterns: List of glob patterns to exclude
            max_retries: Maximum number of retry attempts for failed operations
            retry_delay: Delay between retry attempts in seconds
        """
        super().__init__()
        self.repo_url = repo_url
        self.output_dir = output_dir
        self.json_output_dir = json_output_dir
        self.branch = branch
        self.file_extensions = [ext.lower() for ext in (file_extensions or [])]
        self.exclude_patterns = exclude_patterns or []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.repo_name = self._get_repo_name(repo_url)
        
        # Ensure local directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.json_output_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def _get_repo_name(repo_url: str) -> str:
        """Extract repository name from URL.
        
        Args:
            repo_url: URL of the GitHub repository
            
        Returns:
            str: Repository name
        """
        # Remove .git suffix if present
        repo_url = repo_url.rstrip('/')
        if repo_url.endswith('.git'):
            repo_url = repo_url[:-4]
        
        # Extract the repository name
        return repo_url.split('/')[-1]
    
    def _clone_or_pull_repo(self) -> str:
        """Clone or pull the repository.
        
        Returns:
            str: Path to the local repository
            
        Raises:
            git.GitCommandError: If git operations fail after max retries
        """
        repo_dir = self.output_dir / self.repo_name
        
        for attempt in range(self.max_retries):
            try:
                if repo_dir.exists():
                    # Pull latest changes if repository already exists
                    logger.info(f"Pulling latest changes for {self.repo_name}...")
                    repo = git.Repo(repo_dir)
                    origin = repo.remotes.origin
                    origin.pull()
                else:
                    # Clone the repository if it doesn't exist
                    logger.info(f"Cloning {self.repo_name}...")
                    repo = git.Repo.clone_from(
                        self.repo_url,
                        repo_dir,
                        branch=self.branch,
                        depth=1  # Shallow clone to save space
                    )
                
                return str(repo_dir)
                
            except git.GitCommandError as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to update repository {self.repo_name} after {self.max_retries} attempts")
                    raise
                    
                logger.warning(
                    f"Attempt {attempt + 1} failed for {self.repo_name}. "
                    f"Retrying in {self.retry_delay} seconds..."
                )
                time.sleep(self.retry_delay)
    
    def _should_include_file(self, file_path: str) -> bool:
        """Check if a file should be included based on include/exclude patterns.
        
        Args:
            file_path: Path to the file (relative to repository root)
            
        Returns:
            bool: True if the file should be included, False otherwise
        """
        # Convert to POSIX-style path for consistent pattern matching
        posix_path = file_path.replace('\\', '/')
        
        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(posix_path, pattern) or fnmatch.fnmatch('/' + posix_path, pattern):
                return False
        
        # If include patterns are specified, check against them
        if self.file_extensions:
            # Convert to lowercase for case-insensitive comparison
            lower_path = posix_path.lower()
            # Check each pattern using fnmatch for wildcard support
            for pattern in self.file_extensions:
                # Remove leading * if present (fnmatch handles it at the start)
                clean_pattern = pattern.lstrip('*')
                # Check if the pattern matches the end of the path
                if fnmatch.fnmatch(lower_path, f'*{clean_pattern}'):
                    return True
            return False
        
        return True
    
    def _read_file(self, file_path: Path) -> Optional[str]:
        """Read file content with error handling and retry logic.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            Optional[str]: File content if successful, None otherwise
        """
        encodings = ['utf-8', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Error reading file {file_path} with {encoding}: {str(e)}")
        
        logger.warning(f"Failed to read file {file_path} with any encoding")
        return None
    
    def download(self) -> List[Document]:
        """Download repository and return documents.
        
        Returns:
            List[Document]: List of documents from the repository
        """
        documents = []
        
        try:
            repo_path = self._clone_or_pull_repo()
            repo_path = Path(repo_path)

            json_repo_dir = self.json_output_dir / self.repo_name
            
            logger.info(f"Scanning {self.repo_name} for relevant files...")
            
            # Get all files first to show accurate progress
            all_files = [f for f in repo_path.rglob('*') if f.is_file()]
            
            for file_path in tqdm(all_files, desc=f"Processing {self.repo_name}"):
                rel_path = str(file_path.relative_to(repo_path))
                
                # Skip files that don't match the specified extensions
                if not self._should_include_file(rel_path):
                    self.skipped_files += 1
                    continue
                
                # Read file content
                content = self._read_file(file_path)
                if content is None:
                    self.failed_downloads += 1
                    continue
                
                # Determine document type based on file extension
                doc_type = DocumentType.CODE if file_path.suffix in CODE_FILE_EXTENSIONS else DocumentType.DOCUMENT
                
                # Create document
                doc = Document(
                    id=f"github_{self.repo_name}_{rel_path.replace('/', '_')}",
                    title=f"{self.repo_name}/{rel_path}",
                    content=content,
                    type=doc_type,
                    source=DocumentSource.GITHUB,
                    url=f"{self.repo_url}/blob/{self.branch}/{rel_path}",
                    file_path=str(file_path),
                    metadata={
                        "repo": self.repo_name,
                        "path": rel_path,
                        "branch": self.branch,
                        "language": file_path.suffix[1:] if file_path.suffix else "unknown"
                    }
                )

                try:
                    json_file_path = json_repo_dir / f"{rel_path}.json"
                    json_file_path.parent.mkdir(parents=True, exist_ok=True)
                    json_file_path.write_text(doc.model_dump_json(), encoding='utf-8')
                except Exception as e:
                    logger.warning(f"Failed to save {file_path} to file: {str(e)}")
                
                documents.append(doc)
                self.downloaded_files += 1
            
            logger.info(
                f"Processed {self.repo_name}: "
                f"{self.downloaded_files} files downloaded, "
                f"{self.skipped_files} files skipped, "
                f"{self.failed_downloads} files failed"
            )
            
            return documents
                
        except Exception as e:
            logger.error(f"Error processing repository {self.repo_name}: {str(e)}", exc_info=True)
            raise
