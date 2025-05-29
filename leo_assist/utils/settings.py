"""Configuration settings for the Leo RAG Assistant.

This module provides a centralized configuration system using environment variables
with Pydantic settings management. All configuration is loaded from environment
variables, with sensible defaults provided.

Environment variables can be set in a .env file or directly in the environment.
For JSON fields (REPOSITORIES_JSON, WEBSITES_JSON), use valid JSON strings.
"""
from enum import Enum
import json
import logging
from pathlib import Path
from typing import List, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from leo_assist.core.embedding.types.task_type import TaskType

# Set up logging
logger = logging.getLogger(__name__)

# Custom types for better type hints
PathLike = Union[str, Path]

class RerankerType(str, Enum):
    """Supported reranker types."""
    GEMINI = "gemini"
    HYBRID = "hybrid"
    PARENT = "parent"

class RepositoryConfig(BaseModel):
    """Configuration for a Git repository data source."""
    url: str
    name: str
    branch: str = "main"
    include: List[str] = Field(
        default_factory=list,
        description="File extensions to include (e.g., ['.py', '.md']). If empty, all files are included."
    )
    exclude: List[str] = Field(
        default_factory=list,
        description="File patterns to exclude (e.g., ['*test*', '*/docs/*']). Supports glob patterns."
    )

class WebsiteConfig(BaseModel):
    """Configuration for a website data source."""
    url: str
    name: str
    include_paths: List[str] = Field(default_factory=list)
    exclude_paths: List[str] = Field(default_factory=list)

class RerankerConfig(BaseModel):
    """Reranker configuration."""
    name: RerankerType = RerankerType.PARENT
    model_name: str = "gemini-2.5-flash-preview-05-06"
    temperature: float = 0.0
    
    # Hybrid reranker settings
    alpha: float = 0.6
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    
    # Parent document reranker settings
    top_k_parents: int = 5
    max_chunks_per_parent: int = 3

class ChromaConfig(BaseModel):
    """ChromaDB configuration."""
    collection_name: str = "leo_docs"
    anonymized_telemetry: bool = False
    allow_reset: bool = True
    db_impl: str = "duckdb+parquet"
    api_impl: str = "rest"
    server_host: str = "localhost"
    server_http_port: int = 8000
    server_ssl_enabled: bool = False

# TaskType has been moved to src.core.types.task_type
# Chunking configuration is now handled in src.core.chunking.config

class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""
    # model_name: str = "gemini-embedding-exp-03-07"
    model_name: str = "text-embedding-005"
    dimension: int = 768
    task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT
    batch_size: int = 10
    max_retries: int = 3
    timeout: int = 30  # seconds
    
    @field_validator('task_type', mode='before')
    @classmethod
    def validate_task_type(cls, v: Union[str, TaskType]) -> TaskType:
        """Convert string task type to TaskType enum."""
        if isinstance(v, str):
            try:
                return TaskType(v.lower())
            except ValueError as e:
                raise ValueError(
                    f"Invalid task type: {v}. Must be one of {[t.value for t in TaskType]}"
                ) from e
        return v

class WebDownloaderConfig(BaseModel):
    """Web downloader configuration."""
    max_pages: int = 100
    request_delay: float = 1.0
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    timeout: int = 30

class LLMConfig(BaseModel):
    """LLM model configuration."""
    model: str = "gemini-2.5-pro-preview-05-06"
    max_tokens: int = 8096
    temperature: float = 0.5
    top_p: float = 1.0
    top_k: int = 32

class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    top_k: int = 10
    score_threshold: float = 0.6
    max_chars: int = 8000
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)


class Settings(BaseSettings):
    """Application settings and configuration.
    
    All settings can be configured via environment variables with the same name as the attribute.
    Nested settings use double underscore (__) for separation in environment variables.
    """
    model_config = SettingsConfigDict(
        extra='ignore',
        env_nested_delimiter='__',
    )
    
    # Project Information
    project_name: str = "Leo RAG Assistant"
    version: str = "0.1.0"
    debug: bool = False
    
    # File System Paths
    base_dir: Path = Path(".")
    data_dir: Path = Path("./data")
    vector_store_path: Path = data_dir / "vector_store"
    
    # Component Configurations
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    web_downloader: WebDownloaderConfig = Field(default_factory=WebDownloaderConfig)
    
    # Embedding Model Configuration
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    
    # LLM Configuration
    llm: LLMConfig = Field(default_factory=LLMConfig)
    
    # Retrieval Configuration
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    
    # Google Cloud Configuration
    google_genai_use_vertexai: bool = True  # GCP Project ID
    google_cloud_project: str = ""  # GCP Project ID
    google_cloud_location: str = ""  # GCP Region
    
    # Data Sources (loaded from JSON strings)
    repositories_json: str = '[]'  # JSON array of repository configurations
    websites_json: str = '[]'      # JSON array of website configurations
    
    # Parsed data sources (populated from JSON strings)
    repositories: List[RepositoryConfig] = Field(default_factory=list, exclude=True)
    websites: List[WebsiteConfig] = Field(default_factory=list, exclude=True)
    
    @field_validator('base_dir', 'data_dir', 'vector_store_path', mode='before')
    @classmethod
    def validate_paths(cls, v: PathLike) -> Path:
        """Ensure path-like fields are Path objects."""
        return Path(v) if isinstance(v, str) else v
    
    @field_validator('repositories_json', 'websites_json')
    @classmethod
    def validate_json_strings(cls, v: str) -> str:
        """Validate that JSON strings are properly formatted."""
        if not v or not v.strip():
            return '[]'
        try:
            json.loads(v)
            return v
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON string: {e}")
            return '[]'
    
    @model_validator(mode='after')
    def parse_json_fields(self) -> 'Settings':
        """Parse JSON strings into Python objects after validation."""
        # Parse repositories
        try:
            repos_data = json.loads(self.repositories_json)
            self.repositories = [RepositoryConfig(**repo) for repo in repos_data]
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Error parsing repositories_json: {e}")
            self.repositories = []
        
        # Parse websites
        try:
            websites_data = json.loads(self.websites_json)
            self.websites = [WebsiteConfig(**site) for site in websites_data]
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Error parsing websites_json: {e}")
            self.websites = []
            
        return self
    
    def ensure_dirs(self) -> None:
        """Ensure all required directories exist."""
        for path in [self.data_dir, self.vector_store_path]:
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path}")
            