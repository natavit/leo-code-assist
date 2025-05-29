"""Default configurations for chunkers.

This module contains default configurations for different file types used by the ChunkerManager.
"""
from typing import Dict, Optional, Literal

from pydantic import BaseModel

# Type aliases
ChunkerType = Literal["langchain", "tree_sitter_leo", "leo_regex"]

class ChunkerConfig(BaseModel):
    """Type hint for chunker configuration."""
    type: ChunkerType
    language: Optional[str]
    chunk_size: int
    overlap: int

def _create_config(
    chunker_type: ChunkerType,
    language: Optional[str] = None,
    chunk_size: int = 1000,
    overlap: int = 200
) -> ChunkerConfig:
    """Helper function to create a typed chunker config."""
    return ChunkerConfig(
        type=chunker_type,
        language=language,
        chunk_size=chunk_size,
        overlap=overlap
    )

# Default chunker configurations for different file types
DEFAULT_CONFIGS: Dict[str, ChunkerConfig] = {
    # Text files
    "default": _create_config("langchain"),
    
    # Python files
    ".py": _create_config("langchain", "python"),
    
    # Leo files - use Tree-sitter based chunker
    ".leo": _create_config("langchain", "leo", 1000, 200),
    
    # Configuration files
    ".toml": _create_config("langchain", "toml"),
    ".yaml": _create_config("langchain", "yaml"),
    ".yml": _create_config("langchain", "yaml"),
    
    # Documentation
    ".md": _create_config("langchain", "markdown", 1000, 200),
    
    # Source code files
    ".js": _create_config("langchain", "js"),
    ".ts": _create_config("langchain", "ts"),
    ".jsx": _create_config("langchain", "jsx"),
    ".tsx": _create_config("langchain", "tsx"),
    ".java": _create_config("langchain", "java"),
    ".go": _create_config("langchain", "go"),
    ".rb": _create_config("langchain", "ruby"),
    ".php": _create_config("langchain", "php"),
    ".cs": _create_config("langchain", "csharp"),
    ".cpp": _create_config("langchain", "cpp"),
    ".h": _create_config("langchain", "cpp"),
    ".hpp": _create_config("langchain", "cpp"),
    
    # Data files
    ".json": _create_config("langchain", "json"),
    ".xml": _create_config("langchain", "xml"),
    
    # Shell scripts
    ".sh": _create_config("langchain", "bash"),
    ".bash": _create_config("langchain", "bash"),
    ".zsh": _create_config("langchain", "bash"),
    
    # Other text files
    ".txt": _create_config("langchain"),
    ".log": _create_config("langchain"),
    ".csv": _create_config("langchain"),
}
