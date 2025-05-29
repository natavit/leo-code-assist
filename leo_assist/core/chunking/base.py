"""Base classes for chunking strategies.

This module provides the core chunking functionality that can be used by both
indexing and retrieval components.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
from pydantic import BaseModel
from leo_assist.core.chunking.config import ChunkerConfig
from leo_assist.utils.logger import logger

class Chunk(BaseModel):
    """Represents a chunk of text with metadata.
    
    Attributes:
        id: Unique identifier for the chunk
        content: The actual text content of the chunk
        metadata: Dictionary of metadata associated with the chunk
        parent_id: ID of the parent document
        chunk_index: Position of this chunk in the parent document
        is_complete: Whether this chunk represents a complete logical unit
        chunk_type: Type of the chunk (e.g., "function", "class", "leo_struct")
    """
    id: str
    content: str
    metadata: Dict[str, Any]
    start_line: int
    end_line: int
    parent_id: str
    chunk_index: int = 0
    is_complete: bool = False
    chunk_type: str = "text"

class Chunks(BaseModel):
    chunks: List[Chunk]

class Chunker(ABC):
    """Abstract base class for chunkers.
    
    This class provides a common interface for all chunkers, which are responsible
    for splitting input text into smaller, more manageable pieces. The chunking
    process can be customized based on the specific requirements of the application.
    
    Attributes:
        language: Optional language of the content being chunked (e.g., 'python', 'javascript')
        chunk_size: Target size for chunks in characters
        overlap: Number of characters to overlap between chunks
    """
    
    def __init__(
        self,
        language: Optional[str] = None,
        chunk_size: int = 1000,
        overlap: int = 100
    ):
        """Initialize the base chunker with common parameters.
        
        Args:
            language: Optional language of the content being chunked
            chunk_size: Target size for chunks in characters
            overlap: Number of characters to overlap between chunks
        """
        self.language = language.lower() if language else None
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any]) -> list[Chunk]:
        """Chunk the input text into smaller pieces.
        
        This method takes in the input text and associated metadata, and returns
        a list of Chunk objects. The chunking process can be customized based on
        the specific requirements of the application.
        
        Args:
            text: The text to be chunked
            metadata: Metadata associated with the text
            
        Returns:
            Chunks: List of Chunk objects
        """
        pass

class LangChainChunker(Chunker):
    """Chunker that uses LangChain's RecursiveCharacterTextSplitter with language support.
    
    This chunker is designed to handle multiple programming languages by using
    language-specific separators. It will try to split on language-specific
    constructs first, then fall back to more general separators.
    
    Example:
        ```python
        # For Python code
        text_chunker = LangChainChunker(language="python")
        chunks = text_chunker.chunk(python_code)
        ```
    """
    
    def __init__(
        self, 
        language: Optional[str] = None,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        **kwargs
    ):
        """Initialize the LangChain chunker.
        
        Args:
            language: Programming language for language-specific chunking
            chunk_size: Target size for chunks. If None, uses default from config
            overlap: Number of characters to overlap between chunks. If None, uses default from config
            **kwargs: Additional arguments to pass to the base class
        """
        # Get default config for the specified language or use 'default'
        from ..chunking.config import DEFAULT_CONFIGS
        
        # Get config for the specified language or fall back to default
        config = DEFAULT_CONFIGS.get(language, {}) if language else {}
        default_config = DEFAULT_CONFIGS.get("default", {})
        
        # Use provided values or fall back to config
        chunk_size = chunk_size or config.get("chunk_size", default_config.get("chunk_size", 1000))
        overlap = overlap or config.get("overlap", default_config.get("overlap", 200))
        
        super().__init__(language=language, chunk_size=chunk_size, overlap=overlap, **kwargs)
        
        # Initialize the splitter based on the language
        self._init_splitter()
    
    def _init_splitter(self):
        """Initialize the splitter based on the language."""
        try:
            from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
            
            if self.language:
                try:
                    lang_enum = getattr(Language, self.language.upper(), None)
                    if lang_enum is not None:
                        self.splitter = RecursiveCharacterTextSplitter.from_language(
                            language=lang_enum,
                            chunk_size=self.chunk_size,
                            chunk_overlap=self.overlap
                        )
                        return
                    else:
                        logger.warning(f"Unsupported language: {self.language}. Using default text splitter.")
                except (ImportError, AttributeError) as e:
                    logger.warning(
                        f"Failed to initialize language-specific splitter for {self.language}: {str(e)}. "
                        "Falling back to default text splitter."
                    )
            
            # Fall back to default splitter if no language specified or if language-specific splitter failed
            self._init_default_splitter()
            
        except ImportError:
            logger.warning(
                "langchain_text_splitters not available. Using simple text splitting. "
                "Install with: pip install langchain-text-splitters"
            )
            self._init_default_splitter()
    
    def _init_default_splitter(self):
        """Initialize the default text splitter."""
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            length_function=len,
            is_separator_regex=False,
        )
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> list[Chunk]:
        """Split text into chunks using LangChain's splitter.
        
        Args:
            text: The text to be chunked
            metadata: Metadata to associate with each chunk.
                   May contain 'chunk_type' to specify the type of chunk.
            
        Returns:
            Chunks: List of Chunk objects
        """
        logger.info(f"Chunking text with langchain splitter: {text}")
        
        parent_id = str(uuid.uuid4())
        chunks = []
        
        # Add language to metadata if not already present
        chunk_metadata = metadata.copy()
        if self.language and "language" not in chunk_metadata:
            chunk_metadata["language"] = self.language
        
        try:
            # Use LangChain's splitter to create chunks
            split_texts = self.splitter.split_text(text)
            
            for i, chunk_text in enumerate(split_texts):
                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    content=chunk_text,
                    metadata=chunk_metadata.copy(),
                    start_line=chunk_metadata.get("start_line", 0),
                    end_line=chunk_metadata.get("end_line", 0),
                    parent_id=parent_id,
                    chunk_index=i,
                    is_complete=(i == len(split_texts) - 1),  # Mark last chunk as complete
                    chunk_type=chunk_metadata.get("chunk_type", "text")
                )
                chunks.append(chunk)
                
        except Exception as e:
            logger.warning(
                f"Error during chunking with language {self.language}: {str(e)}. "
                "Falling back to simple text splitting."
            )
            # Fall back to default splitter if language-specific splitting fails
            self._init_default_splitter()
            return self.chunk(text, metadata)
            
        return chunks

class ChunkerFactory:
    """Factory for creating chunkers based on configuration.
    
    This class maintains a registry of available chunker types and their corresponding
    factory functions. Chunkers can be registered dynamically at runtime.
    """
    _registry: Dict[str, Any] = {}
    
    @classmethod
    def register_chunker(cls, name: str, chunker_class: type) -> None:
        """Register a chunker class with the factory.
        
        Args:
            name: The name to register the chunker under
            chunker_class: The chunker class to register
        """
        cls._registry[name.lower()] = chunker_class
    
    @classmethod
    def create_chunker(cls, config: ChunkerConfig) -> 'Chunker':
        """Create a chunker based on the configuration.
        
        Args:
            config: Dictionary containing chunker configuration with the following keys:
                - type: The type of chunker ('langchain', 'tree_sitter_leo', etc.)
                - language: The programming language for language-specific chunking
                - chunk_size: Target size for chunks (overrides config if provided)
                - overlap: Number of characters to overlap between chunks (overrides config if provided)
            
        Returns:
            An instance of a Chunker subclass
        """
            
        chunker_type = config.type
        
        # Get default config for the specified language or use 'default'
        from .config import DEFAULT_CONFIGS
        
        # Get language from config or use None
        language = config.language
        
        # Get config for the specified language or fall back to default
        lang_config = DEFAULT_CONFIGS.get(language, {}) if language else {}
        default_config = DEFAULT_CONFIGS.get("default", {})
        
        # Use provided values or fall back to config
        chunk_size = config.chunk_size or lang_config.chunk_size or default_config.chunk_size
        overlap = config.overlap or lang_config.overlap or default_config.overlap
        
        # Try to create a registered chunker
        if chunker_type in cls._registry:
            return cls._registry[chunker_type](
                language=language,
                chunk_size=chunk_size,
                overlap=overlap
            )
            
        raise ValueError(f"Unknown chunker type: {chunker_type}")
