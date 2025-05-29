"""Manager for handling multiple chunkers based on file types."""
import os
from typing import Any, Dict, Optional, TypeVar

from leo_assist.core.chunking.base import Chunker, ChunkerFactory, Chunk
from leo_assist.core.chunking.config import DEFAULT_CONFIGS, ChunkerConfig

T = TypeVar('T', bound=Chunker)

class ChunkerManager:
    """Manages multiple chunkers based on file extensions."""
    
    # Default chunker configurations for different file types
    DEFAULT_CONFIGS: Dict[str, ChunkerConfig] = DEFAULT_CONFIGS
    
    def __init__(self, default_config: Optional[ChunkerConfig] = None):
        """Initialize the chunker manager.
        
        Args:
            default_config: Default chunker configuration. If None, uses the default config
                from DEFAULT_CONFIGS.
        """
        self.chunkers: Dict[str, Chunker] = {}
        self.default_config: ChunkerConfig = default_config or self.DEFAULT_CONFIGS['default'].model_copy()
        # Initialize default chunker
        self._get_chunker_for_extension('default')
    
    def get_chunker_for_file(self, file_path: str) -> Chunker:
        """Get the appropriate chunker for a file based on its extension.
        
        Args:
            file_path: Path to the file. The file extension is used to determine
                the appropriate chunker.
                
        Returns:
            An instance of Chunker appropriate for the file type.
            
        Example:
            ```python
            chunker = manager.get_chunker_for_file("example.leo")
            chunks = chunker.chunk(code, {})
            ```
        """
        ext = os.path.splitext(file_path)[1].lower()
        return self._get_chunker_for_extension(ext)
    
    def _get_chunker_for_extension(self, ext: str) -> Chunker:
        """Get or create a chunker for the given file extension.
        
        Args:
            ext: File extension (with dot, e.g., '.py')
            
        Returns:
            An instance of Chunker
            
        Raises:
            ValueError: If no configuration is found for the extension
        """
        # Use the chunker from cache if available
        if ext in self.chunkers:
            return self.chunkers[ext]
        
        # Get config for this extension or use default
        config = self.DEFAULT_CONFIGS.get(ext, self.default_config).model_copy()
        
        # Create and cache the chunker
        self.chunkers[ext] = ChunkerFactory.create_chunker(config)
        return self.chunkers[ext]
    
    def chunk_document(self, text: str, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> list[Chunk]:
        """Chunk a document using the appropriate chunker for its file type.
        
        Args:
            text: The document text to chunk
            file_path: Path to the file (used to determine chunker)
            metadata: Optional metadata for the chunks
            
        Returns:
            List of Chunk objects
            
        Raises:
            ValueError: If no appropriate chunker can be found for the file type
        """
        chunker = self.get_chunker_for_file(file_path)
        return chunker.chunk(text, metadata or {})
