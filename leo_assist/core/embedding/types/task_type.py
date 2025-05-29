"""Task type enumeration for different embedding tasks.

This module defines the TaskType enum which represents different types of tasks
that can be performed with embedding models.
"""
from enum import Enum, auto
from typing import Optional, Union, Type, TypeVar

T = TypeVar('T', bound='TaskType')

class TaskType(str, Enum):
    """Supported task types for embedding models.
    
    Attributes:
        SEMANTIC_SIMILARITY: For embedding documents for semantic similarity
        RETRIEVAL_DOCUMENT: For embedding documents for retrieval
        CODE_RETRIEVAL_QUERY: For embedding code documents for retrieval
        CLASSIFICATION: For embedding text for classification
        CLUSTERING: For embedding text for clustering
    """
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    CODE_RETRIEVAL_QUERY = "CODE_RETRIEVAL_QUERY"
    CLASSIFICATION = "CLASSIFICATION"
    CLUSTERING = "CLUSTERING"
    
    @classmethod
    def _missing_(cls: Type[T], value: object) -> Optional[T]:
        """Handle case-insensitive string lookups."""
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        return None
    
    @classmethod
    def from_str(cls: Type[T], value: str) -> T:
        """Create a TaskType from a string value.
        
        Args:
            value: String value to convert to TaskType
            
        Returns:
            The corresponding TaskType enum member
            
        Raises:
            ValueError: If the string doesn't match any TaskType
        """
        try:
            return cls(value)
        except ValueError as e:
            raise ValueError(
                f"'{value}' is not a valid {cls.__name__}. "
                f"Must be one of: {[t.value for t in cls]}"
            ) from e

# __all__ = ['TaskType']
