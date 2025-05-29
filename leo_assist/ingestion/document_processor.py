"""Document processing utilities.

This module provides utility functions for processing documents from various sources.
"""
from pathlib import Path
from typing import List, Union, Optional

from leo_assist.core.vector_store import Document


def process_single_file(file_path: Union[str, Path]) -> List[Document]:
    """Process a single file and return a list of Document objects.
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        List containing a single Document object for the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        UnicodeDecodeError: If the file can't be read as UTF-8 text
    """
    input_path = Path(file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
        
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
            
    return [
        Document(
            content=content,
            metadata={
                'source': str(input_path.absolute()),
                'filename': input_path.name,
                'filetype': input_path.suffix.lstrip('.')
            }
        )
    ]


def process_directory(directory_path: Union[str, Path], 
                     file_extensions: Optional[List[str]] = None) -> List[Document]:
    """Process all files in a directory and return a list of Document objects.
    
    Args:
        directory_path: Path to the directory to process
        file_extensions: Optional list of file extensions to include (e.g., ['.txt', '.md'])
                        If None, all files are included.
        
    Returns:
        List of Document objects, one for each file in the directory
        
    Raises:
        NotADirectoryError: If the path is not a directory
    """
    input_path = Path(directory_path)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_path}")
        
    documents = []
    
    for file_path in input_path.rglob('*'):
        if not file_path.is_file() or file_path.name.startswith('.'):
            continue
            
        if file_extensions and file_path.suffix.lower() not in file_extensions:
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            documents.append(
                Document(
                    content=content,
                    metadata={
                        'source': str(file_path.absolute()),
                        'filename': file_path.name,
                        'filetype': file_path.suffix.lstrip('.')
                    }
                )
            )
        except (UnicodeDecodeError, PermissionError) as e:
            # Skip files that can't be read as text or don't have read permissions
            continue
    
    return documents
