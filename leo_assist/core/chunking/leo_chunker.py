"""Tree-sitter based chunker for Leo programming language.

This module provides a chunker that uses Tree-sitter for accurate parsing of Leo code.
If Tree-sitter is not available, it falls back to a regex-based chunker that looks
for Leo-specific patterns.
"""
import os
import re
import warnings
import uuid
from typing import List, Dict, Any, Optional

# Try to import Tree-sitter, but make it optional
try:
    from tree_sitter import Language, Parser, Node
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    warnings.warn(
        "Tree-sitter is not installed. Falling back to regex-based Leo chunker. "
        "Install with: pip install tree-sitter"
    )

from typing import Optional
from leo_assist.core.chunking.base import Chunk, Chunks, Chunker, LangChainChunker
from leo_assist.utils.logger import logger

class TreeSitterLeoChunker(Chunker):
    """Chunker for Leo language using Tree-sitter for accurate parsing.
    
    This chunker uses the Tree-sitter parsing library to understand the structure
    of Leo code, allowing it to create chunks that respect the semantic boundaries
    of the code (functions, structs, programs, etc.).
    
    If Tree-sitter is not available or fails to parse the code, it will fall back
    to a regex-based chunker that looks for Leo-specific patterns.
    
    Example:
        ```python
        chunker = TreeSitterLeoChunker()
        chunks = chunker.chunk(leo_code, {"source": "program.leo"})
        ```
    """
    
    def __init__(self, language: Optional[str] = "leo", chunk_size: int = 1000, overlap: int = 100):
        """Initialize the Tree-sitter based Leo chunker.
        
        Args:
            language: The programming language (defaults to 'leo')
            chunk_size: Target size for chunks
            overlap: Number of characters to overlap between chunks
        """
        super().__init__(language=language, chunk_size=chunk_size, overlap=overlap)
        self.parser = None
        
        if TREE_SITTER_AVAILABLE:
            self._initialize_parser()
    
    def _initialize_parser(self) -> None:
        """Initialize the Tree-sitter parser for Leo."""
        try:
            # Try to load the language from the compiled library
            from tree_sitter_languages import get_language, get_parser
            self.language = get_language("leo")
            self.parser = get_parser("leo")
        except (ImportError, Exception) as e:
            logger.warning(
                f"Failed to load Tree-sitter Leo parser: {str(e)}. "
                "Falling back to regex-based chunking."
            )
            self.language = None
            self.parser = None
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> Chunks:
        """Chunk Leo code using Tree-sitter for accurate parsing.
        
        Args:
            text: The Leo code to be chunked
            metadata: Metadata to associate with each chunk
            
        Returns:
            Chunks: List of Chunk objects with preserved Leo structure
        """
        # If Tree-sitter is not available or failed to initialize, fall back to regex
        if not self.parser or not self.language:
            return self._fallback_chunk(text, metadata)
            
        try:
            # Parse the code
            tree = self.parser.parse(bytes(text, 'utf-8'))
            root_node = tree.root_node
            
            # Extract top-level nodes (program, struct, function, etc.)
            chunks = Chunks(chunks=[])
            parent_id = metadata.get("parent_id", "root")
            
            # Process each top-level declaration
            for i, node in enumerate(self._get_top_level_nodes(root_node)):
                node_type = node.type
                node_text = text[node.start_byte:node.end_byte].decode('utf-8')
                
                # Get the name of the declaration if available
                name_node = self._get_name_node(node)
                name = name_node.text.decode('utf-8') if name_node else f"unnamed_{i}"
                
                # Create chunk metadata
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_type': f"leo_{node_type}",
                    'node_type': node_type,
                    'node_name': name,
                    'is_leo_structure': True,
                    'start_line': node.start_point[0],
                    'end_line': node.end_point[0]
                })
                
                chunk = Chunk(
                    id=f"{parent_id}_{node_type}_{name}_{i}",
                    content=node_text,
                    metadata=chunk_metadata,
                    parent_id=parent_id,
                    chunk_index=i,
                    is_complete=True,
                    chunk_type=f"leo_{node_type}"
                )
                chunks.append(chunk)
                
            return chunks
            
        except Exception as e:
            logger.warning(f"Error parsing Leo code with Tree-sitter: {str(e)}. Falling back to regex chunker.")
            return self._fallback_chunk(text, metadata)
    
    def _get_top_level_nodes(self, root_node):
        """Get top-level declaration nodes from the AST."""
        top_level_types = {
            "program_declaration",
            "struct_declaration",
            "record_declaration",
            "mapping_declaration",
            "function_definition",
            "transition_definition",
            "import_declaration",
            "interface_declaration",
            "program_definition"
        }
        
        # Get all nodes that are direct children of the root and have a type we're interested in
        return [
            node for node in root_node.children
            if node.type in top_level_types
        ]
    
    def _get_name_node(self, node):
        """Get the name node from a declaration node."""
        # Different node types have name nodes in different positions
        if node.type in ("function_definition", "transition_definition"):
            # Function/transition name is the first identifier after 'function'/'transition' keyword
            for child in node.children:
                if child.type == "identifier":
                    return child
        elif node.type in ("struct_declaration", "record_declaration", "mapping_declaration"):
            # Name is the first identifier after the declaration keyword
            for child in node.children:
                if child.type == "type_identifier":
                    return child
        return None
    
    def _fallback_chunk(self, text: str, metadata: Dict[str, Any]) -> Chunks:
        """Fallback chunking method when Tree-sitter is not available.
        
        Args:
            text: The Leo code to be chunked
            metadata: Metadata to associate with each chunk
            
        Returns:
            Chunks: List of Chunk objects with preserved Leo structure
        """
        # Use the dedicated LeoRegexChunker as fallback
        return LeoRegexChunker(
            chunk_size=self.chunk_size,
            overlap=self.overlap
        ).chunk(text, metadata)



    
class LeoRegexChunker(Chunker):
    """Regex-based chunker specifically for Leo programming language.
    
    This chunker uses regex patterns to identify Leo language constructs
    and create meaningful chunks. It's designed to be used as a fallback
    when Tree-sitter is not available.
    
    Example:
        ```python
        chunker = LeoRegexChunker()
        chunks = chunker.chunk(leo_code, {"source": "program.leo"})
        ```
    """
    
    def __init__(self, language: Optional[str] = 'leo', chunk_size: int = 1000, overlap: int = 100):
        """Initialize the regex-based Leo chunker.
        
        Args:
            language: The programming language (defaults to 'leo')
            chunk_size: Target size for chunks
            overlap: Number of characters to overlap between chunks
        """
        super().__init__(language=language, chunk_size=chunk_size, overlap=overlap)
        self._init_patterns()
    
    def _init_patterns(self) -> None:
        """Initialize the regex patterns for Leo code structures."""
        # Patterns for different Leo constructs
        self.program_regex = re.compile(r'program\s+([a-zA-Z0-9_\.]+)\s*{')
        self.import_regex = re.compile(r'import\s+(["\'])(.+?)\1\s*;')
        self.function_regex = re.compile(r'function\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)(?:\s*->\s*([a-zA-Z0-9_<>]+))?\s*{')
        self.async_function_regex = re.compile(r'async function\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)(?:\s*->\s*([a-zA-Z0-9_<>]+))?\s*{')
        self.transition_regex = re.compile(r'transition\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)(?:\s*->\s*([a-zA-Z0-9_<>]+))?\s*{')
        self.async_transition_regex = re.compile(r'async transition\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)(?:\s*->\s*([a-zA-Z0-9_<>]+))?\s*{')
        self.finalize_regex = re.compile(r'finalize\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)\s*{')
        self.struct_regex = re.compile(r'struct\s+([a-zA-Z0-9_]+)\s*{([^}]*)}')
        self.record_regex = re.compile(r'record\s+([a-zA-Z0-9_]+)\s*{([^}]*)}')
        self.mapping_regex = re.compile(r'mapping\s+([a-zA-Z0-9_]+)\s*:\s*([a-zA-Z0-9_<>]+)\s*=>\s*([a-zA-Z0-9_<>]+)\s*;')
        self.const_regex = re.compile(r'const\s+([a-zA-Z0-9_]+)\s*:\s*([a-zA-Z0-9_<>]+)\s*=\s*(.+?)\s*;')
    
    def _extract_function_like(self, content: str, match, has_finalize_check: bool = False) -> Dict[str, Any]:
        """Helper method to extract function-like blocks (functions, transitions, finalize functions).
        
        Args:
            content: The source code content
            match: The regex match object
            has_finalize_check: Whether to check for finalize block in the body
            
        Returns:
            dict: The extracted function/transition data
        """
        logger.info(f"Extracting function/transition: {match}")
        name = match.group(1)
        params_str = match.group(2)
        return_type = match.group(3) if len(match.groups()) > 2 and match.group(3) else ''
        
        # Get line numbers
        start_line = content[:match.start()].count('\n') + 1
        
        # Find the end of the block
        block_start = match.end()
        brace_count = 1
        block_end = block_start
        while brace_count > 0 and block_end < len(content):
            if content[block_end] == '{':
                brace_count += 1
            elif content[block_end] == '}':
                brace_count -= 1
            block_end += 1
        
        end_line = content[:block_end].count('\n') + 1
        
        # Extract body
        body = content[match.start():block_end]
        
        # Check for finalize block if needed
        has_finalize = False
        if has_finalize_check:
            has_finalize = self.finalize_regex.search(body) is not None
        
        # Parse parameters
        params = []
        if params_str:
            param_parts = params_str.split(',')
            for part in param_parts:
                part = part.strip()
                if part:
                    if ':' in part:
                        param_name, param_type = part.split(':', 1)
                        params.append({
                            'name': param_name.strip(),
                            'type': param_type.strip()
                        })
                    else:
                        params.append({
                            'name': part,
                            'type': ''
                        })
        
        # Create the result dictionary
        result = {
            'name': name,
            'params': params,
            'start_line': start_line,
            'end_line': end_line,
            'body': body
        }
        
        # Add return type if present
        if return_type:
            result['return_type'] = return_type
        
        # Add finalize flag if needed
        if has_finalize_check:
            result['has_finalize'] = has_finalize
        
        return result

    def _extract_information_regex(self, content: str) -> Dict[str, Any]:
        """
        Extract information from Leo code using regex.
        
        Args:
            content: The Leo code content
            
        Returns:
            Dictionary containing extracted information
        """
        # Initialize result dictionary
        result = {
            'imports': [],
            'program': None,
            'functions': [],
            'async_functions': [],
            'transitions': [],
            'async_transitions': [],
            'finalize_functions': [],
            'structs': [],
            'records': [],
            'mappings': [],
            'constants': [],
            'relationships': []
        }
        
        # Extract program name
        program_match = self.program_regex.search(content)
        if program_match:
            result['program'] = program_match.group(1)
        
        # Extract imports
        for match in self.import_regex.finditer(content):
            import_path = match.group(2)
            result['imports'].append(import_path)
            
            # Add to relationships
            result['relationships'].append({
                'type': 'IMPORT',
                'source': result.get('program', ''),
                'target': import_path
            })
        
        # Extract functions
        for match in self.function_regex.finditer(content):
            func_data = self._extract_function_like(content, match)
            result['functions'].append(func_data)
        
        # Extract async functions
        for match in self.async_function_regex.finditer(content):
            func_data = self._extract_function_like(content, match)
            result['async_functions'].append(func_data)
        
        # Extract transitions
        for match in self.transition_regex.finditer(content):
            transition = self._extract_function_like(content, match, has_finalize_check=True)
            result['transitions'].append(transition)
        
        # Extract async transitions
        for match in self.async_transition_regex.finditer(content):
            transition = self._extract_function_like(content, match, has_finalize_check=True)
            result['async_transitions'].append(transition)
        
        # Extract finalize functions
        for match in self.finalize_regex.finditer(content):
            finalize = self._extract_function_like(content, match)
            finalize['name'] = f"{finalize['name']}_finalize"
            
            # Find the parent transition
            finalize['parent_transition'] = None
            for transition in result['transitions']:
                if transition.get('has_finalize', False) and finalize['name'].replace('_finalize', '') == transition['name']:
                    finalize['parent_transition'] = transition['name']
                    break
            
            result['finalize_functions'].append(finalize)
        
        # Extract structs
        for match in self.struct_regex.finditer(content):
            name = match.group(1)
            fields_str = match.group(2)
            
            # Get line numbers
            start_line = content[:match.start()].count('\n') + 1
            end_line = content[:match.end()].count('\n') + 1
            
            # Extract struct body
            body = content[match.start():match.end()]
            
            # Parse fields
            fields = []
            if fields_str:
                field_parts = fields_str.split(';')
                for part in field_parts:
                    part = part.strip()
                    if part:
                        # Handle field format: name: type
                        if ':' in part:
                            field_name, field_type = part.split(':', 1)
                            fields.append({
                                'name': field_name.strip(),
                                'type': field_type.strip()
                            })
                        else:
                            # If no type is specified
                            fields.append({
                                'name': part,
                                'type': ''
                            })
            
            result['structs'].append({
                'name': name,
                'fields': fields,
                'start_line': start_line,
                'end_line': end_line,
                'body': body
            })
        
        # Extract records
        for match in self.record_regex.finditer(content):
            name = match.group(1)
            fields_str = match.group(2)
            
            # Get line numbers
            start_line = content[:match.start()].count('\n') + 1
            end_line = content[:match.end()].count('\n') + 1
            
            # Extract record body
            body = content[match.start():match.end()]
            
            # Parse fields
            fields = []
            if fields_str:
                field_parts = fields_str.split(';')
                for part in field_parts:
                    part = part.strip()
                    if part:
                        # Handle field format: name: type
                        if ':' in part:
                            field_name, field_type = part.split(':', 1)
                            fields.append({
                                'name': field_name.strip(),
                                'type': field_type.strip()
                            })
                        else:
                            # If no type is specified
                            fields.append({
                                'name': part,
                                'type': ''
                            })
            
            result['records'].append({
                'name': name,
                'fields': fields,
                'start_line': start_line,
                'end_line': end_line,
                'body': body
            })
        
        # Extract mappings
        for match in self.mapping_regex.finditer(content):
            name = match.group(1)
            key_type = match.group(2)
            value_type = match.group(3)
            
            # Get line numbers
            start_line = content[:match.start()].count('\n') + 1
            end_line = content[:match.end()].count('\n') + 1
            
            # Extract mapping body
            body = content[match.start():match.end()]
            
            result['mappings'].append({
                'name': name,
                'key_type': key_type,
                'value_type': value_type,
                'start_line': start_line,
                'end_line': end_line,
                'body': body
            })
        
        # Extract constants
        for match in self.const_regex.finditer(content):
            name = match.group(1)
            const_type = match.group(2)
            value = match.group(3)
            
            # Get line numbers
            start_line = content[:match.start()].count('\n') + 1
            end_line = content[:match.end()].count('\n') + 1
            
            # Extract constant body
            body = content[match.start():match.end()]
            
            result['constants'].append({
                'name': name,
                'type': const_type,
                'value': value,
                'start_line': start_line,
                'end_line': end_line,
                'body': body
            })
        
        return result

    def chunk(self, text: str, metadata: Dict[str, Any]) -> list[Chunk]:
        """Chunk Leo code using regex patterns.
        
        Args:
            text: The Leo code to be chunked
            metadata: Metadata to associate with each chunk
            
        Returns:
            Chunks: List of Chunk objects with preserved Leo structure
        """
        logger.info(f"Chunking Leo code with regex patterns: {text}")

        parsed_info = self._extract_information_regex(text)

        chunks = []
        parent_id = metadata.get("parent_id", str(uuid.uuid4()))

        # Add program declaration as a chunk
        if parsed_info.get('program'):
            chunks.append(Chunk(
                id=f"{parent_id}_program_{parsed_info['program']}",
                content=f"program {parsed_info['program']}",
                metadata=metadata,
                start_line=1,
                end_line=1,
                parent_id=parent_id,
                chunk_index=len(chunks),
                is_complete=True,
                chunk_type="program"
            ))
        
        # Add functions as chunks
        for func in parsed_info.get('functions', []):
            chunks.append(Chunk(
                id=f"{parent_id}_function_{func['name']}",
                content=func['body'],
                metadata=metadata,
                start_line=func['start_line'],
                end_line=func['end_line'],
                parent_id=parent_id,
                chunk_index=len(chunks),
                is_complete=True,
                chunk_type="function"
            ))
            
        # Add transitions as chunks
        for transition in parsed_info.get('transitions', []):
            chunks.append(Chunk(
                id=f"{parent_id}_transition_{transition['name']}",
                content=transition['body'],
                metadata=metadata,
                start_line=transition['start_line'],
                end_line=transition['end_line'],
                parent_id=parent_id,
                chunk_index=len(chunks),
                is_complete=True,
                chunk_type="transition"
            ))
        
        # Add finalize functions as chunks
        for finalize in parsed_info.get('finalize_functions', []):
            chunks.append(Chunk(
                id=f"{parent_id}_finalize_{finalize['name']}",
                content=finalize['body'],
                metadata=metadata,
                start_line=finalize['start_line'],
                end_line=finalize['end_line'],
                parent_id=parent_id,
                chunk_index=len(chunks),
                is_complete=True,
                chunk_type="finalize"
            ))
        
        # Add structs as chunks
        for struct in parsed_info.get('structs', []):
            chunks.append(Chunk(
                id=f"{parent_id}_struct_{struct['name']}",
                content=struct['body'],
                metadata=metadata,
                start_line=struct['start_line'],
                end_line=struct['end_line'],
                parent_id=parent_id,
                chunk_index=len(chunks),
                is_complete=True,
                chunk_type="struct"
            ))
        
        # Add records as chunks
        for record in parsed_info.get('records', []):
            chunks.append(Chunk(
                id=f"{parent_id}_record_{record['name']}",
                content=record['body'],
                metadata=metadata,
                start_line=record['start_line'],
                end_line=record['end_line'],
                parent_id=parent_id,
                chunk_index=len(chunks),
                is_complete=True,
                chunk_type="record"
            ))
        
        # Add mappings as chunks
        for mapping in parsed_info.get('mappings', []):
            chunks.append(Chunk(
                id=f"{parent_id}_mapping_{mapping['name']}",
                content=mapping['body'],
                metadata=metadata,
                start_line=mapping['start_line'],
                end_line=mapping['end_line'],
                parent_id=parent_id,
                chunk_index=len(chunks),
                is_complete=True,
                chunk_type="mapping"
            ))
        
        # Add constants as chunks
        for const in parsed_info.get('constants', []):
            chunks.append(Chunk(
                id=f"{parent_id}_constant_{const['name']}",
                content=const['body'],
                metadata=metadata,
                start_line=const['start_line'],
                end_line=const['end_line'],
                parent_id=parent_id,
                chunk_index=len(chunks),
                is_complete=True,
                chunk_type="constant"
            ))
        
        chunks = sorted(chunks, key=lambda x: x.start_line)
        logger.info(f"Chunked Leo code with regex patterns: {Chunks(chunks=chunks).model_dump_json()}")

        # If no Leo structures found, fall back to LangChain chunker
        if not chunks:
            return LangChainChunker(
                chunk_size=self.chunk_size,
                overlap=self.overlap
            ).chunk(text, metadata)
            
        return chunks
