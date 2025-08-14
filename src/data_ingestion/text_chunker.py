"""
Text Chunking for RAG Pipeline

Implements multiple text chunking strategies including:
- Fixed-size chunking with overlap
- Semantic chunking based on content
- Rule-based chunking (paragraphs, sentences)
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import tiktoken

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a text chunk with metadata."""
    content: str
    chunk_id: str
    start_index: int
    end_index: int
    metadata: Dict[str, Any]


class TextChunker:
    """Advanced text chunking with multiple strategies."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n",
        preserve_paragraphs: bool = True
    ):
        """Initialize text chunker with configuration."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.preserve_paragraphs = preserve_paragraphs
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def chunk_by_tokens(self, text: str) -> List[TextChunk]:
        """Chunk text by token count with overlap."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        chunk_id = 0
        
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            if chunk_text.strip():
                chunk = TextChunk(
                    content=chunk_text.strip(),
                    chunk_id=f"chunk_{chunk_id}",
                    start_index=i,
                    end_index=min(i + self.chunk_size, len(tokens)),
                    metadata={
                        'chunk_type': 'token_based',
                        'token_count': len(chunk_tokens),
                        'overlap_tokens': self.chunk_overlap
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
        
        return chunks
    
    def chunk_by_sentences(self, text: str) -> List[TextChunk]:
        """Chunk text by sentences while respecting chunk size."""
        # Split by sentences (basic implementation)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        start_index = 0
        
        for i, sentence in enumerate(sentences):
            sentence_with_punctuation = sentence + "."
            
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence_with_punctuation) > self.chunk_size:
                if current_chunk:
                    # Create chunk from accumulated sentences
                    chunk = TextChunk(
                        content=current_chunk.strip(),
                        chunk_id=f"chunk_{chunk_id}",
                        start_index=start_index,
                        end_index=start_index + len(current_chunk),
                        metadata={
                            'chunk_type': 'sentence_based',
                            'num_sentences': current_chunk.count('.'),
                            'char_count': len(current_chunk)
                        }
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                # Start new chunk
                current_chunk = sentence_with_punctuation
                start_index = text.find(sentence, start_index)
            else:
                current_chunk += " " + sentence_with_punctuation
        
        # Add final chunk
        if current_chunk:
            chunk = TextChunk(
                content=current_chunk.strip(),
                chunk_id=f"chunk_{chunk_id}",
                start_index=start_index,
                end_index=start_index + len(current_chunk),
                metadata={
                    'chunk_type': 'sentence_based',
                    'num_sentences': current_chunk.count('.'),
                    'char_count': len(current_chunk)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[TextChunk]:
        """Chunk text by paragraphs while respecting chunk size."""
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        start_index = 0
        
        for i, paragraph in enumerate(paragraphs):
            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                if current_chunk:
                    # Create chunk from accumulated paragraphs
                    chunk = TextChunk(
                        content=current_chunk.strip(),
                        chunk_id=f"chunk_{chunk_id}",
                        start_index=start_index,
                        end_index=start_index + len(current_chunk),
                        metadata={
                            'chunk_type': 'paragraph_based',
                            'num_paragraphs': current_chunk.count('\n\n') + 1,
                            'char_count': len(current_chunk)
                        }
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                # Start new chunk
                current_chunk = paragraph
                start_index = text.find(paragraph, start_index)
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk:
            chunk = TextChunk(
                content=current_chunk.strip(),
                chunk_id=f"chunk_{chunk_id}",
                start_index=start_index,
                end_index=start_index + len(current_chunk),
                metadata={
                    'chunk_type': 'paragraph_based',
                    'num_paragraphs': current_chunk.count('\n\n') + 1,
                    'char_count': len(current_chunk)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def smart_chunk(self, text: str, strategy: str = "hybrid") -> List[TextChunk]:
        """Smart chunking that combines multiple strategies."""
        if strategy == "hybrid":
            # Try paragraph-based first, fall back to sentence-based, then token-based
            try:
                chunks = self.chunk_by_paragraphs(text)
                if len(chunks) > 1:
                    return chunks
            except Exception as e:
                logger.warning(f"Paragraph chunking failed: {e}")
            
            try:
                chunks = self.chunk_by_sentences(text)
                if len(chunks) > 1:
                    return chunks
            except Exception as e:
                logger.warning(f"Sentence chunking failed: {e}")
            
            # Fall back to token-based chunking
            return self.chunk_by_tokens(text)
        
        elif strategy == "paragraphs":
            return self.chunk_by_paragraphs(text)
        elif strategy == "sentences":
            return self.chunk_by_sentences(text)
        elif strategy == "tokens":
            return self.chunk_by_tokens(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def merge_small_chunks(self, chunks: List[TextChunk], min_size: int = 100) -> List[TextChunk]:
        """Merge chunks that are too small."""
        if not chunks:
            return chunks
        
        merged_chunks = []
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            if len(current_chunk.content) < min_size:
                # Merge with next chunk
                current_chunk.content += "\n\n" + next_chunk.content
                current_chunk.end_index = next_chunk.end_index
                current_chunk.metadata['merged_chunks'] = current_chunk.metadata.get('merged_chunks', 1) + 1
            else:
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk
        
        merged_chunks.append(current_chunk)
        return merged_chunks
    
    def get_chunk_statistics(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Get statistics about the generated chunks."""
        if not chunks:
            return {}
        
        total_chunks = len(chunks)
        total_chars = sum(len(chunk.content) for chunk in chunks)
        avg_chunk_size = total_chars / total_chunks
        
        chunk_types = {}
        for chunk in chunks:
            chunk_type = chunk.metadata.get('chunk_type', 'unknown')
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        return {
            'total_chunks': total_chunks,
            'total_characters': total_chars,
            'average_chunk_size': round(avg_chunk_size, 2),
            'chunk_type_distribution': chunk_types,
            'size_range': {
                'min': min(len(chunk.content) for chunk in chunks),
                'max': max(len(chunk.content) for chunk in chunks)
            }
        }
