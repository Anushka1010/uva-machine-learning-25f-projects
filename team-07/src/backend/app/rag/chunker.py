"""
Text chunking module for RAG pipeline.

This module handles splitting documents into overlapping chunks for better
context preservation during retrieval. Overlapping chunks ensure that 
important information at chunk boundaries isn't lost.
"""

from typing import List, Dict
import re


class TextChunker:
    """
    Splits text into overlapping chunks so retrieval keeps nearby context.
    
    Notes:
    - Chunk size: about 500 tokens (~375 words) to stay within model limits while keeping detail
    - Overlap: about 100 tokens (~75 words) to carry context across boundaries
    - Word-based splitting avoids chopping words in half
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        """
        Initialize chunker with configurable parameters.
        
        Args:
            chunk_size: Target tokens per chunk (approximated as ~0.75 words per token)
            overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Approximate token count: ~0.75 words per token (English text)
        self.words_per_chunk = int(chunk_size * 0.75)
        self.words_overlap = int(overlap * 0.75)
        
    def chunk_text(self, text: str, filename: str, file_type: str) -> List[Dict]:
        """
        Split text into overlapping chunks with metadata.
        
        Args:
            text: Raw text to chunk
            filename: Source filename
            file_type: Type of source file (pdf, txt, md, docx)
            
        Returns:
            List of dicts containing chunk text and metadata
        """
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Split into words
        words = text.split()
        
        if len(words) <= self.words_per_chunk:
            # Document is smaller than one chunk
            return [{
                'text': text,
                'filename': filename,
                'file_type': file_type,
                'chunk_index': 0,
                'total_chunks': 1
            }]
        
        chunks = []
        chunk_index = 0
        start_idx = 0
        
        while start_idx < len(words):
            # Extract chunk
            end_idx = start_idx + self.words_per_chunk
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'filename': filename,
                'file_type': file_type,
                'chunk_index': chunk_index,
                'total_chunks': None  # Will update after
            })
            
            # Move to next chunk with overlap
            start_idx += (self.words_per_chunk - self.words_overlap)
            chunk_index += 1
            
            # Break if we've covered all words
            if end_idx >= len(words):
                break
        
        # Update total chunks count
        total = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total
            
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        - Remove excessive whitespace
        - Normalize line breaks
        - Remove special characters that might interfere with chunking
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text