"""
Embedding module using local sentence-transformers.

This module handles converting text chunks into dense vector embeddings
that can be used for semantic similarity search.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union


class LocalEmbedder:
    """
    Local embedding model using sentence-transformers.
    
    Design Decision: Using 'all-MiniLM-L6-v2' because:
    - Small model size (~80MB) - fast and efficient for local use
    - 384-dimensional embeddings - good balance of quality and speed
    - Trained on large corpus for general-purpose semantic similarity
    - No API costs or rate limits
    - Works offline
    
    Alternative considered: 'all-mpnet-base-v2' (higher quality but 2x slower)
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Convert text to embedding vector(s).
        
        Args:
            text: Single string or list of strings to embed
            
        Returns:
            Numpy array of shape (embedding_dim,) for single text
            or (n, embedding_dim) for list of texts
        """
        # Convert single string to list for uniform processing
        if isinstance(text, str):
            text = [text]
            return_single = True
        else:
            return_single = False
            
        # Generate embeddings
        # normalize_embeddings=True ensures vectors are unit length
        # This makes cosine similarity equivalent to dot product
        embeddings = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=len(text) > 10  # Show progress for large batches
        )
        
        if return_single:
            return embeddings[0]
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a search query.
        
        Args:
            query: Search query string
            
        Returns:
            Normalized embedding vector
        """
        return self.embed_text(query)
    
    def get_embedding_dimension(self) -> int:
        """Return the dimension of embedding vectors."""
        return self.embedding_dim
