"""ChromaDB vector store for semantic search."""

import re
import logging
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import Optional
from pathlib import Path
from app.config import get_settings
from app.services.gemini_service import GeminiService

logging.getLogger("chromadb").setLevel(logging.ERROR) # fix chromadb logging

class VectorStore:
    """Vector database for storing and searching course embeddings."""
    
    COLLECTION_NAME = "courses"
    
    def __init__(self):
        settings = get_settings()
        persist_dir = Path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "UVA course information"}
        )
        
        self.gemini_service = GeminiService()
    
    def add_documents(
        self,
        documents: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of text documents
            metadatas: Optional metadata for each document
            ids: Optional IDs for each document
        """
        if not documents:
            return
        
        # Generate IDs if not provided
        if ids is None:
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]
        
        # Generate embeddings
        embeddings = self.gemini_service.get_embeddings_batch(documents)
        
        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas or [{}] * len(documents),
            ids=ids,
        )
    
    def _extract_course_number(self, query: str) -> Optional[str]:
        """Extract course number from query if present.
        
        Args:
            query: Search query
            
        Returns:
            Course number string or None
        """
        # Match patterns like "CS 4774", "CS4774", "4774"
        patterns = [
            r'(?:CS|cs)\s*(\d{4})',  # CS 4774 or CS4774
            r'\b(\d{4})\b',  # Just a 4-digit number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1)
        return None
    
    def search(
        self,
        query: str,
        n_results: int = 5,
    ) -> dict:
        """Search for similar documents with hybrid approach.
        
        First checks for specific course number matches, then uses semantic search.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Dictionary with documents, metadatas, and distances
        """
        documents = []
        metadatas = []
        distances = []
        
        # Check if query mentions a specific course number
        course_num = self._extract_course_number(query)
        
        if course_num:
            # First, get exact course matches by metadata
            try:
                exact_results = self.collection.get(
                    where={"catalog_number": course_num},
                    include=["documents", "metadatas"],
                )
                
                if exact_results['ids']:
                    # Add exact matches first with distance 0
                    for doc, meta in zip(exact_results['documents'], exact_results['metadatas']):
                        if doc not in documents:  # Avoid duplicates
                            documents.append(doc)
                            metadatas.append(meta)
                            distances.append(0.0)  # Perfect match
            except Exception:
                pass  # If metadata search fails, fall back to semantic
        
        # Fill remaining slots with semantic search
        remaining = n_results - len(documents)
        if remaining > 0:
            query_embedding = self.gemini_service.get_query_embedding(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=remaining + len(documents),  # Get extra to account for duplicates
                include=["documents", "metadatas", "distances"],
            )
            
            # Add semantic results (avoiding duplicates)
            if results["documents"] and results["documents"][0]:
                for doc, meta, dist in zip(
                    results["documents"][0], 
                    results["metadatas"][0], 
                    results["distances"][0]
                ):
                    if doc not in documents and len(documents) < n_results:
                        documents.append(doc)
                        metadatas.append(meta)
                        distances.append(dist)
        
        return {
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
        }
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "UVA course information"}
        )
    
    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()

