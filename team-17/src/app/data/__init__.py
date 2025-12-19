"""Data management modules.

Structure:
    sources/    - Data fetching APIs and scrapers (SIS, Hooslist, TCF)
    stores/     - Data persistence (RMP reviews)
    
    document_builder.py  - Builds RAG documents from multiple sources
    indexer.py           - Orchestrates fetching and vector indexing
    vector_store.py      - ChromaDB vector database operations
"""

from .vector_store import VectorStore
from .indexer import CourseIndexer
from .document_builder import DocumentBuilder

__all__ = [
    "VectorStore",
    "CourseIndexer",
    "DocumentBuilder",
]
