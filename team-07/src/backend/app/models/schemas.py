"""Pydantic models for API requests and responses."""

from pydantic import BaseModel
from typing import List, Optional


class UploadResponse(BaseModel):
    """Response from file upload endpoint."""
    filename: str
    chunks_created: int
    file_type: str


class FileInfo(BaseModel):
    """Information about an uploaded file."""
    filename: str
    file_type: str
    num_chunks: int


class QueryRequest(BaseModel):
    """Request body for query endpoint."""
    question: str
    active_files: Optional[List[str]] = None
    top_k: int = 5


class SourceInfo(BaseModel):
    """Information about a retrieved source chunk."""
    filename: str
    chunk: str
    score: float
    chunk_index: int


class QueryResponse(BaseModel):
    """Response from query endpoint."""
    answer: str
    sources: List[SourceInfo]
