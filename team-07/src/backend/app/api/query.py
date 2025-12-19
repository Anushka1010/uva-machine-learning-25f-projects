"""
Query endpoint for RAG system.

Handles natural language queries over indexed documents.
"""

from fastapi import APIRouter, HTTPException
from typing import List

from app.models.schemas import QueryRequest, QueryResponse, SourceInfo, FileInfo

router = APIRouter()

# Dependency injection from main.py
rag_pipeline = None
vector_store = None


def init_query_dependencies(pipeline, vs):
    """Initialize dependencies (called from main.py)."""
    global rag_pipeline, vector_store
    rag_pipeline = pipeline
    vector_store = vs


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query indexed documents using natural language.
    
    Process:
    1. Embed the query using same model as documents
    2. Perform similarity search in vector database
    3. Retrieve top-k most relevant chunks
    4. Construct prompt with retrieved context
    5. Generate answer using LLM
    
    Args:
        request: Query request with question and optional file filters
        
    Returns:
        Generated answer and source chunks with similarity scores
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Execute RAG pipeline
        result = rag_pipeline.query(
            question=request.question,
            top_k=request.top_k,
            active_files=request.active_files
        )
        
        # Format response
        sources = [
            SourceInfo(
                filename=src['filename'],
                chunk=src['text'],
                score=src['score'],
                chunk_index=src['chunk_index']
            )
            for src in result['sources']
        ]
        
        return QueryResponse(
            answer=result['answer'],
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/files", response_model=List[FileInfo])
async def list_files():
    """
    Get list of all indexed files.
    
    Returns:
        List of files with metadata
    """
    try:
        filenames = vector_store.get_all_filenames()
        
        # Count chunks per file
        file_info = []
        for filename in filenames:
            num_chunks = sum(
                1 for meta in vector_store.metadata 
                if meta['filename'] == filename
            )
            
            # Get file type from first chunk
            file_type = next(
                (meta['file_type'] for meta in vector_store.metadata 
                 if meta['filename'] == filename),
                'unknown'
            )
            
            file_info.append(FileInfo(
                filename=filename,
                file_type=file_type,
                num_chunks=num_chunks
            ))
        
        return file_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")
