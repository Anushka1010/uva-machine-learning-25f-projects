"""
File upload endpoint.

Handles file upload, text extraction, chunking, and indexing.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil
from typing import Dict

from app.models.schemas import UploadResponse
from app.utils.file_processor import FileProcessor
from app.rag.chunker import TextChunker

router = APIRouter()

# These will be injected by main.py
vector_store = None
embedder = None
chunker = None
uploads_dir = None


def init_upload_dependencies(vs, emb, chnk, upload_path):
    """Initialize dependencies (called from main.py)."""
    global vector_store, embedder, chunker, uploads_dir
    vector_store = vs
    embedder = emb
    chunker = chnk
    uploads_dir = Path(upload_path)
    uploads_dir.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and index a document file.
    
    Process:
    1. Save file to local storage
    2. Extract text based on file type
    3. Chunk text into overlapping segments
    4. Embed chunks using local model
    5. Store embeddings in vector database
    
    Supported formats: PDF, TXT, MD, DOCX
    """
    # Validate file type
    allowed_extensions = {'.pdf', '.txt', '.md', '.markdown', '.docx'}
    file_path = Path(file.filename)
    
    if file_path.suffix.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file
    save_path = uploads_dir / file.filename
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Extract text from file
    try:
        text = FileProcessor.extract_text(str(save_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text content found in file")
    
    # Chunk the text
    file_type = file_path.suffix.lstrip('.')
    chunks = chunker.chunk_text(text, file.filename, file_type)
    
    # Embed chunks
    try:
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = embedder.embed_text(chunk_texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed text: {str(e)}")
    
    # Store in vector database
    try:
        vector_store.add_vectors(embeddings, chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store vectors: {str(e)}")
    
    return UploadResponse(
        filename=file.filename,
        chunks_created=len(chunks),
        file_type=file_type
    )
