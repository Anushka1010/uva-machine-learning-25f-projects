"""
FastAPI application entry point.

This module initializes all RAG components and sets up API routes.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.rag.embedder import LocalEmbedder
from app.rag.chunker import TextChunker
from app.rag.vector_store import FAISSVectorStore
from app.rag.rag_pipeline import RAGPipeline
from app.api import upload, query

# Initialize FastAPI app
app = FastAPI(
    title="Local RAG API",
    description="Retrieval-Augmented Generation system for local document search",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite and Next.js default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG components
print("Initializing RAG system...")

# 1. Initialize embedder (loads ML model)
embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2")

# 2. Initialize vector store
vector_store = FAISSVectorStore(
    embedding_dim=embedder.get_embedding_dimension(),
    storage_path="../../data/vectordb"
)

# 3. Initialize chunker
chunker = TextChunker(chunk_size=500, overlap=100)

# 4. Initialize RAG pipeline
rag_pipeline = RAGPipeline(
    vector_store=vector_store,
    embedder=embedder,
    llm_endpoint="http://localhost:11434/api/generate"
)

# 5. Initialize upload dependencies
upload.init_upload_dependencies(
    vs=vector_store,
    emb=embedder,
    chnk=chunker,
    upload_path="../../data/uploads"
)

# 6. Initialize query dependencies
query.init_query_dependencies(
    pipeline=rag_pipeline,
    vs=vector_store
)

# Register routes
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(query.router, prefix="/api", tags=["query"])

print("RAG system initialized successfully!")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "Local RAG API is operational",
        "indexed_files": len(vector_store.get_all_filenames()),
        "total_vectors": vector_store.index.ntotal
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "components": {
            "embedder": "operational",
            "vector_store": "operational",
            "rag_pipeline": "operational"
        },
        "stats": {
            "indexed_files": len(vector_store.get_all_filenames()),
            "total_chunks": vector_store.index.ntotal,
            "embedding_dimension": embedder.get_embedding_dimension()
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
