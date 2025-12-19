# Backend - Local RAG System

Python FastAPI backend for document indexing and retrieval-augmented generation.

## Architecture

```
┌─────────────┐
│   FastAPI   │ - REST API endpoints
└──────┬──────┘
       │
       ├─> File Upload → Text Extraction → Chunking → Embedding → Vector Store
       │
       └─> Query → Embedding → Similarity Search → Context Retrieval → LLM → Response
```

## Components

### 1. **File Processing** (`utils/file_processor.py`)
- Extracts text from PDF, TXT, MD, DOCX files
- Handles encoding and format-specific parsing

### 2. **Text Chunking** (`rag/chunker.py`)
- Splits documents into ~500 token chunks with 100 token overlap
- Preserves context across chunk boundaries
- Stores metadata (filename, chunk index, file type)

### 3. **Embedding** (`rag/embedder.py`)
- Uses `sentence-transformers` with `all-MiniLM-L6-v2` model
- Generates 384-dimensional embeddings
- Local execution (no API calls)

### 4. **Vector Store** (`rag/vector_store.py`)
- FAISS for efficient similarity search
- Persists to disk (`data/vectordb/`)
- Supports metadata filtering

### 5. **RAG Pipeline** (`rag/rag_pipeline.py`)
- Orchestrates retrieval + generation
- Constructs prompts with retrieved context
- Calls local Ollama LLM

## API Endpoints

### `POST /api/upload`
Upload and index a document.

**Request:**
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "filename": "document.pdf",
  "chunks_created": 42,
  "file_type": "pdf"
}
```

### `GET /api/files`
List all indexed files.

**Response:**
```json
[
  {
    "filename": "document.pdf",
    "file_type": "pdf",
    "num_chunks": 42
  }
]
```

### `POST /api/query`
Query indexed documents.

**Request:**
```json
{
  "question": "What is the main topic?",
  "active_files": ["document.pdf"],
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "The main topic is...",
  "sources": [
    {
      "filename": "document.pdf",
      "chunk": "Retrieved text chunk...",
      "score": 0.87,
      "chunk_index": 5
    }
  ]
}
```

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Ollama (for LLM)
```bash
# macOS
brew install ollama

# Start Ollama service
ollama serve

# Pull a model (in another terminal)
ollama pull llama3.2
```

### 3. Run Server
```bash
cd backend
python -m app.main
```

Server will start on `http://localhost:8000`

## Design Decisions

### Chunking Strategy
- **Size: 500 tokens (~375 words)**
  - Large enough to capture complete ideas
  - Small enough to fit in embedding model context
  - Balances retrieval precision vs context richness
  
- **Overlap: 100 tokens (~75 words)**
  - Prevents information loss at boundaries
  - Increases storage by ~20% but improves recall
  
### Embedding Model
- **all-MiniLM-L6-v2**
  - Fast: ~400 sentences/sec on CPU
  - Small: ~80MB download
  - Quality: Trained on 1B+ sentence pairs
  - Trade-off: Lower quality than larger models (e.g., all-mpnet-base-v2) but 2x faster

### Vector Database
- **FAISS**
  - Production-grade (used by Meta)
  - Efficient: O(log n) search with IndexIVFFlat
  - Simple: IndexFlatL2 for exact search (current implementation)
  - Persistent: Serializes to disk

### LLM Integration
- **Ollama (local)**
  - No API costs
  - Privacy: Data stays local
  - Offline capable
  - Models: LLaMA 3.2, Mistral, etc.
  - Fallback: Returns context if LLM unavailable

## How RAG Works

1. **Indexing Phase**
   ```
   Document → Chunks → Embeddings → Vector DB
   ```
   - Transform documents into searchable vectors
   - Semantic similarity in embedding space

2. **Query Phase**
   ```
   Question → Embedding → Similarity Search → Top-K Chunks
   ```
   - Find chunks with similar meaning to question
   - Cosine similarity in 384-dim space

3. **Generation Phase**
   ```
   Question + Retrieved Context → LLM → Answer
   ```
   - LLM grounds answer in retrieved facts
   - Reduces hallucinations

## Why Vector Similarity Works

- Embeddings map semantic meaning to geometric space
- Similar concepts cluster together (e.g., "car" near "automobile")
- Distance metrics (L2, cosine) measure semantic similarity
- Neural networks learn these representations from massive text corpora

## Limitations

1. **No re-ranking**: First-stage retrieval only
   - Could add cross-encoder for better precision
   
2. **Fixed chunk size**: Not adaptive to document structure
   - Could use semantic chunking (split on topics)
   
3. **No deletion support**: FAISS doesn't support efficient deletion
   - Would need to rebuild index or use different DB
   
4. **Single-vector per chunk**: No multi-vector retrieval
   - Could use ColBERT for token-level matching
   
5. **No query expansion**: Single query embedding only
   - Could generate multiple query variations

## Future Improvements

- [ ] Add cross-encoder re-ranking
- [ ] Implement semantic chunking
- [ ] Add query expansion (HyDE, etc.)
- [ ] Support multiple embedding models
- [ ] Add evaluation metrics (retrieval recall, answer quality)
- [ ] Implement hybrid search (vector + keyword)
- [ ] Add streaming responses
- [ ] Support incremental updates
