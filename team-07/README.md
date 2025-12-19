# Local RAG Vector Search System

**Team ID:** 07  
**Team Members:** Rui Zhang, Shivam Agrawal

## What is this?

A privacy-focused document search tool that lets you upload your own files (PDFs, Word docs, text files, etc.) and ask questions about them. Everything runs on your machine—no cloud, no API keys needed. We use Ollama to run an LLM locally, so you stay in control of your data.

## How it works

Upload a document → we break it into chunks → convert chunks to embeddings → store in a vector database → when you search, we find the most relevant chunks → send those to an LLM for an answer.

We used:
- **Embeddings:** all-MiniLM-L6-v2 (small and fast)
- **Vector DB:** FAISS (keeps things local)
- **LLM:** Ollama (run llama3.2 or mistral locally)
- **Frontend:** React + Vite 
- **Backend:** FastAPI

## Before you start

You'll need:
- Python 3.9+
- Node.js 18+
- Ollama installed

## Getting started

**Terminal 1 - Backend:**
```bash
cd src/backend
python3 -m venv venv
source venv/bin/activate
pip install -r ../../requirements.txt
python -m app.main
```
Backend runs on `http://localhost:8000`

**Terminal 2 - Frontend:**
```bash
cd src/frontend
npm install
npm run dev
```
Frontend runs on `http://localhost:5173`

**Terminal 3 - Ollama:**
```bash
ollama serve
# In another terminal, grab a model:
ollama pull llama3.2
```

**Then go to `http://localhost:5173` to use the app** - upload a file and start asking questions!

## Stuck?

- **Ports already in use?** Kill them with:
  ```bash
  lsof -ti:8000 | xargs kill -9    # Backend
  lsof -ti:5173 | xargs kill -9    # Frontend
  lsof -ti:11434 | xargs kill -9   # Ollama
  ```
- **TypeScript errors?** Run `npm install` in `src/frontend` and reload VS Code
- **Ollama crashing?** Make sure it's running with `ollama serve`
- **No answer from the LLM?** The retrieval still works, you'll just see the source chunks
