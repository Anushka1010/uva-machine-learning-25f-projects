# Local RAG Vector Search System

Upload your documents and ask questions about them. Everything stays on your computer—no cloud, no sending data anywhere. Uses Ollama to run an LLM locally.

### Our VIDEO DEMO 
https://www.youtube.com/watch?v=N8b3yETTTbA

## What you need

- Python 3.9+
- Node.js 18+
- Ollama

## Getting it running

**Start the backend** (Terminal 1):
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m app.main
```
It'll run on `http://localhost:8000`

**Start the frontend** (Terminal 2):
```bash
cd frontend
npm install
npm run dev
```
Go to `http://localhost:5173`

**Start Ollama** (Terminal 3):
```bash
ollama serve
# In another terminal:
ollama pull llama3.2
```

## How to use it

1. Go to `http://localhost:5173`
2. Upload a PDF, Word doc, text file, or markdown
3. Pick which files you want to search
4. Ask a question
5. Get back an answer with the source chunks

We've got `sample_document.md` in the folder if you want to test it out.

## How it works

You upload a file → we split it into chunks → turn each chunk into embeddings → store them in a database → when you search, we find the closest chunks → send those to an LLM → get an answer.

Using:
- **Embeddings:** all-MiniLM-L6-v2 (small, local, works fast)
- **Database:** FAISS (stores vectors locally)
- **LLM:** Ollama (runs on your machine)
- **Frontend:** React + Vite
- **Backend:** FastAPI

## Stuff that breaks

**Ports already taken?**
```bash
lsof -ti:8000 | xargs kill -9    # Kill backend
lsof -ti:5173 | xargs kill -9    # Kill frontend
lsof -ti:11434 | xargs kill -9   # Kill Ollama
```

**TypeScript errors?** Run `npm install` in the frontend folder and reload VS Code.

**Ollama isn't working?** Make sure you ran `ollama serve`.

**No answers coming back?** The search still works, you'll just see the document chunks without LLM answers.

**Can't upload files?** Make sure `data/uploads/` exists and your file isn't corrupted.

**No search results?** Check that files are actually selected and try a more specific question.

## What's where

- `backend/` - FastAPI app that handles search and embedding
- `frontend/` - React interface
- `data/uploads/` - Your uploaded documents
- `data/vectordb/` - The vector database
- `sample_document.md` - Example file to test with
