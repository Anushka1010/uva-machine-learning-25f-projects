"""Chat router for AI assistant interactions."""

import uuid
import json
import logging
import mistune
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from app.config import get_settings
from app.services.rag_engine import RAGEngine

# Setup logging
logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/chat", tags=["chat"])
settings = get_settings()
templates = Jinja2Templates(directory=str(settings.templates_dir))

# Markdown renderer - mistune is fast and handles edge cases well
md = mistune.create_markdown(escape=False, plugins=['table', 'strikethrough'])

def render_markdown(text: str) -> str:
    """Convert markdown to HTML."""
    return md(text)

# Register markdown filter for templates
templates.env.filters['markdown'] = render_markdown

# In-memory session storage (for demo purposes)
chat_sessions: dict[str, list[dict]] = {}


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def chat_page(request: Request, session_id: str = None):
    """Render the chat interface."""
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
    
    # Get conversation history for this session
    messages = chat_sessions.get(session_id, [])
    
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "session_id": session_id,
            "messages": messages,
            "title": "Chat - HoosAdvisor",
        }
    )


@router.post("/send", response_class=HTMLResponse)
async def send_message(
    request: Request,
    message: str = Form(...),
    session_id: str = Form(...),
):
    """Process a chat message and return response directly (no redirect for speed)."""
    # Initialize session if needed
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    # Add user message
    chat_sessions[session_id].append({
        "role": "user",
        "content": message,
    })
    
    try:
        logger.info(f"[CHAT] Received message: {message}")
        
        # Get RAG response with session memory
        rag_engine = RAGEngine()
        logger.info("[CHAT] Calling RAG query...")
        result = rag_engine.query(message, session_id=session_id)
        logger.info(f"[CHAT] Got result with {result.get('context_used', 0)} context docs")
        
        response_text = result["response"]
        
    except Exception as e:
        import traceback
        logger.error(f"[CHAT ERROR] {e}")
        traceback.print_exc()
        response_text = f"I apologize, but I encountered an error processing your request. Please try again. (Error: {str(e)})"
    
    # Add assistant response
    chat_sessions[session_id].append({
        "role": "assistant",
        "content": response_text,
    })
    
    # Return rendered page directly (faster than redirect)
    messages = chat_sessions.get(session_id, [])
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "session_id": session_id,
            "messages": messages,
            "title": "Chat - HoosAdvisor",
        }
    )


@router.post("/clear", response_class=HTMLResponse)
async def clear_chat(session_id: str = Form(...)):
    """Clear chat history for a session."""
    # Clear local message storage
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    
    # Clear Gemini conversation memory
    rag_engine = RAGEngine()
    rag_engine.clear_session(session_id)
    
    return RedirectResponse(
        url=f"/chat?session_id={session_id}",
        status_code=303,
    )


@router.post("/api")
async def api_send_message(
    message: str = Form(...),
    session_id: str = Form(...),
):
    """Simple JSON API for chat - returns response as HTML (markdown rendered)."""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    chat_sessions[session_id].append({"role": "user", "content": message})
    
    try:
        rag_engine = RAGEngine()
        result = rag_engine.query(message, session_id=session_id)
        response_text = result["response"]
    except Exception as e:
        response_text = f"Error: {str(e)}"
    
    chat_sessions[session_id].append({"role": "assistant", "content": response_text})
    
    # Return markdown-rendered HTML
    return {"response": render_markdown(response_text)}


@router.get("/stream")
async def stream_response(message: str, session_id: str):
    """Stream a chat response using Server-Sent Events (SSE).
    
    This endpoint returns a stream of text chunks as they are generated,
    allowing for real-time display of the AI response.
    """
    def generate():
        try:
            # Add user message to session
            if session_id not in chat_sessions:
                chat_sessions[session_id] = []
            
            chat_sessions[session_id].append({
                "role": "user",
                "content": message,
            })
            
            # Stream RAG response
            rag_engine = RAGEngine()
            full_response = ""
            
            for chunk in rag_engine.query_stream(message):
                full_response += chunk
                # SSE format: data: <content>\n\n
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            
            # Save complete response to session
            chat_sessions[session_id].append({
                "role": "assistant",
                "content": full_response,
            })
            
            # Signal completion
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            logger.error(f"[STREAM ERROR] {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

