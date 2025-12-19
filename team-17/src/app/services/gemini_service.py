"""Google Gemini API service wrapper."""

from typing import Optional
import google.generativeai as genai
from app.config import get_settings

# Module-level chat session storage (persists across GeminiService instances)
_chat_sessions: dict[str, "genai.ChatSession"] = {}


class GeminiService:
    """Service for interacting with Google Gemini API."""
    
    def __init__(self):
        settings = get_settings()
        genai.configure(api_key=settings.gemini_api_key)
        self.model_name = settings.gemini_model
        self.embedding_model = settings.gemini_embedding_model
        self.model = genai.GenerativeModel(self.model_name)
    
    def get_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 999999,
    ) -> str:
        """Get a chat completion from Gemini.
        
        Args:
            prompt: The user's prompt/question
            system_prompt: Optional system instructions
            temperature: Creativity level (0-1)
            max_tokens: Maximum response length
            
        Returns:
            The assistant's response text
        """
        # Combine system prompt and user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        response = self.model.generate_content(
            full_prompt,
            generation_config=generation_config,
        )
        
        return response.text
    
    def get_completion_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 999999,
    ):
        """Get a streaming chat completion from Gemini.
        
        Args:
            prompt: The user's prompt/question
            system_prompt: Optional system instructions
            temperature: Creativity level (0-1)
            max_tokens: Maximum response length
            
        Yields:
            Text chunks as they are generated
        """
        # Combine system prompt and user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        response = self.model.generate_content(
            full_prompt,
            generation_config=generation_config,
            stream=True,
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
    
    def get_chat_completion(
        self,
        prompt: str,
        session_id: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 999999,
    ) -> str:
        """Get a chat completion with conversation memory.
        
        Args:
            prompt: The user's prompt/question
            session_id: Unique session ID for conversation tracking
            system_prompt: Optional system instructions (only used for new sessions)
            temperature: Creativity level (0-1)
            max_tokens: Maximum response length
            
        Returns:
            The assistant's response text
        """
        # Get or create chat session (using module-level storage)
        if session_id not in _chat_sessions:
            # Create new chat with system prompt in history
            history = []
            if system_prompt:
                history = [
                    {"role": "user", "parts": [f"[SYSTEM INSTRUCTIONS - Follow these for all responses]\n{system_prompt}"]},
                    {"role": "model", "parts": ["Understood. I will follow these instructions for our conversation."]}
                ]
            _chat_sessions[session_id] = self.model.start_chat(history=history)
        
        chat = _chat_sessions[session_id]
        
        print("=" * 80)
        print("[GEMINI CHAT PROMPT]")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        response = chat.send_message(prompt, generation_config=generation_config)
        return response.text
    
    def clear_chat_session(self, session_id: str) -> bool:
        """Clear a chat session's history.
        
        Args:
            session_id: Session ID to clear
            
        Returns:
            True if session was cleared, False if it didn't exist
        """
        if session_id in _chat_sessions:
            del _chat_sessions[session_id]
            return True
        return False
    
    def get_embedding(self, text: str, task_type: str = "retrieval_document") -> list[float]:
        """Get embedding vector for text.
        
        Args:
            text: Text to embed
            task_type: One of "retrieval_document" (for stored docs) or "retrieval_query" (for searches)
            
        Returns:
            Embedding vector as list of floats
        """
        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type=task_type,
        )
        
        return result['embedding']
    
    def get_query_embedding(self, text: str) -> list[float]:
        """Get embedding vector for a search query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        return self.get_embedding(text, task_type="retrieval_query")
    
    def get_embeddings_batch(self, texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
        """Get embedding vectors for multiple texts.
        
        Args:
            texts: List of texts to embed
            task_type: One of "retrieval_document" or "retrieval_query"
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text, task_type=task_type)
            embeddings.append(embedding)
        
        return embeddings

