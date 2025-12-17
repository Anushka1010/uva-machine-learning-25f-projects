"""
RAG (Retrieval-Augmented Generation) Pipeline.

This module orchestrates the complete RAG workflow:
1. Retrieve relevant chunks from vector store
2. Construct prompt with retrieved context
3. Generate answer using LLM

RAG is a technique that enhances LLM responses by:
- Grounding answers in retrieved factual information
- Allowing LLMs to answer questions about custom data
- Reducing hallucinations by providing explicit context
"""

import requests
from typing import List, Dict, Optional
import json


class RAGPipeline:
    """
    Complete RAG pipeline from query to answer.
    
    Components:
    1. Vector store: Retrieves relevant document chunks
    2. Prompt template: Structures context and question for LLM
    3. LLM client: Generates final answer
    """
    
    # Prompt template defines how we feed context to the LLM
    # This is critical for RAG quality
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context to answer the question.
If the context doesn't contain enough information to answer the question, say so clearly.
Be concise but thorough."""

    PROMPT_TEMPLATE = """Context information from documents:
---
{context}
---

Question: {question}

Answer based on the context above:"""
    
    def __init__(self, vector_store, embedder, llm_endpoint: str = "http://localhost:11434/api/generate"):
        """
        Initialize RAG pipeline.
        
        Args:
            vector_store: FAISSVectorStore instance
            embedder: LocalEmbedder instance
            llm_endpoint: Ollama API endpoint for local LLM
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_endpoint = llm_endpoint
        
    def query(self, question: str, top_k: int = 5, 
              active_files: Optional[List[str]] = None) -> Dict:
        """
        Execute complete RAG pipeline.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            active_files: Optional list of filenames to search within
            
        Returns:
            Dict containing:
                - answer: Generated response
                - sources: Retrieved chunks with scores
        """
        # Step 1: Embed the question
        query_embedding = self.embedder.embed_query(question)
        
        # Step 2: Retrieve relevant chunks
        retrieved_chunks = self.vector_store.search(
            query_embedding,
            top_k=top_k,
            filter_filenames=active_files
        )
        
        if not retrieved_chunks:
            return {
                'answer': "No relevant information found in the indexed documents.",
                'sources': []
            }
        
        # Step 3: Construct prompt with retrieved context
        context_str = self._format_context(retrieved_chunks)
        prompt = self.PROMPT_TEMPLATE.format(
            context=context_str,
            question=question
        )
        
        # Step 4: Generate answer using LLM
        answer = self._generate_answer(prompt)
        
        return {
            'answer': answer,
            'sources': retrieved_chunks
        }
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into context string.
        
        Includes source information to help LLM cite sources.
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk['filename']}, chunk {chunk['chunk_index']}]\n"
                f"{chunk['text']}\n"
            )
        return "\n".join(context_parts)
    
    def _generate_answer(self, prompt: str, model: str = "llama3.2") -> str:
        """
        Generate answer using local Ollama LLM.
        
        Args:
            prompt: Complete prompt with context and question
            model: Ollama model name
            
        Returns:
            Generated answer text
            
        Note: This assumes Ollama is running locally on default port 11434.
        Fallback to a simple response if Ollama is not available.
        """
        try:
            # Call Ollama API
            payload = {
                "model": model,
                "prompt": prompt,
                "system": self.SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            }
            
            response = requests.post(
                self.llm_endpoint,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return self._fallback_answer(prompt)
                
        except requests.exceptions.RequestException as e:
            print(f"LLM request failed: {e}")
            return self._fallback_answer(prompt)
    
    def _fallback_answer(self, prompt: str) -> str:
        """
        Fallback response when LLM is unavailable.
        
        For a college project, we can provide context directly as fallback.
        In production, this would return an error.
        """
        return (
            "LLM service unavailable. Showing the retrieved context instead:\n\n"
            f"{prompt}\n\n"
            "Start Ollama with `ollama serve` and ensure the `llama3.2` model is installed."
        )
