"""Service modules for the application.

Services:
    GeminiService - AI/LLM integration (Gemini API)
    RAGEngine     - Retrieval-Augmented Generation engine
"""

from .gemini_service import GeminiService

def __getattr__(name):
    """Lazy imports to avoid circular dependencies."""
    if name == "RAGEngine":
        from .rag_engine import RAGEngine
        return RAGEngine
    if name == "SISService":
        # Deprecated - redirect to new location
        import warnings
        warnings.warn(
            "SISService is deprecated. Use SISApi from app.data.sources instead.",
            DeprecationWarning,
            stacklevel=2
        )
        from app.data.sources import SISApi
        return SISApi
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "GeminiService",
    "RAGEngine",
]
