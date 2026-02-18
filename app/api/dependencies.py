"""
Shared FastAPI dependencies for dependency injection.
Currently minimal, but provides a pattern for future expansion.
"""
from functools import lru_cache
from app.services.vector_store import get_qdrant_client
from app.services.embedder import get_embedder
from app.core.config import get_settings


@lru_cache()
def get_vector_store():
    """Dependency for Qdrant client - cached singleton."""
    return get_qdrant_client()


@lru_cache()
def get_embedding_model():
    """Dependency for embedding model - cached singleton."""
    return get_embedder()


def get_app_settings():
    """Dependency for settings - use this if you want fresh config per request."""
    return get_settings()