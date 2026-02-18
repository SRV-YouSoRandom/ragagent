from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "google/gemma-3-27b-it:free"

    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "rag_docs"

    app_env: str = "production"
    log_level: str = "INFO"
    max_file_size_mb: int = 50
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    rerank_top_k: int = 3

    # NEW: Observability
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "rag-agent"
    
    # Metrics
    enable_metrics: bool = True

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()