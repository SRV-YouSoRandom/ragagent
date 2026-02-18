from sentence_transformers import SentenceTransformer
from functools import lru_cache
from core.config import get_settings


@lru_cache()
def get_embedder() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    embedder = get_embedder()
    return embedder.encode(texts, show_progress_bar=False).tolist()


def embed_query(query: str) -> list[float]:
    embedder = get_embedder()
    return embedder.encode(query).tolist()