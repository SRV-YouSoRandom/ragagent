from sentence_transformers import CrossEncoder
from functools import lru_cache
from core.config import get_settings
import logging

logger = logging.getLogger("rag_agent")


@lru_cache()
def get_reranker() -> CrossEncoder:
    settings = get_settings()
    logger.info(f"Loading reranker: {settings.reranker_model}")
    return CrossEncoder(settings.reranker_model)


def rerank_documents(query: str, documents: list[dict]) -> list[dict]:
    """
    Rerank documents using cross-encoder.
    Returns documents sorted by relevance score.
    """
    if not documents:
        return documents
    
    settings = get_settings()
    reranker = get_reranker()
    
    # Prepare pairs for cross-encoder
    pairs = [(query, doc["text"]) for doc in documents]
    
    # Get relevance scores
    scores = reranker.predict(pairs)
    
    # Attach scores and sort
    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)
    
    # Sort by rerank score (descending)
    reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
    
    # Keep top K after reranking
    top_reranked = reranked[:settings.rerank_top_k]
    
    logger.info(f"Reranked {len(documents)} → {len(top_reranked)} documents")
    return top_reranked