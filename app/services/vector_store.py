from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue
)
from functools import lru_cache
from core.config import get_settings
import logging

logger = logging.getLogger("rag_agent")
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dim


@lru_cache()
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ensure_collection(collection_name: str = None):
    """Create collection if it doesn't exist."""
    settings = get_settings()
    client = get_qdrant_client()
    if collection_name is None:
        collection_name = settings.qdrant_collection
    
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection: {collection_name}")


def list_collections() -> list[str]:
    """List all available collections."""
    client = get_qdrant_client()
    return [c.name for c in client.get_collections().collections]


def delete_collection(collection_name: str):
    """Delete a collection."""
    client = get_qdrant_client()
    client.delete_collection(collection_name=collection_name)
    logger.info(f"Deleted collection: {collection_name}")


def get_collection_stats(collection_name: str) -> dict:
    """Get collection statistics."""
    client = get_qdrant_client()
    info = client.get_collection(collection_name=collection_name)
    return {
        "name": collection_name,
        "vectors_count": info.vectors_count,
        "points_count": info.points_count,
    }


def upsert_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    chunk_hashes: list[str],
    doc_hash: str,
    filename: str,
    collection_name: str = None,
):
    settings = get_settings()
    client = get_qdrant_client()
    if collection_name is None:
        collection_name = settings.qdrant_collection

    # Fetch existing hashes for this doc to skip duplicates
    existing_hashes = set()
    try:
        results, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="doc_hash", match=MatchValue(value=doc_hash))]
            ),
            with_payload=True,
            limit=10000,
        )
        existing_hashes = {r.payload.get("chunk_hash") for r in results}
    except Exception:
        pass

    points = []
    import uuid
    for chunk, embedding, chunk_hash in zip(chunks, embeddings, chunk_hashes):
        if chunk_hash in existing_hashes:
            continue
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk,
                    "chunk_hash": chunk_hash,
                    "doc_hash": doc_hash,
                    "filename": filename,
                },
            )
        )

    if points:
        client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Upserted {len(points)} chunks to {collection_name}")
    else:
        logger.info(f"All chunks already indexed in {collection_name}")

    return len(points)


def search(query_vector: list[float], top_k: int, collection_name: str = None) -> list[dict]:
    settings = get_settings()
    client = get_qdrant_client()
    if collection_name is None:
        collection_name = settings.qdrant_collection
    
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {"text": r.payload["text"], "score": r.score, "filename": r.payload.get("filename")}
        for r in results
    ]