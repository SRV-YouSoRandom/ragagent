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


def ensure_collection():
    settings = get_settings()
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection: {settings.qdrant_collection}")


def upsert_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    chunk_hashes: list[str],
    doc_hash: str,
    filename: str,
):
    settings = get_settings()
    client = get_qdrant_client()

    # Fetch existing hashes for this doc to skip duplicates
    existing_hashes = set()
    try:
        results, _ = client.scroll(
            collection_name=settings.qdrant_collection,
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
            continue  # Skip duplicate chunks
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
        client.upsert(collection_name=settings.qdrant_collection, points=points)
        logger.info(f"Upserted {len(points)} chunks for doc: {filename}")
    else:
        logger.info(f"All chunks already indexed for doc: {filename}")

    return len(points)


def search(query_vector: list[float], top_k: int) -> list[dict]:
    settings = get_settings()
    client = get_qdrant_client()
    results = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {"text": r.payload["text"], "score": r.score, "filename": r.payload.get("filename")}
        for r in results
    ]