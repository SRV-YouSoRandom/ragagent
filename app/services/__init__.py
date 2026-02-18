"""Business logic services - parser, chunker, embedder, vector store."""

from app.services.parser import parse_pdf
from app.services.chunker import chunk_text
from app.services.embedder import embed_texts, embed_query, get_embedder
from app.services.hasher import hash_chunk, hash_document
from app.services.vector_store import (
    get_qdrant_client,
    ensure_collection,
    upsert_chunks,
    search,
)

__all__ = [
    "parse_pdf",
    "chunk_text",
    "embed_texts",
    "embed_query",
    "get_embedder",
    "hash_chunk",
    "hash_document",
    "get_qdrant_client",
    "ensure_collection",
    "upsert_chunks",
    "search",
]