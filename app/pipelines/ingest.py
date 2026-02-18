from app.services.parser import parse_pdf
from app.services.chunker import chunk_text
from app.services.embedder import embed_texts
from app.services.hasher import hash_chunk, hash_document
from app.services.vector_store import ensure_collection, upsert_chunks
import logging

logger = logging.getLogger("rag_agent")


def run_ingest(file_bytes: bytes, filename: str) -> dict:
    logger.info(f"Starting ingest for: {filename}")

    # 1. Hash document for dedup
    doc_hash = hash_document(file_bytes)

    # 2. Parse PDF
    raw_text = parse_pdf(file_bytes)
    if not raw_text.strip():
        raise ValueError("PDF appears to be empty or unreadable.")

    # 3. Chunk
    chunks = chunk_text(raw_text)
    logger.info(f"Produced {len(chunks)} chunks")

    # 4. Hash chunks
    chunk_hashes = [hash_chunk(c) for c in chunks]

    # 5. Embed
    embeddings = embed_texts(chunks)

    # 6. Ensure collection exists + upsert
    ensure_collection()
    upserted = upsert_chunks(chunks, embeddings, chunk_hashes, doc_hash, filename)

    return {
        "filename": filename,
        "doc_hash": doc_hash,
        "total_chunks": len(chunks),
        "new_chunks_indexed": upserted,
    }