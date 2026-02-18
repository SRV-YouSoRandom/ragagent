import hashlib


def hash_chunk(text: str) -> str:
    """SHA256 hash of chunk text for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_document(file_bytes: bytes) -> str:
    """SHA256 hash of entire PDF for doc-level dedup."""
    return hashlib.sha256(file_bytes).hexdigest()