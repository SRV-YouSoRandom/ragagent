from pydantic import BaseModel


class IngestResponse(BaseModel):
    filename: str
    doc_hash: str
    total_chunks: int
    new_chunks_indexed: int
    message: str