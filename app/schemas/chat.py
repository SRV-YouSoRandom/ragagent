from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    collection_name: str = None  # NEW: Optional collection


class SourceDoc(BaseModel):
    filename: str
    score: float
    rerank_score: float = None  # NEW: Reranking score


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]
    collection_name: str  # NEW: Which collection was used