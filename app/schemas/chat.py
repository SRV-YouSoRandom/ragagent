from pydantic import BaseModel
from typing import Optional, List


class ChatRequest(BaseModel):
    question: str
    collection_name: Optional[str] = None
    session_id: Optional[str] = None  # NEW: For conversation memory


class SourceDoc(BaseModel):
    filename: str
    score: float
    rerank_score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]
    collection_name: str
    session_id: str  # NEW: Return session ID for follow-ups


class ChatHistoryItem(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    history: List[ChatHistoryItem]