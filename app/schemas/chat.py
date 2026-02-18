from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class SourceDoc(BaseModel):
    filename: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]