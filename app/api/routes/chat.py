from fastapi import APIRouter, HTTPException
from pipelines.rag import run_rag
from schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = run_rag(
            request.question,
            collection_name=request.collection_name,
            use_reranking=True  # NEW: Reranking enabled
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline failed: {str(e)}")

    return ChatResponse(**result)