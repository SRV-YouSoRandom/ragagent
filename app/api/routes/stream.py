from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pipelines.rag import run_rag_streaming
from schemas.chat import ChatRequest

router = APIRouter()


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint - returns tokens as they're generated."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        return StreamingResponse(
            run_rag_streaming(
                request.question,
                collection_name=request.collection_name,
                use_reranking=True
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")