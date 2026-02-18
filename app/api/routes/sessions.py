from fastapi import APIRouter, HTTPException
from services.memory import (
    get_session_history,
    clear_session,
    list_active_sessions,
)
from schemas.chat import SessionHistoryResponse, ChatHistoryItem
from schemas.sessions import SessionListResponse, SessionClearResponse

router = APIRouter()


@router.get("/sessions", response_model=SessionListResponse)
async def get_sessions():
    """List all active sessions."""
    try:
        sessions = list_active_sessions()
        return SessionListResponse(sessions=sessions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/history", response_model=SessionHistoryResponse)
async def get_history(session_id: str):
    """Get conversation history for a session."""
    try:
        history = get_session_history(session_id)
        return SessionHistoryResponse(
            session_id=session_id,
            history=[ChatHistoryItem(**item) for item in history]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}", response_model=SessionClearResponse)
async def delete_session(session_id: str):
    """Clear a session's conversation history."""
    try:
        clear_session(session_id)
        return SessionClearResponse(
            message=f"Session {session_id} cleared successfully.",
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))