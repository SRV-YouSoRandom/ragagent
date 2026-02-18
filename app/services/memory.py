from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, List
import uuid
import logging

logger = logging.getLogger("rag_agent")

# In-memory session store (in production, use Redis or database)
_sessions: Dict[str, ConversationBufferWindowMemory] = {}


def get_or_create_session(session_id: str = None, window_size: int = 5) -> tuple[str, ConversationBufferWindowMemory]:
    """
    Get existing session or create new one.
    Returns: (session_id, memory)
    """
    if session_id is None:
        session_id = str(uuid.uuid4())
        logger.info(f"Created new session: {session_id}")
    
    if session_id not in _sessions:
        _sessions[session_id] = ConversationBufferWindowMemory(
            k=window_size,  # Keep last 5 exchanges
            return_messages=True,
            memory_key="chat_history",
        )
        logger.info(f"Initialized memory for session: {session_id}")
    
    return session_id, _sessions[session_id]


def add_to_memory(session_id: str, question: str, answer: str):
    """Add Q&A pair to session memory."""
    if session_id not in _sessions:
        logger.warning(f"Session {session_id} not found, creating new")
        _, memory = get_or_create_session(session_id)
    else:
        memory = _sessions[session_id]
    
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(answer)
    logger.info(f"Added exchange to session {session_id}")


def get_session_history(session_id: str) -> List[Dict[str, str]]:
    """Get conversation history for a session."""
    if session_id not in _sessions:
        return []
    
    memory = _sessions[session_id]
    messages = memory.chat_memory.messages
    
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})
    
    return history


def clear_session(session_id: str):
    """Clear a session's memory."""
    if session_id in _sessions:
        del _sessions[session_id]
        logger.info(f"Cleared session: {session_id}")


def list_active_sessions() -> List[str]:
    """List all active session IDs."""
    return list(_sessions.keys())