from pydantic import BaseModel
from typing import List


class SessionListResponse(BaseModel):
    sessions: List[str]


class SessionClearResponse(BaseModel):
    message: str
    session_id: str