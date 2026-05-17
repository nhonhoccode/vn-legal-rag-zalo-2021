"""Session management routes.

- GET /api/sessions/{id} — Read history.
- DELETE /api/sessions/{id} — Clear history.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Path, status

from src.api.dependencies import get_session_manager
from src.api.middleware.auth import AuthenticatedUser, get_current_user
from src.api.schemas import SessionDeleteResponse, SessionResponse, SessionTurnResponse
from src.generation.conversation import SessionManager

router = APIRouter(prefix="/api/sessions", tags=["sessions"])
logger = logging.getLogger(__name__)


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str = Path(..., min_length=1, max_length=64),
    user: AuthenticatedUser = Depends(get_current_user),
):
    manager: SessionManager = get_session_manager()
    history = await manager.get_history(session_id)
    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found or empty",
        )
    return SessionResponse(
        session_id=session_id,
        turns=[
            SessionTurnResponse(
                role=t.role,
                content=t.content,
                timestamp=t.timestamp,
                citations=t.citations,
            )
            for t in history
        ],
        turn_count=len(history),
    )


@router.delete("/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(
    session_id: str = Path(..., min_length=1, max_length=64),
    user: AuthenticatedUser = Depends(get_current_user),
):
    manager: SessionManager = get_session_manager()
    deleted = await manager.clear(session_id)
    return SessionDeleteResponse(session_id=session_id, deleted=deleted)
