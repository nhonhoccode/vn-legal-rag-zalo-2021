"""Admin stats endpoint: GET /api/admin/stats — query log aggregates."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.dependencies import get_query_logger
from src.api.middleware.auth import AuthenticatedUser, get_current_user
from src.api.middleware.query_log import QueryLogger

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/stats")
async def stats(
    hours: int = Query(24, ge=1, le=720),
    user: AuthenticatedUser = Depends(get_current_user),
    logger: QueryLogger = Depends(get_query_logger),
):
    # Chỉ allow API key path (admin), không cho user qua JWT.
    if user.auth_type != "apikey":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin endpoint — chỉ API key.",
        )
    since = int(time.time()) - hours * 3600
    return {"window_hours": hours, **logger.stats(since_ts=since)}
