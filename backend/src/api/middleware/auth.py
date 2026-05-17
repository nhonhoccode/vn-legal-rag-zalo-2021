"""Auth middleware: dual-track API key + JWT.

Per Decision 14:
- **Track 1**: end users qua UI → JWT (NextAuth.js issue, FastAPI verify).
- **Track 2**: scripts/admin → API key trong header `X-API-Key`.

Endpoint nhận EITHER track. Cả 2 đều không có → 401.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Header, HTTPException, status
from jose import JWTError, jwt

from src.config import settings


@dataclass(slots=True)
class AuthenticatedUser:
    """Identity sau khi auth thành công."""

    id: str
    email: str | None
    auth_type: str  # "jwt" | "apikey"


def verify_jwt(token: str) -> AuthenticatedUser:
    """Decode JWT, verify signature."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid JWT: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

    user_id = payload.get("sub") or payload.get("email", "")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT missing sub/email",
        )
    return AuthenticatedUser(
        id=str(user_id),
        email=payload.get("email"),
        auth_type="jwt",
    )


def verify_api_key(api_key: str) -> AuthenticatedUser:
    if not settings.allowed_api_keys or api_key not in settings.allowed_api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return AuthenticatedUser(
        id=f"apikey:{api_key[:6]}",
        email=None,
        auth_type="apikey",
    )


async def get_current_user(
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> AuthenticatedUser:
    """FastAPI dependency: verify EITHER Authorization Bearer (JWT) OR X-API-Key."""
    # Track 1: JWT
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
        return verify_jwt(token)

    # Track 2: API key
    if x_api_key:
        return verify_api_key(x_api_key)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing credentials. Use Authorization: Bearer <jwt> hoặc X-API-Key: <key>.",
        headers={"WWW-Authenticate": "Bearer"},
    )
