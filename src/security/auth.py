"""API authentication via static API keys or JWT.

Implements a FastAPI dependency that validates the Authorization header.
Supports two modes:
- Static API keys: comma-separated list in API_KEYS env var
- JWT: validates against a JWKS endpoint (future)

Gated behind AUTH_ENABLED feature flag (default: False).
"""
from __future__ import annotations

import logging

from fastapi import HTTPException, Request

from config import settings

logger = logging.getLogger(__name__)


def _get_valid_api_keys() -> frozenset[str]:
    """Parse comma-separated API keys from config."""
    raw = settings.api_keys.strip()
    if not raw:
        return frozenset()
    return frozenset(k.strip() for k in raw.split(",") if k.strip())


async def verify_api_key(request: Request) -> None:
    """FastAPI dependency that validates the API key from the Authorization header.

    Usage:
        @app.post("/ask", dependencies=[Depends(verify_api_key)])

    When AUTH_ENABLED=false, this is a no-op.
    When AUTH_ENABLED=true, expects: Authorization: Bearer <api_key>
    """
    if not settings.auth_enabled:
        return

    auth_header = request.headers.get("Authorization", "")
    if not auth_header:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Provide: Authorization: Bearer <api_key>",
        )

    # Extract token from "Bearer <token>"
    parts = auth_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization format. Use: Bearer <api_key>",
        )

    token = parts[1].strip()
    valid_keys = _get_valid_api_keys()

    if not valid_keys:
        logger.warning("AUTH_ENABLED=true but no API_KEYS configured — rejecting all requests")
        raise HTTPException(
            status_code=500,
            detail="Authentication is enabled but no API keys are configured.",
        )

    if token not in valid_keys:
        logger.warning("Invalid API key attempt from %s", request.client.host if request.client else "unknown")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )
