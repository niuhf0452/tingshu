"""Bearer-token authentication for the public API.

Wire format::

    Authorization: Bearer <base64(username:password)>

Server-side check:
1. If ``auth.enabled=False`` in config → dep is a no-op (dev mode).
2. Otherwise extract the Bearer payload, base64-decode it, split on
   ``:``, and constant-time compare both username and password against
   ``auth.username`` / ``auth.password`` from config.
3. Any mismatch → 401 with ``WWW-Authenticate: Bearer`` so well-behaved
   clients know what scheme was expected.

The ``/health`` endpoint is intentionally not gated (see app.main).
Everything else uses ``require_auth`` as a router-level dependency.
"""
from __future__ import annotations

import base64
import hmac
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from ..config import Settings, get_settings


def _expected_token(username: str, password: str) -> str:
    """Compute the canonical Bearer payload — what the client should
    send. Exposed for tests / clients that want to derive the same
    string offline."""
    raw = f"{username}:{password}".encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def require_auth(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """FastAPI dependency that gates a route on Bearer credentials.

    Disabled (no-op) when ``settings.auth.enabled`` is False, so adding
    this dep to every router is safe even in environments that haven't
    configured auth yet.
    """
    if not settings.auth.enabled:
        return

    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or malformed Authorization header (expected `Bearer <token>`)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = header[len("Bearer "):].strip()

    try:
        decoded = base64.b64decode(token, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="bearer token is not valid base64 utf-8",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if ":" not in decoded:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="bearer token format must be base64('user:pass')",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user, pwd = decoded.split(":", 1)
    # Constant-time compare to avoid leaking match length via timing.
    user_ok = hmac.compare_digest(user, settings.auth.username)
    pwd_ok = hmac.compare_digest(pwd, settings.auth.password)
    if not (user_ok and pwd_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="username or password mismatch",
            headers={"WWW-Authenticate": "Bearer"},
        )
