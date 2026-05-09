"""Tests for the Bearer-token auth dependency.

Covers the four states the dep can be in:
1. ``auth.enabled=False`` → no-op (existing dev setups keep working).
2. Enabled + valid token → 200.
3. Enabled + missing/malformed/wrong-credentials → 401 with the
   ``WWW-Authenticate: Bearer`` header that signals the expected scheme.
4. ``/health`` is always reachable, even with auth enabled, so liveness
   probes don't need creds.
"""
from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.chapter_meta_stream import ChapterMetaStreamManager
from app.api.deps import (
    get_book_service,
    get_chapter_meta_stream_manager,
    get_llm_client,
    get_repository,
)
from app.config import Settings, get_settings
from app.core.service import BookService
from app.core.storage import BookRepository
from app.main import create_app
from app.services.llm_stub import StubLLMClient


def _settings_with_auth(enabled: bool, user: str = "alice", pwd: str = "s3cret") -> Settings:
    return Settings.model_validate(
        {"auth": {"enabled": enabled, "username": user, "password": pwd}}
    )


def _bearer(user: str, pwd: str) -> str:
    raw = f"{user}:{pwd}".encode("utf-8")
    return "Bearer " + base64.b64encode(raw).decode("ascii")


@pytest.fixture
def auth_client_factory(tmp_path: Path):
    """Build a TestClient with auth-enabled (or disabled) settings.

    Returns a callable so individual tests can pick the auth state
    without sharing fixture state.
    """

    def _factory(*, enabled: bool):
        repo = BookRepository(tmp_path / "books")
        llm = StubLLMClient()
        service = BookService(repo=repo, llm=llm)
        manager = ChapterMetaStreamManager(service=service, heartbeat_interval_s=10.0)
        app = create_app()
        app.dependency_overrides[get_repository] = lambda: repo
        app.dependency_overrides[get_book_service] = lambda: service
        app.dependency_overrides[get_llm_client] = lambda: llm
        app.dependency_overrides[get_chapter_meta_stream_manager] = lambda: manager
        app.dependency_overrides[get_settings] = lambda: _settings_with_auth(enabled)
        return TestClient(app)

    return _factory


def test_disabled_means_no_auth_required(auth_client_factory):
    with auth_client_factory(enabled=False) as c:
        # /api/books is gated when auth.enabled=True; with it disabled
        # the dep is a no-op so the call succeeds even without a token.
        r = c.get("/api/books")
        assert r.status_code == 200


def test_enabled_valid_credentials(auth_client_factory):
    with auth_client_factory(enabled=True) as c:
        r = c.get("/api/books", headers={"Authorization": _bearer("alice", "s3cret")})
        assert r.status_code == 200


def test_enabled_missing_header(auth_client_factory):
    with auth_client_factory(enabled=True) as c:
        r = c.get("/api/books")
        assert r.status_code == 401
        # Well-behaved clients use this to pick the right scheme.
        assert r.headers.get("WWW-Authenticate") == "Bearer"


def test_enabled_wrong_password(auth_client_factory):
    with auth_client_factory(enabled=True) as c:
        r = c.get("/api/books", headers={"Authorization": _bearer("alice", "wrong")})
        assert r.status_code == 401


def test_enabled_wrong_username(auth_client_factory):
    with auth_client_factory(enabled=True) as c:
        r = c.get("/api/books", headers={"Authorization": _bearer("eve", "s3cret")})
        assert r.status_code == 401


def test_malformed_non_bearer(auth_client_factory):
    with auth_client_factory(enabled=True) as c:
        r = c.get(
            "/api/books",
            headers={"Authorization": "Basic " + base64.b64encode(b"alice:s3cret").decode()},
        )
        assert r.status_code == 401


def test_malformed_not_base64(auth_client_factory):
    with auth_client_factory(enabled=True) as c:
        r = c.get("/api/books", headers={"Authorization": "Bearer !!!not-base64!!!"})
        assert r.status_code == 401


def test_malformed_no_colon(auth_client_factory):
    with auth_client_factory(enabled=True) as c:
        # Valid base64 but no `:` separator — token format is wrong.
        bad = base64.b64encode(b"justusername").decode("ascii")
        r = c.get("/api/books", headers={"Authorization": f"Bearer {bad}"})
        assert r.status_code == 401


def test_health_is_always_public(auth_client_factory):
    with auth_client_factory(enabled=True) as c:
        # Liveness probes shouldn't need creds.
        r = c.get("/health")
        assert r.status_code == 200
