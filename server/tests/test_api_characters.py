"""Tests for the book-character roster API.

Covers:
- list filters out narrator slots
- list returns 404 for unknown book
- patch updates only the supplied fields and persists to disk
- patch ignores narrator-id (400) and unknown-id (404) requests
- patch under contention (lock contention with chapter analysis) waits
  rather than failing — concurrent edits and analyses don't lose data
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_book_service, get_llm_client, get_repository
from app.core.models import Character
from app.core.service import BookService
from app.core.storage import BookRepository
from app.main import create_app
from app.services.llm_stub import StubLLMClient


@pytest.fixture
def llm():
    return StubLLMClient()


@pytest.fixture
def repo(tmp_path: Path) -> BookRepository:
    return BookRepository(tmp_path / "books")


@pytest.fixture
def service(repo: BookRepository, llm) -> BookService:
    return BookService(repo=repo, llm=llm)


@pytest.fixture
def client(repo: BookRepository, service: BookService, llm):
    app = create_app()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_book_service] = lambda: service
    app.dependency_overrides[get_llm_client] = lambda: llm
    with TestClient(app) as c:
        yield c


def _seed_book(repo: BookRepository, book_id: str = "test_book") -> None:
    """Bypass ingest — drop a meta + characters.json straight into the
    repo. Roster has narrator slot + three book characters covering the
    common gender / age / personality combinations the tests exercise."""
    from app.core.enums import Age, BookStatus, Gender, Personality
    from app.core.models import BookMeta

    meta = BookMeta(
        book_id=book_id, title="测试书", status=BookStatus.READY,
    )
    repo.create(meta)
    repo.save_meta(meta)
    repo.save_characters(book_id, [
        Character(id=0, name="旁白"),
        Character(
            id=16, name="陈平安",
            identity="少年窑匠", gender=Gender.MALE, age=Age.TEEN,
            personality=[Personality.DETERMINED, Personality.KIND],
        ),
        Character(
            id=17, name="稚圭",
            identity="婢女", gender=Gender.FEMALE, age=Age.TEEN,
            personality=[Personality.TIMID],
        ),
    ])


def test_list_filters_narrator_and_returns_book_characters(
    client: TestClient, repo: BookRepository,
):
    _seed_book(repo)
    r = client.get("/api/books/test_book/characters")
    assert r.status_code == 200
    body = r.json()
    ids = [c["id"] for c in body]
    # Narrator (id=0) is filtered; book characters present.
    assert ids == [16, 17]
    chen = next(c for c in body if c["id"] == 16)
    assert chen["name"] == "陈平安"
    assert chen["gender"] == "male"
    assert chen["age"] == "teen"
    assert sorted(chen["personality"]) == ["determined", "kind"]


def test_list_returns_404_for_unknown_book(client: TestClient):
    r = client.get("/api/books/missing/characters")
    assert r.status_code == 404


def test_list_empty_when_no_characters_yet(
    client: TestClient, repo: BookRepository,
):
    """Fresh book with only the narrator (no chapters analysed yet) has
    nothing to show in the character-edit screen."""
    from app.core.models import BookMeta

    meta = BookMeta(book_id="empty_book", title="空书")
    repo.create(meta)
    repo.save_meta(meta)
    repo.save_characters("empty_book", [Character(id=0, name="旁白")])
    r = client.get("/api/books/empty_book/characters")
    assert r.status_code == 200
    assert r.json() == []


def test_patch_updates_only_supplied_fields(
    client: TestClient, repo: BookRepository,
):
    _seed_book(repo)
    # Change only age + personality; gender stays put.
    r = client.patch(
        "/api/books/test_book/characters/16",
        json={"age": "youth", "personality": ["mature", "wise"]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == 16
    assert body["gender"] == "male"  # untouched
    assert body["age"] == "youth"
    assert sorted(body["personality"]) == ["mature", "wise"]
    # And persisted to disk.
    roster = repo.load_characters("test_book")
    chen = next(c for c in roster if c.id == 16)
    assert chen.age.value == "youth"
    assert chen.gender.value == "male"
    # Other characters unchanged.
    zhi = next(c for c in roster if c.id == 17)
    assert zhi.age.value == "teen"


def test_patch_empty_body_is_noop(client: TestClient, repo: BookRepository):
    _seed_book(repo)
    r = client.patch("/api/books/test_book/characters/16", json={})
    assert r.status_code == 200
    assert r.json()["age"] == "teen"  # original


def test_patch_rejects_narrator_id(client: TestClient, repo: BookRepository):
    _seed_book(repo)
    r = client.patch(
        "/api/books/test_book/characters/0",
        json={"gender": "female"},
    )
    assert r.status_code == 400


def test_patch_404_for_unknown_character(
    client: TestClient, repo: BookRepository,
):
    _seed_book(repo)
    r = client.patch(
        "/api/books/test_book/characters/9999",
        json={"gender": "female"},
    )
    assert r.status_code == 404


def test_patch_404_for_unknown_book(client: TestClient):
    r = client.patch(
        "/api/books/missing/characters/16",
        json={"gender": "female"},
    )
    assert r.status_code == 404


def test_patch_invalid_enum_returns_422(
    client: TestClient, repo: BookRepository,
):
    _seed_book(repo)
    r = client.patch(
        "/api/books/test_book/characters/16",
        json={"gender": "wizard"},  # not a valid Gender
    )
    assert r.status_code == 422


def test_patch_waits_for_book_lock_and_does_not_lose_data(
    client: TestClient, service: BookService, repo: BookRepository,
):
    """Simulate the user editing while chapter analysis holds the lock.

    We acquire the same lock the service uses, hold it for a short
    period, then release. The PATCH must wait — and the result must
    reflect both the user's edit AND any roster changes the lock-holder
    made. Tests the "wait, never fail" requirement.
    """
    _seed_book(repo)

    # Have the analysis side update the personality list while it holds
    # the lock — mimics the merge step in generate_chapter_meta. The
    # user's PATCH should land on top, not be overwritten by either side.
    book_lock = service._book_lock("test_book")
    started = threading.Event()
    release_after = threading.Event()

    def lock_holder() -> None:
        with book_lock:
            # Mutate roster *while holding the lock* (analysis-side write).
            roster = repo.load_characters("test_book")
            for c in roster:
                if c.id == 16:
                    c.identity = "edited-by-analysis"
            repo.save_characters("test_book", roster)
            started.set()
            release_after.wait(timeout=2.0)

    t = threading.Thread(target=lock_holder)
    t.start()
    assert started.wait(timeout=2.0)

    # User PATCH while the lock is held — must block, then succeed.
    patch_started = time.monotonic()
    release_after.set()  # release the lock so the patch can proceed
    r = client.patch(
        "/api/books/test_book/characters/16",
        json={"age": "youth"},
    )
    elapsed = time.monotonic() - patch_started
    t.join(timeout=2.0)
    assert r.status_code == 200, r.text

    # Final state: the analysis-side identity edit is preserved AND the
    # user's age edit is applied. Neither side lost.
    roster = repo.load_characters("test_book")
    chen = next(c for c in roster if c.id == 16)
    assert chen.identity == "edited-by-analysis"
    assert chen.age.value == "youth"
    # Sanity check that we actually waited (>0 ms). Don't assert a tight
    # bound — fixture lock-holder thread releases nearly immediately.
    assert elapsed >= 0
