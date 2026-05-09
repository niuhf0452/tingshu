from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.chapter_meta_stream import EVENT_META, ChapterMetaStreamManager
from app.api.deps import (
    get_book_service,
    get_chapter_meta_stream_manager,
    get_llm_client,
    get_repository,
)
from app.core.service import BookService
from app.core.storage import BookRepository
from app.main import create_app
from app.services.llm_stub import StubLLMClient


@pytest.fixture
def llm():
    return StubLLMClient()


@pytest.fixture
def client(tmp_path: Path, llm):
    repo = BookRepository(tmp_path / "books")
    service = BookService(repo=repo, llm=llm)
    # Bind the SSE manager to this fixture's service so chapter-meta
    # streams hit the right repo. Heartbeat interval is large here —
    # tests aren't asserting heartbeat cadence.
    manager = ChapterMetaStreamManager(service=service, heartbeat_interval_s=10.0)
    app = create_app()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_book_service] = lambda: service
    app.dependency_overrides[get_llm_client] = lambda: llm
    app.dependency_overrides[get_chapter_meta_stream_manager] = lambda: manager
    with TestClient(app) as c:
        yield c


def _parse_sse(body: str) -> list[tuple[str, str]]:
    """Minimal SSE parser shared with the chapter-meta tests — returns
    a list of ``(event, data)`` pairs."""
    events: list[tuple[str, str]] = []
    event_name = "message"
    data_lines: list[str] = []
    for line in body.splitlines():
        if line == "":
            if data_lines:
                events.append((event_name, "\n".join(data_lines)))
            event_name = "message"
            data_lines = []
        elif line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip(" "))
    return events


def _read_chapter_meta_sse(client: TestClient, book_id: str, chapter_id: int):
    """Fire a chapter-meta GET, parse the SSE stream, return the parsed
    JSON body of the final ``meta`` event. Helper used by tests that
    just want the meta payload, not the streaming detail."""
    import json as _json

    r = client.get(f"/api/books/{book_id}/chapters/{chapter_id}/meta")
    assert r.status_code == 200, r.text
    events = _parse_sse(r.text)
    assert events, "stream produced no events"
    name, data = events[-1]
    assert name == EVENT_META, f"expected meta event, got {name}: {data}"
    return _json.loads(data)


SAMPLE_TXT = (
    "第一章 开端\n"
    "萧炎紧握拳头，望向远方。\n"
    "药老缓缓开口：小子别急。\n"
    "萧炎点头，药老微笑。\n"
    "萧炎再次开口。药老叹息。\n\n"
    "第二章 发展\n"
    "萧炎与药老结伴前行。\n"
    "药老指点萧炎修炼。\n"
    "萧炎凝神，药老守护。\n"
)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_upload_triggers_background_analysis(client):
    r = client.post(
        "/api/books/upload",
        files={"file": ("斗破.txt", SAMPLE_TXT.encode("utf-8"), "text/plain")},
    )
    assert r.status_code == 200, r.text
    up = r.json()
    assert up["chapter_count"] == 2
    assert up["title"] == "斗破"
    # Response returns while status is still "processing"; BackgroundTasks
    # flips it to "ready" before TestClient returns control.
    book_id = up["book_id"]

    r = client.get("/api/books")
    assert r.status_code == 200
    book = next(b for b in r.json()["books"] if b["book_id"] == book_id)
    assert book["status"] == "ready"


def test_download_excludes_chapter_metadata(client):
    up = client.post(
        "/api/books/upload",
        files={"file": ("t.txt", SAMPLE_TXT.encode("utf-8"), "text/plain")},
    ).json()
    book_id = up["book_id"]

    r = client.get(f"/api/books/{book_id}/download")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert "meta.json" in names
    assert "chapters/0001.txt" in names
    assert "chapters/0002.txt" in names
    assert not any(n.endswith(".json") and n.startswith("chapters/") for n in names)


def test_upload_rejects_unsupported_extension(client):
    r = client.post(
        "/api/books/upload",
        files={"file": ("x.pdf", b"data", "application/pdf")},
    )
    assert r.status_code == 415


def test_upload_rejects_malformed_epub(client):
    """EPUB is accepted, but garbage bytes should return 400 with a
    meaningful error rather than crashing or silently creating a book."""
    r = client.post(
        "/api/books/upload",
        files={"file": ("bad.epub", b"not a zip file", "application/epub+zip")},
    )
    assert r.status_code == 400
    assert "epub" in r.json()["detail"].lower() or "zip" in r.json()["detail"].lower()


def test_chapter_meta_generated_on_demand(client):
    up = client.post(
        "/api/books/upload",
        files={"file": ("t.txt", SAMPLE_TXT.encode("utf-8"), "text/plain")},
    ).json()
    book_id = up["book_id"]

    # SSE: stream-parse the response and pull the final meta event.
    body = _read_chapter_meta_sse(client, book_id, 1)
    assert body["sentences"]

    r = client.get(f"/api/books/{book_id}/chapters/99/meta")
    assert r.status_code == 404


def test_delete_book_removes_all_files(client, tmp_path):
    """After DELETE, the book's storage directory and everything in it is gone."""
    up = client.post(
        "/api/books/upload",
        files={"file": ("bye.txt", SAMPLE_TXT.encode("utf-8"), "text/plain")},
    ).json()
    book_id = up["book_id"]

    # Trigger chapter-meta generation so there's also a chapters/*.json on disk.
    _read_chapter_meta_sse(client, book_id, 1)

    book_dir = tmp_path / "books" / book_id
    assert book_dir.exists()
    assert (book_dir / "meta.json").exists()
    assert any((book_dir / "chapters").glob("*.txt"))
    assert any((book_dir / "chapters").glob("*.json"))

    r = client.delete(f"/api/books/{book_id}")
    assert r.status_code == 204
    assert r.content == b""
    assert not book_dir.exists()

    # Book drops out of the list.
    r = client.get("/api/books")
    assert not any(b["book_id"] == book_id for b in r.json()["books"])


def test_delete_missing_book_returns_404(client):
    r = client.delete("/api/books/does-not-exist")
    assert r.status_code == 404


def test_delete_is_not_idempotent(client):
    """Second DELETE on the same id returns 404 (matches download/meta 404 behavior)."""
    up = client.post(
        "/api/books/upload",
        files={"file": ("once.txt", SAMPLE_TXT.encode("utf-8"), "text/plain")},
    ).json()
    book_id = up["book_id"]

    assert client.delete(f"/api/books/{book_id}").status_code == 204
    assert client.delete(f"/api/books/{book_id}").status_code == 404


def test_tts_endpoint_smoke(client):
    # Smoke: hits the production TTS service (qwen3 in current config),
    # which emits M4A. Stub-backed cases live in tests/test_tts.py.
    # Use the narrator path (character_id=0) so we don't need a real
    # book / characters.json on disk.
    r = client.post("/api/tts", json={
        "book_id": "smoke",
        "chapter_id": 1,
        "character_id": 0,
        "text": "hello",
        "tone": "neutral",
    })
    assert r.status_code == 200
    # Either audio format is acceptable here — the endpoint sniffs
    # bytes, so this stays correct if the config is later switched
    # back to the stub backend (which still emits WAV).
    assert r.headers["content-type"] in ("audio/mp4", "audio/wav")
