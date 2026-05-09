"""End-to-end tests for chapter metadata generation: SSE streaming +
caching + lookahead + concurrent-request coalescing.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.chapter_meta_stream import (
    EVENT_ERROR,
    EVENT_HEARTBEAT,
    EVENT_META,
    ChapterMetaStreamManager,
)
from app.api.deps import (
    get_book_service,
    get_chapter_meta_stream_manager,
    get_llm_client,
    get_repository,
)
from app.config import get_settings
from app.core.models import (
    AnalyzedSentence,
    ChapterMeta,
    Character,
    ClassifiedCharacters,
)
from app.core.service import BookService
from app.core.storage import BookRepository
from app.main import create_app
from app.services.llm_stub import StubLLMClient


def _settings_with_lookahead(n: int):
    """Return a deep-copied Settings with a custom lookahead count.
    ``get_settings()`` itself is lru-cached and shared, so we copy
    instead of mutating it (which would leak across tests)."""
    fresh = get_settings().model_copy(deep=True)
    fresh.preprocess.lookahead_chapters = n
    return fresh


SAMPLE_TXT = (
    "第一章 开端\n"
    "萧炎紧握拳头。\n"
    "药老说：「小子别急。」\n\n"
    "第二章 发展\n"
    "药老指点萧炎。萧炎凝神。\n"
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def llm():
    return StubLLMClient()


@pytest.fixture
def service_client(tmp_path: Path, llm):
    repo = BookRepository(tmp_path / "books")
    service = BookService(repo=repo, llm=llm)
    # Each test gets a fresh manager bound to this fixture's service —
    # otherwise the lru_cache'd default keeps stale jobs across tests.
    manager = ChapterMetaStreamManager(
        service=service, heartbeat_interval_s=0.1,
    )
    app = create_app()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_book_service] = lambda: service
    app.dependency_overrides[get_llm_client] = lambda: llm
    app.dependency_overrides[get_chapter_meta_stream_manager] = lambda: manager
    with TestClient(app) as c:
        yield service, repo, c, manager


def _upload(client: TestClient) -> str:
    r = client.post(
        "/api/books/upload",
        files={"file": ("t.txt", SAMPLE_TXT.encode("utf-8"), "text/plain")},
    )
    assert r.status_code == 200
    return r.json()["book_id"]


# ---------------------------------------------------------------------------
# SSE parser
# ---------------------------------------------------------------------------


def _parse_sse(body: str) -> list[tuple[str, str]]:
    """Parse SSE wire format into a list of (event, data) pairs.
    ``data`` is joined with ``\\n`` if multi-line. Empty lines separate
    events; lines not starting with ``event:`` / ``data:`` are ignored.
    """
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


# ---------------------------------------------------------------------------
# service-level tests (unchanged — these don't go through the API)
# ---------------------------------------------------------------------------


def test_service_generates_meta_with_positions(service_client):
    service, repo, client, _ = service_client
    book_id = _upload(client)

    meta = service.generate_chapter_meta(book_id, 1)
    assert isinstance(meta, ChapterMeta)
    assert meta.sentences, "stub must produce at least one sentence"

    text = repo.read_chapter_text(book_id, 1)
    lines = text.split("\n")
    for s in meta.sentences:
        assert 1 <= s.start_line <= len(lines)
        line = lines[s.start_line - 1]
        assert 0 <= s.start_col <= len(line)
        assert s.start_col < s.end_col
        if s.start_line == s.end_line:
            assert s.end_col <= len(line)


def test_service_get_or_generate_caches(service_client):
    service, repo, client, _ = service_client
    book_id = _upload(client)

    first = service.get_or_generate_chapter_meta(book_id, 1)
    assert repo.chapter_meta_path(book_id, 1).exists()

    second = service.get_or_generate_chapter_meta(book_id, 1)
    assert second.model_dump() == first.model_dump()


def test_chapter_meta_embeds_character_snapshot(service_client):
    service, _, client, _ = service_client
    book_id = _upload(client)

    meta = service.generate_chapter_meta(book_id, 1)
    assert meta.characters
    names = [c.name for c in meta.characters]
    assert "药老" in names
    yaolao = next(c for c in meta.characters if c.name == "药老")
    assert yaolao.id != 0
    assert yaolao.identity


def test_stub_attributes_dialogue_to_named_speaker(service_client):
    service, repo, client, _ = service_client
    book_id = _upload(client)

    meta = service.generate_chapter_meta(book_id, 1)
    assert meta.sentences

    cumulative = repo.load_characters(book_id)
    names = {c.name for c in cumulative}
    assert "药老" in names

    lines = repo.read_chapter_text(book_id, 1).split("\n")
    dialogue_sents = [
        s for s in meta.sentences
        if "「" in lines[s.start_line - 1][s.start_col:s.end_col]
    ]
    assert dialogue_sents
    yaolao_id = next(c.id for c in cumulative if c.name == "药老")
    assert dialogue_sents[0].character_id == yaolao_id


# ---------------------------------------------------------------------------
# API tests (SSE)
# ---------------------------------------------------------------------------


def test_api_sse_returns_meta_event(service_client):
    """Cache miss → final ``meta`` event carries the full ChapterMeta JSON."""
    _, _, client, _ = service_client
    book_id = _upload(client)

    r = client.get(f"/api/books/{book_id}/chapters/1/meta")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(r.text)
    assert events, "stream must yield at least one event"
    name, data = events[-1]
    assert name == EVENT_META
    body = json.loads(data)
    sentences = body["sentences"]
    assert len(sentences) >= 1
    assert set(sentences[0].keys()) == {
        "start_line", "start_col", "end_line", "end_col", "character_id", "tone",
    }
    assert isinstance(body["characters"], list)


def test_api_sse_cache_hit_no_heartbeat(service_client):
    """When the chapter is already cached, only a single ``meta`` event
    fires — no heartbeats, no LLM call."""
    service, _, client, _ = service_client
    book_id = _upload(client)
    # Pre-populate the cache.
    service.generate_chapter_meta(book_id, 1)

    r = client.get(f"/api/books/{book_id}/chapters/1/meta")
    assert r.status_code == 200
    events = _parse_sse(r.text)
    assert len(events) == 1, f"expected 1 event for cache hit, got {events}"
    assert events[0][0] == EVENT_META


def test_api_sse_heartbeat_during_slow_analysis(tmp_path):
    """A slow LLM run must produce at least one heartbeat before the
    final ``meta`` event."""
    slow_llm = _SlowLLM(StubLLMClient(), delay_s=0.4)
    repo = BookRepository(tmp_path / "books")
    service = BookService(repo=repo, llm=slow_llm)
    # Heartbeat every 0.1 s → at least 3 heartbeats during a 0.4 s job.
    manager = ChapterMetaStreamManager(service=service, heartbeat_interval_s=0.1)
    app = create_app()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_book_service] = lambda: service
    app.dependency_overrides[get_llm_client] = lambda: slow_llm
    app.dependency_overrides[get_chapter_meta_stream_manager] = lambda: manager

    with TestClient(app) as client:
        book_id = _upload(client)
        r = client.get(f"/api/books/{book_id}/chapters/1/meta")
        assert r.status_code == 200
        events = _parse_sse(r.text)

    names = [name for name, _ in events]
    assert names.count(EVENT_HEARTBEAT) >= 1, f"no heartbeat in {names}"
    assert names[-1] == EVENT_META

    # Heartbeat payloads carry elapsed seconds + stage hint.
    first_hb = next(data for name, data in events if name == EVENT_HEARTBEAT)
    payload = json.loads(first_hb)
    assert payload["stage"] == "analyzing"
    assert payload["elapsed_s"] >= 0


def test_api_sse_concurrent_requests_share_one_llm_call(tmp_path):
    """Three concurrent requests for the same chapter must hit the LLM
    only once — the in-flight job is shared via the manager. Lookahead
    is disabled so the assertion measures only the coalescing of the
    requested chapter."""
    counting_llm = _CountingLLM(StubLLMClient(), delay_s=0.4)
    repo = BookRepository(tmp_path / "books")
    service = BookService(repo=repo, llm=counting_llm)
    manager = ChapterMetaStreamManager(service=service, heartbeat_interval_s=0.1)
    app = create_app()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_book_service] = lambda: service
    app.dependency_overrides[get_llm_client] = lambda: counting_llm
    app.dependency_overrides[get_chapter_meta_stream_manager] = lambda: manager
    # Lookahead would also fire `analyze_chapter` for ch=2 and skew the
    # call count. Coalescing of lookahead is verified separately below.
    app.dependency_overrides[get_settings] = lambda: _settings_with_lookahead(0)

    with TestClient(app) as client:
        book_id = _upload(client)

        results: list[tuple[int, list[tuple[str, str]]]] = []
        lock = threading.Lock()

        def fire():
            r = client.get(f"/api/books/{book_id}/chapters/1/meta")
            with lock:
                results.append((r.status_code, _parse_sse(r.text)))

        threads = [threading.Thread(target=fire) for _ in range(3)]
        # Start them as close together as possible so they collide on
        # the manager lock before any can finish.
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
            assert not t.is_alive(), "request hung"

    assert len(results) == 3
    for status, events in results:
        assert status == 200
        assert events[-1][0] == EVENT_META

    # The whole point: only one LLM call across 3 concurrent requests.
    assert counting_llm.call_count == 1, (
        f"expected 1 LLM call (coalesced), got {counting_llm.call_count}"
    )


def test_api_sse_error_event_on_failure(tmp_path):
    """LLM errors are reported via ``event: error``; the connection
    closes gracefully instead of HTTP-500-ing mid-stream."""
    failing_llm = _FailingLLM(message="boom")
    repo = BookRepository(tmp_path / "books")
    service = BookService(repo=repo, llm=failing_llm)
    manager = ChapterMetaStreamManager(service=service, heartbeat_interval_s=0.1)
    app = create_app()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_book_service] = lambda: service
    app.dependency_overrides[get_llm_client] = lambda: failing_llm
    app.dependency_overrides[get_chapter_meta_stream_manager] = lambda: manager

    with TestClient(app) as client:
        # Upload uses the LLM for chapter detection — supply the override
        # only for chapter analysis. Easier: use a stub for upload, then
        # patch the service to use the failing LLM.
        upload_stub = StubLLMClient()
        app.dependency_overrides[get_llm_client] = lambda: upload_stub
        service.llm = upload_stub
        book_id = _upload(client)

        # Now flip to the failing LLM for the chapter-analysis call.
        service.llm = failing_llm
        r = client.get(f"/api/books/{book_id}/chapters/1/meta")

    assert r.status_code == 200, "errors are sent in-band, not via HTTP status"
    events = _parse_sse(r.text)
    assert events[-1][0] == EVENT_ERROR
    payload = json.loads(events[-1][1])
    assert "boom" in payload["detail"]


def test_api_sse_lookahead_after_meta(service_client):
    """Lookahead fires after the current chapter's meta event so the
    next request hits the cache."""
    _, repo, client, manager = service_client
    book_id = _upload(client)
    assert not repo.chapter_meta_path(book_id, 2).exists()

    r = client.get(f"/api/books/{book_id}/chapters/1/meta")
    assert r.status_code == 200
    events = _parse_sse(r.text)
    assert events[-1][0] == EVENT_META

    # Lookahead is a fire-and-forget asyncio.create_task. Wait briefly
    # for it to land on disk — the stub LLM finishes well under 1 s.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if repo.chapter_meta_path(book_id, 2).exists():
            break
        time.sleep(0.05)
    assert repo.chapter_meta_path(book_id, 2).exists()


def test_api_unknown_book_404(service_client):
    """Pre-stream 404 — clients can distinguish bad URL from LLM failure."""
    _, _, client, _ = service_client
    r = client.get("/api/books/doesnotexist/chapters/1/meta")
    assert r.status_code == 404


def test_api_unknown_chapter_404(service_client):
    _, _, client, _ = service_client
    book_id = _upload(client)
    r = client.get(f"/api/books/{book_id}/chapters/99/meta")
    assert r.status_code == 404


def test_no_x_characters_version_header(service_client):
    """The deprecated header stays gone."""
    _, _, client, _ = service_client
    book_id = _upload(client)
    r = client.get(f"/api/books/{book_id}/chapters/1/meta")
    assert r.status_code == 200
    assert "x-characters-version" not in {h.lower() for h in r.headers}


# ---------------------------------------------------------------------------
# manager-level idempotency test (no HTTP — closer to a unit test)
# ---------------------------------------------------------------------------


def test_manager_idempotency_via_asyncio(tmp_path):
    """Direct manager test: 3 concurrent ``stream`` calls for the same
    chapter — only one LLM run, all three see the same final meta."""
    counting_llm = _CountingLLM(StubLLMClient(), delay_s=0.3)
    repo = BookRepository(tmp_path / "books")
    service = BookService(repo=repo, llm=counting_llm)

    # Need a real book for the service to chew on.
    parsed = service.ingest_txt(SAMPLE_TXT.encode("utf-8"), "t.txt")
    book_id = parsed.book_id

    manager = ChapterMetaStreamManager(service=service, heartbeat_interval_s=0.05)

    async def consume_one() -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        async for event in manager.stream(book_id, 1):
            out.append(event)
        return out

    async def main() -> list[list[tuple[str, str]]]:
        return await asyncio.gather(consume_one(), consume_one(), consume_one())

    results = asyncio.run(main())
    assert counting_llm.call_count == 1, (
        f"expected 1 LLM call, got {counting_llm.call_count}"
    )
    for events in results:
        assert events[-1][0] == EVENT_META


# ---------------------------------------------------------------------------
# LLM test doubles
# ---------------------------------------------------------------------------


class _SlowLLM:
    """Wraps a real stub but sleeps in every chapter-analysis sub-call
    to provoke heartbeats. The pipeline runs the calls sequentially —
    classify → profile_new → segment — so each delay adds to total
    wall time."""

    def __init__(self, inner: StubLLMClient, delay_s: float):
        self._inner = inner
        self._delay_s = delay_s

    def detect_chapters(self, *args, **kwargs):
        return self._inner.detect_chapters(*args, **kwargs)

    def classify_chapter_characters(
        self, chapter_text: str, known_characters: list[Character],
    ) -> ClassifiedCharacters:
        time.sleep(self._delay_s)
        return self._inner.classify_chapter_characters(chapter_text, known_characters)

    def profile_new_characters(
        self, name_to_contexts: dict[str, list[str]],
        known_characters: list[Character],
    ) -> list[Character]:
        time.sleep(self._delay_s)
        return self._inner.profile_new_characters(name_to_contexts, known_characters)

    def segment_chapter(
        self, chapter_text: str, known_characters: list[Character],
    ) -> list[AnalyzedSentence]:
        time.sleep(self._delay_s)
        return self._inner.segment_chapter(chapter_text, known_characters)


class _CountingLLM:
    """Like _SlowLLM but also counts logical chapter-analysis calls so
    we can assert coalescing behaviour. Counts each chapter analysis
    once (on ``classify_chapter_characters`` — it's the entry point
    of the pipeline and fires exactly once per chapter)."""

    def __init__(self, inner: StubLLMClient, delay_s: float):
        self._inner = inner
        self._delay_s = delay_s
        self.call_count = 0
        self._lock = threading.Lock()

    def detect_chapters(self, *args, **kwargs):
        return self._inner.detect_chapters(*args, **kwargs)

    def classify_chapter_characters(
        self, chapter_text: str, known_characters: list[Character],
    ) -> ClassifiedCharacters:
        with self._lock:
            self.call_count += 1
        time.sleep(self._delay_s)
        return self._inner.classify_chapter_characters(chapter_text, known_characters)

    def profile_new_characters(
        self, name_to_contexts: dict[str, list[str]],
        known_characters: list[Character],
    ) -> list[Character]:
        time.sleep(self._delay_s)
        return self._inner.profile_new_characters(name_to_contexts, known_characters)

    def segment_chapter(
        self, chapter_text: str, known_characters: list[Character],
    ) -> list[AnalyzedSentence]:
        time.sleep(self._delay_s)
        return self._inner.segment_chapter(chapter_text, known_characters)


class _FailingLLM:
    """Chapter analysis raises; chapter detection returns a no-op
    (the upload path uses the stub via the test harness, not this).
    Failure is injected at the very first analysis call so the failure
    path doesn't depend on which sub-call we use."""

    def __init__(self, message: str):
        self._message = message

    def detect_chapters(self, *args, **kwargs):
        from app.core.models import ChapterDetection
        return ChapterDetection()

    def classify_chapter_characters(
        self, chapter_text: str, known_characters: list[Character],
    ) -> ClassifiedCharacters:
        raise RuntimeError(self._message)

    def profile_new_characters(
        self, name_to_contexts: dict[str, list[str]],
        known_characters: list[Character],
    ) -> list[Character]:
        raise RuntimeError(self._message)

    def segment_chapter(
        self, chapter_text: str, known_characters: list[Character],
    ) -> list[AnalyzedSentence]:
        raise RuntimeError(self._message)
