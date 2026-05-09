"""In-memory job coalescer + SSE fan-out for chapter-meta generation.

Why this exists:

LLM-driven chapter analysis can routinely take 30-60s (longer for
unusually long chapters). HTTP clients with a 30-40s read timeout will
disconnect mid-analysis, then retry — and a naive endpoint would re-run
the LLM on every retry, multiplying cost and starving the actual user.

This module turns the chapter-meta endpoint into an SSE stream:

- Heartbeats every ``heartbeat_interval_s`` seconds keep the connection
  alive while the LLM call is running, so clients (and intermediaries)
  don't time out.
- The final ``meta`` (or ``error``) event delivers the result.
- Concurrent requests for the **same** ``(book_id, chapter_id)`` attach
  to the same in-flight job — only one LLM call runs.
- Client disconnects do **not** cancel the job. The result still ends
  up in the on-disk cache, so a retry hits the cache instantly.

Single-machine only — the in-flight map lives in this process. That's
fine for the current deployment (one Mac Mini); if we ever go
multi-process, swap the map for a shared store (Redis pubsub etc.).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from ..core.models import ChapterMeta
from ..core.service import BookService


log = logging.getLogger(__name__)

# SSE event names. These are the public API consumed by the iOS client.
EVENT_HEARTBEAT = "heartbeat"
EVENT_META = "meta"
EVENT_ERROR = "error"


@dataclass
class _Job:
    """One in-flight chapter-meta analysis. Multiple HTTP requests can
    observe the same job — they all wait on ``done`` and read
    ``result`` / ``error_message`` once it's set.
    """
    started_at: float
    task: asyncio.Task | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    result: ChapterMeta | None = None
    error_message: str | None = None


class ChapterMetaStreamManager:
    """Coalesce concurrent ``GET .../chapters/{n}/meta`` SSE requests onto
    a single LLM analysis per chapter.

    Public surface:

    - ``stream(book_id, chapter_id)`` — async generator yielding
      ``(event_name, data_json_str)`` tuples for an SSE endpoint.
    - ``warm(book_id, chapter_ids)`` — non-blocking lookahead trigger;
      dedups against in-flight jobs and the disk cache.

    Heartbeats are generated **per subscriber** (each request times out
    its own ``done.wait()``), so a slow consumer doesn't starve a fast
    one. Subscribers don't share a queue — they share the ``_Job``.
    """

    def __init__(
        self,
        service: BookService,
        heartbeat_interval_s: float = 5.0,
    ):
        self._service = service
        self._heartbeat_interval_s = heartbeat_interval_s
        self._jobs: dict[tuple[str, int], _Job] = {}
        # Brief-hold lock around the in-flight map. Released before
        # awaiting the job — heartbeat loops must not block the manager.
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # public surface
    # ------------------------------------------------------------------

    async def stream(
        self, book_id: str, chapter_id: int,
    ) -> AsyncIterator[tuple[str, str]]:
        """Yield SSE events for one HTTP request. See module docstring."""
        # Cache hit fast path — no lock, no job.
        if (cached := self._load_cached(book_id, chapter_id)) is not None:
            yield (EVENT_META, _serialize_for_api(cached))
            return

        async with self._lock:
            # Re-check under lock: a concurrent job may have written the
            # file between the cheap check above and now.
            if (cached := self._load_cached(book_id, chapter_id)) is not None:
                yield (EVENT_META, _serialize_for_api(cached))
                return

            key = (book_id, chapter_id)
            job = self._jobs.get(key)
            if job is None:
                job = self._spawn_job(book_id, chapter_id)
                self._jobs[key] = job

        async for event in self._consume(job):
            yield event

    def warm(self, book_id: str, chapter_ids: list[int]) -> None:
        """Schedule background analysis for chapters not yet cached.
        Non-blocking. Idempotent — a chapter already in flight or on
        disk is skipped.
        """
        for cid in chapter_ids:
            if self._service.chapter_meta_exists(book_id, cid):
                continue
            # Detached task: lookahead survives the originating request.
            asyncio.create_task(self._ensure_job_started(book_id, cid))

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _load_cached(self, book_id: str, chapter_id: int) -> ChapterMeta | None:
        # Wraps the repo lookup so the lock-recheck above stays compact.
        return self._service.repo.load_chapter_meta(book_id, chapter_id)

    async def _ensure_job_started(self, book_id: str, chapter_id: int) -> None:
        # Re-check cache once the task wakes — cheap and avoids spawning
        # a job whose work was already done by a concurrent stream.
        if self._service.chapter_meta_exists(book_id, chapter_id):
            return
        async with self._lock:
            if self._service.chapter_meta_exists(book_id, chapter_id):
                return
            key = (book_id, chapter_id)
            if key in self._jobs:
                return
            self._jobs[key] = self._spawn_job(book_id, chapter_id)

    def _spawn_job(self, book_id: str, chapter_id: int) -> _Job:
        """Create and launch a job. Caller MUST hold ``self._lock``
        and insert the returned job into ``self._jobs`` immediately —
        otherwise concurrent callers can race and spawn duplicates.
        """
        job = _Job(started_at=time.monotonic())

        async def run() -> None:
            try:
                # generate_chapter_meta is a synchronous, IO + CPU
                # blocking call (HTTP to DeepSeek + JSON parsing +
                # locate_sentences). Push it to the threadpool so the
                # event loop stays responsive for heartbeats and other
                # requests.
                meta = await asyncio.to_thread(
                    self._service.generate_chapter_meta,
                    book_id, chapter_id,
                )
                job.result = meta
            except asyncio.CancelledError:
                # Loop shutdown — don't try to wake subscribers (they're
                # being cancelled too) and don't touch the lock (also
                # being torn down). The threadpool task may have run to
                # completion in the background; the on-disk result is
                # the only thing that matters past this point.
                raise
            except BaseException as exc:
                job.error_message = str(exc) or exc.__class__.__name__
                log.exception(
                    "chapter meta generation failed: book=%s ch=%d",
                    book_id, chapter_id,
                )

            job.done.set()
            # Drop from the in-flight map so future requests check the
            # disk cache (which the successful path just wrote). On
            # error, the next request retries — we don't cache failures
            # to avoid permanent breakage from transients.
            async with self._lock:
                # Identity-check before removing: paranoia against a
                # spawn that replaced ours.
                if self._jobs.get((book_id, chapter_id)) is job:
                    self._jobs.pop((book_id, chapter_id), None)

        job.task = asyncio.create_task(run())
        return job

    async def _consume(
        self, job: _Job,
    ) -> AsyncIterator[tuple[str, str]]:
        """Wait for the job, emitting heartbeats every interval, then
        emit the final event."""
        while not job.done.is_set():
            try:
                await asyncio.wait_for(
                    job.done.wait(),
                    timeout=self._heartbeat_interval_s,
                )
            except asyncio.TimeoutError:
                elapsed = int(time.monotonic() - job.started_at)
                yield (
                    EVENT_HEARTBEAT,
                    json.dumps({"elapsed_s": elapsed, "stage": "analyzing"}),
                )

        if job.error_message is not None:
            yield (EVENT_ERROR, json.dumps({"detail": job.error_message}))
        elif job.result is not None:
            yield (EVENT_META, _serialize_for_api(job.result))
        else:
            # Should not happen — done.set() runs only after one of the
            # two is populated. Belt-and-suspenders error event.
            yield (
                EVENT_ERROR,
                json.dumps({"detail": "internal error: job done without result"}),
            )


def _serialize_for_api(meta: ChapterMeta) -> str:
    """Serialize ChapterMeta for the public API.

    The on-disk JSON keeps each sentence's verbatim text in
    ``Sentence.text`` for debugging — easy eyeballing of segmentation
    quality from `chapters/N.json`. The API response strips that field
    so the App still gets the lean positions-only payload it expects
    (and so the wire is small, since text is already in the chapter
    body the App downloaded with the book).
    """
    return meta.model_dump_json(
        exclude={"sentences": {"__all__": {"text"}}}
    )


def format_sse(event: str, data: str) -> bytes:
    """Encode one event in the SSE wire format. Each ``data`` line is
    prefixed; multi-line data is split per the SSE spec.

    Returns bytes (already utf-8 encoded) for direct streaming.
    """
    lines = [f"event: {event}"]
    for data_line in data.splitlines() or [""]:
        lines.append(f"data: {data_line}")
    lines.append("")  # terminating empty line
    lines.append("")
    return ("\n".join(lines)).encode("utf-8")
