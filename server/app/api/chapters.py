"""Chapter metadata endpoint (SSE).

The endpoint streams Server-Sent Events because the underlying LLM
analysis can take 30-60s, well past most HTTP read timeouts. Heartbeats
keep the connection alive; the final ``meta`` event delivers the result.

Idempotency: concurrent requests for the same ``(book_id, chapter_id)``
attach to the same in-flight job — see ``chapter_meta_stream`` for the
coalescing logic.

Lookahead: after the current chapter's stream emits its final event,
schedule background generation of the next N chapters via
``manager.warm`` so the next request hits the disk cache.
"""
from __future__ import annotations

import logging
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from starlette.responses import StreamingResponse

from ..config import Settings, get_settings
from ..core.models import BookMeta
from ..core.storage import BookRepository
from .chapter_meta_stream import (
    EVENT_ERROR,
    EVENT_META,
    ChapterMetaStreamManager,
    format_sse,
)
from .deps import get_chapter_meta_stream_manager, get_repository


log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/books", tags=["chapters"])


@router.get("/{book_id}/chapters/{chapter_id}/meta")
async def get_chapter_meta(
    book_id: str,
    chapter_id: int,
    repo: Annotated[BookRepository, Depends(get_repository)],
    manager: Annotated[ChapterMetaStreamManager, Depends(get_chapter_meta_stream_manager)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> StreamingResponse:
    """SSE endpoint. Yields:

    - ``event: heartbeat`` every ~5s while LLM analysis is running.
      ``data`` is ``{"elapsed_s": N, "stage": "analyzing"}``.
    - ``event: meta`` (terminal, on success). ``data`` is the full
      ``ChapterMeta`` JSON.
    - ``event: error`` (terminal, on failure). ``data`` is
      ``{"detail": "..."}``.

    404 (non-streaming) when the book or chapter doesn't exist —
    returned eagerly so clients can distinguish "wrong URL" from "LLM
    failed".
    """
    if not repo.exists(book_id):
        raise HTTPException(status_code=404, detail="book not found")

    meta = repo.load_meta(book_id)
    if not _has_chapter(meta, chapter_id):
        raise HTTPException(status_code=404, detail="chapter not found")

    cache_hit = repo.load_chapter_meta(book_id, chapter_id) is not None
    log.info(
        "chapter meta SSE request: book=%s ch=%d cache=%s",
        book_id, chapter_id, "hit" if cache_hit else "miss",
    )

    lookahead = max(0, settings.preprocess.lookahead_chapters)
    lookahead_ids = [
        chapter_id + offset
        for offset in range(1, lookahead + 1)
        if _has_chapter(meta, chapter_id + offset)
    ]

    async def event_stream() -> AsyncIterator[bytes]:
        terminal_seen = False
        async for event_name, data in manager.stream(book_id, chapter_id):
            yield format_sse(event_name, data)
            if event_name in (EVENT_META, EVENT_ERROR):
                terminal_seen = True
        if terminal_seen and lookahead_ids:
            # Schedule lookahead AFTER the current chapter is delivered:
            # cheap if the user keeps reading, free if they don't (no
            # API spend on chapters they never reach).
            log.info(
                "lookahead scheduled: book=%s after_ch=%d ids=%s",
                book_id, chapter_id, lookahead_ids,
            )
            manager.warm(book_id, lookahead_ids)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # Disable proxy buffering — without this, intermediaries can
            # batch heartbeats and defeat the whole point of streaming.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _has_chapter(meta: BookMeta, chapter_id: int) -> bool:
    return any(c.id == chapter_id for c in meta.chapters)
