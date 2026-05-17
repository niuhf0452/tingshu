"""Background sweeper that keeps the TTS cache directory under a size cap.

The TTS cache (``app/core/tts_cache.py``) has no eviction policy of its
own — it grows unbounded as new ``(speaker, text)`` pairs are
synthesized. This janitor runs as an asyncio task for the lifetime of
the app: every ``interval_seconds`` it checks the cache directory size
and, when it exceeds the cap, deletes the oldest files (by modification
time — an approximate LRU) until it's back under the cap.

The actual scan + delete is ``TTSCache.evict_to_limit``; this module is
just the scheduling loop. The filesystem work is pushed to a worker
thread so a large cache doesn't block the event loop.
"""
from __future__ import annotations

import asyncio
import logging

from starlette.concurrency import run_in_threadpool

from .tts_cache import TTSCache


log = logging.getLogger(__name__)


async def run_cache_janitor(
    cache: TTSCache, max_bytes: int, interval_seconds: float,
) -> None:
    """Sweep ``cache`` every ``interval_seconds``, evicting oldest files
    whenever it exceeds ``max_bytes``.

    Runs until cancelled (the app lifespan cancels it on shutdown). A
    transient sweep failure is logged and swallowed so the loop keeps
    running; only cancellation ends it.
    """
    log.info(
        "tts cache janitor started: cap=%d MB, sweep every %.0fs",
        max_bytes // (1024 * 1024), interval_seconds,
    )
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            removed, freed = await run_in_threadpool(
                cache.evict_to_limit, max_bytes,
            )
            if removed:
                log.info(
                    "tts cache janitor: evicted %d file(s), freed %.1f MB",
                    removed, freed / (1024 * 1024),
                )
        except asyncio.CancelledError:
            log.info("tts cache janitor stopped")
            raise
        except Exception:
            # A transient FS error must not kill the loop — log and
            # retry on the next tick.
            log.exception("tts cache janitor sweep failed")
