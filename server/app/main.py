"""FastAPI application entrypoint.

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from .api import books, chapters, characters, health, tts
from .api.auth import require_auth
from .config import get_settings
from .core.cache_janitor import run_cache_janitor
from .core.tts_cache import TTSCache
from .keepalive import prevent_system_sleep


log = logging.getLogger(__name__)


def _install_emergency_sigint() -> None:
    """On Ctrl-C, exit the process immediately — no graceful shutdown,
    no waiting on in-flight requests.

    Rationale: ``mlx_audio`` (Qwen3-TTS) runs long blocking native loops
    inside the threadpool. uvicorn's default SIGINT handler waits on
    those threads, which makes the dev server feel frozen for tens of
    seconds after ^C. In a dev / single-user context we'd rather drop
    in-flight TTS work (cache writes are atomic renames, so data
    integrity survives) and get the prompt back now.

    Production deploys that need graceful shutdown should use SIGTERM
    (which uvicorn still handles normally) via a process manager —
    this handler only replaces SIGINT.

    No-op when not running on the main thread — ``signal.signal`` only
    works from the main thread, and pytest spawns tests in workers.
    """
    if threading.current_thread() is not threading.main_thread():
        return

    def handler(signum, frame):  # type: ignore[no-untyped-def]
        print(
            "\n[ctrl-c — exiting immediately (in-flight TTS jobs dropped)]",
            file=sys.stderr, flush=True,
        )
        os._exit(130)  # 128 + SIGINT

    try:
        signal.signal(signal.SIGINT, handler)
    except ValueError:
        # Also raised when the import happens under an embedded interpreter
        # that has disabled signal handling. Safe to ignore.
        pass


def _configure_logging() -> None:
    """Surface INFO-level logs from ``app.*`` modules.

    Uvicorn configures its own access/error loggers but doesn't touch the
    root logger, so our ``logging.getLogger(__name__)`` calls would
    otherwise stay silent. We install a single StreamHandler on the
    ``app`` logger if no handlers exist yet — uvicorn's --log-level flag
    can still lower us to WARNING if desired.
    """
    app_logger = logging.getLogger("app")
    if app_logger.handlers:  # already configured (e.g. re-imported under --reload)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                          datefmt="%H:%M:%S")
    )
    app_logger.addHandler(handler)
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Ensure data dirs exist before any request handler touches them.
    settings.books_dir.mkdir(parents=True, exist_ok=True)
    settings.tts_cache_dir.mkdir(parents=True, exist_ok=True)

    # Chain an emergency ^C handler behind uvicorn's (installed at server
    # startup, before lifespan runs) — see _install_emergency_sigint docstring.
    _install_emergency_sigint()

    # Pin the host awake for the entire server lifetime; wakepy releases
    # the assertion when the `with` block exits (uvicorn shutdown), so
    # cleanup is explicit instead of relying on the OS reaping a dead
    # process. See app/keepalive.py for backend choice rationale.
    with prevent_system_sleep():
        # Eagerly construct the LLM client so misconfig (e.g. missing
        # DEEPSEEK_API_KEY) fails the server at startup with a clear stderr
        # message — instead of returning HTTP 500 on every request because
        # the lazily-cached factory keeps re-raising.
        #
        # Honour ``app.dependency_overrides`` so test fixtures that inject a
        # StubLLMClient bypass the real factory (which would need network /
        # local model weights / API keys).
        from .api.deps import get_llm_client
        factory = app.dependency_overrides.get(get_llm_client, get_llm_client)
        try:
            factory()
            log.info("LLM backend ready (provider=%s)", settings.llm.provider)
        except Exception as exc:
            print(
                f"\n[FATAL] LLM provider init failed: {exc}",
                file=sys.stderr, flush=True,
            )
            raise

        # Warm up the TTS backend so the first request doesn't block on weight
        # loading. Stub is a no-op; Qwen3-TTS MLX ~3-8 s.
        if settings.tts.provider.lower() == "qwen3_tts":
            from .api.deps import get_tts_service
            log.info("warming up TTS backend (%s)", settings.tts.provider)
            get_tts_service()

        # Periodic size-cap eviction for the TTS cache — it has no
        # eviction policy of its own (see app/core/tts_cache.py).
        janitor_task: asyncio.Task | None = None
        if settings.tts.cache_sweep_seconds > 0:
            janitor_task = asyncio.create_task(
                run_cache_janitor(
                    TTSCache(settings.tts_cache_dir),
                    max_bytes=settings.tts.cache_max_mb * 1024 * 1024,
                    interval_seconds=settings.tts.cache_sweep_seconds,
                )
            )

        try:
            yield
        finally:
            if janitor_task is not None:
                janitor_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await janitor_task


def create_app() -> FastAPI:
    _configure_logging()
    app = FastAPI(title="TingShu Server", version="0.1.0", lifespan=lifespan)
    # /health stays public so liveness probes don't need creds.
    app.include_router(health.router)
    protected = [Depends(require_auth)]
    app.include_router(books.router, dependencies=protected)
    app.include_router(chapters.router, dependencies=protected)
    app.include_router(characters.router, dependencies=protected)
    app.include_router(tts.router, dependencies=protected)
    return app


app = create_app()
