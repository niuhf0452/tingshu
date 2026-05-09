"""Keep the host machine awake while the TTS server is running.

A headless server is useless if the host suspends mid-request:
- iOS clients time out and surface scary error alerts.
- In-flight Qwen3-TTS jobs get interrupted and the MLX GPU has to
  re-warm on the next request (~3-8 s lost).
- Auto-prefetched audio sitting in the threadpool is dropped.

Implementation: ``wakepy.keep.running`` is a cross-platform power-
assertion library — IOKit on macOS, systemd-inhibit / D-Bus on Linux,
SetThreadExecutionState on Windows — so the same call works on whatever
host the homelab actually runs on.

Wired as a context manager from the FastAPI lifespan so the assertion
is **explicitly released** on shutdown (uvicorn ^C / SIGTERM /
exception). We don't rely on OS-side cleanup as the only release path —
explicit teardown means tools like ``pmset -g assertions`` show our
entry disappearing the instant we stop, which makes leaks
diagnose-at-a-glance.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from wakepy import keep


log = logging.getLogger(__name__)


@contextmanager
def prevent_system_sleep() -> Iterator[None]:
    """Hold a ``wakepy.keep.running`` lock for the duration of the
    with-block.

    Fail-soft: any error from wakepy (no working method on the host,
    permission denied on D-Bus, etc.) degrades to a warning and the
    server still runs — losing sleep inhibition is not a fatal
    condition.

    Activation succeeded ↔ ``mode.active`` is True ↔ at least one
    backend method engaged. Logged with the chosen method name so
    deploys can confirm the right backend was picked.
    """
    try:
        cm = keep.running()
        mode = cm.__enter__()
    except Exception as exc:  # noqa: BLE001 — defensive, never fatal
        log.warning(
            "wakepy keep.running could not start: %s — "
            "system may sleep mid-request",
            exc,
        )
        yield
        return

    if mode.active:
        log.info(
            "wakepy keep.running active (method=%s) — "
            "system sleep blocked while server runs",
            mode.used_method,
        )
    else:
        log.warning(
            "wakepy keep.running activated no method — "
            "system may sleep mid-request",
        )

    try:
        yield
    finally:
        try:
            cm.__exit__(None, None, None)
            log.info("wakepy keep.running released — sleep inhibitor cleared")
        except Exception as exc:  # noqa: BLE001
            log.warning("wakepy keep.running release failed: %s", exc)
