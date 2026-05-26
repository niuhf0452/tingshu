"""Process-wide lock that serializes MLX / Metal GPU work across
multiple MLX-backed services.

The Apple GPU command buffer isn't safe for concurrent submission from
separate threads. If we ever run a local MLX LLM alongside Qwen3-TTS,
both must take turns or Metal aborts the process with:

    AGXG16GFamilyCommandBuffer ... A command encoder is already encoding
    to this command buffer

Every MLX-backed backend must hold ``gpu_guard()`` around its
``generate()`` call. MLX inference saturates the GPU internally, so the
lock costs nothing in throughput.

**Note on per-model thread affinity**: MLX models are thread-bound at
load time — the model can only be invoked from the thread that loaded
it (see ``Qwen3TTSClient`` for the dedicated-thread executor pattern).
The lock here serialises *between* MLX-backed services, but does not
solve "wrong thread for this model"; that requires structuring each
backend so all its calls land on the same thread.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

_LOCK = threading.Lock()


def gpu_lock() -> threading.Lock:
    return _LOCK


@contextmanager
def gpu_guard() -> Iterator[None]:
    """``with gpu_guard():`` — hold the shared MLX GPU lock for the block."""
    with _LOCK:
        yield
