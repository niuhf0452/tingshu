"""Process-wide lock that serializes all MLX / Metal GPU work.

The Apple GPU command buffer isn't safe for concurrent submission from
separate threads. MLX (Qwen3 LLM) and mlx-audio (Qwen3-TTS) run in
different Python threads — FastAPI's threadpool for TTS, our background
``ThreadPoolExecutor`` for character profile analysis, and the request
thread for chapter detection / analyze_chapter. Without shared
serialization, two threads can each submit Metal commands simultaneously
and trigger assertions like:

    AGXG16GFamilyCommandBuffer ... A command encoder is already encoding
    to this command buffer

which abort the whole process. Both the LLM and TTS backends MUST hold
this lock around their ``generate()`` call.

MLX inference already serialises internally per model anyway (one call
saturates the GPU), so a global lock here costs nothing in throughput.
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
