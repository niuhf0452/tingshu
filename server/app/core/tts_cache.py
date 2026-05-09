"""File-system cache for synthesized audio.

Key = SHA1 of ``speaker_id || text``. Speed is intentionally **excluded**
— playback rate is applied client-side via ``AVAudioUnitTimePitch`` so a
single audio file serves all speeds. Tone is also excluded by design (per
2026-04-26 decision): the same (speaker, text) pair always returns
whatever was first synthesized for it. Cross-tone variation gets lost,
but cache hit rate goes up significantly because LLM analyses don't
guarantee identical tone tags across re-runs.

Values live directly on disk as ``<root>/<key>.m4a`` (AAC 48 kbps mono
@ 24 kHz, see ``app/services/tts_qwen3.py`` for the encode pipeline) —
no manifest, no eviction policy at this layer. Client-side caching has
its own lifecycle per docs §3.6.1; server-side is just a compute reuse
layer.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path


log = logging.getLogger(__name__)


class TTSCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    # --- key ---

    @staticmethod
    def cache_key(speaker_id: str, text: str) -> str:
        payload = f"{speaker_id}||{text}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.m4a"

    # --- ops ---

    def get(self, speaker_id: str, text: str) -> bytes | None:
        path = self._path(self.cache_key(speaker_id, text))
        if not path.exists():
            return None
        return path.read_bytes()

    def put(self, speaker_id: str, text: str, data: bytes) -> None:
        key = self.cache_key(speaker_id, text)
        path = self._path(key)
        # Atomic-ish write: write to a temp file then rename so a concurrent
        # reader never sees a truncated audio file.
        tmp = path.with_suffix(".m4a.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
