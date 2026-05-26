"""TTS abstraction + service.

Two layers:

- ``TTSClient`` (protocol) — backend that knows how to turn
  ``(text, speaker_id, tone)`` into raw WAV/M4A bytes. Speed is **not**
  a parameter; backends always synthesize at 1.0x. Playback rate is
  applied client-side via ``AVAudioUnitTimePitch`` so a single audio
  file serves all speeds (cf. design discussion 2026-04-26).
- ``TTSService`` — consults the disk cache, falls through to the client
  on miss. Speaker resolution (character_id → Speaker) lives upstream
  in the API layer (``api/tts.py``) which uses ``core.voice.resolve_speaker``.

Cache responsibility sits in ``TTSService`` so the same cache hit /
miss path is exercised regardless of whether the backend is the silent
stub, Qwen3-TTS MLX, or a future remote TTS.
"""
from __future__ import annotations

import io
import logging
import time
import wave
from typing import Protocol

from ..core.enums import Tone
from ..core.models import Speaker
from ..core.tts_cache import TTSCache


log = logging.getLogger(__name__)


class TTSClient(Protocol):
    def synthesize(
        self,
        text: str,
        speaker_id: str,
        tone: Tone,
    ) -> bytes:
        """Return raw audio bytes for the given synthesis parameters,
        always at 1.0x speed."""


class NoSpeakerError(RuntimeError):
    """Raised when speaker resolution fails (e.g. empty library, missing
    narrator voice). The TTS endpoint surfaces this as HTTP 503."""


# 100 ms of silence at 24 kHz — the sample rate Qwen3-TTS emits — packaged
# as a valid 16-bit mono WAV so clients can decode with the same codec
# path as real TTS output. Memoised; the payload is ~5 KB.
_SILENCE_SAMPLE_RATE = 24000
_SILENCE_DURATION_MS = 100
_silence_cache: bytes | None = None


def _silence_wav() -> bytes:
    global _silence_cache
    if _silence_cache is None:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(_SILENCE_SAMPLE_RATE)
            n_samples = _SILENCE_SAMPLE_RATE * _SILENCE_DURATION_MS // 1000
            w.writeframes(b"\x00\x00" * n_samples)
        _silence_cache = buf.getvalue()
    return _silence_cache


# Punctuation replacements for Qwen3-TTS zero-shot mode. Each entry
# must be backed by an observed failure — see scripts/test_punctuation.py
# for the bleed-detection benchmark. **Don't add speculative entries**;
# only marks that demonstrably cause the model to bleed ref-text or
# misread the input.
#
# Confirmed:
# - U+2026 ×2 ("……") at end of sentence → ref-text bleed
#   (verified in production 2026-04-26 — sentence ending with `……`
#   started with the tail of the voice clone's reference text).
_TTS_PUNCT_REPLACEMENTS = {
    "……": "，",
}


def normalize_for_tts(text: str) -> str:
    """Strip / replace punctuation that destabilises Qwen3-TTS.

    Pure-text helper so unit tests can pin behaviour without going
    through the full TTSService. Applied at the cache-key boundary so
    variants like ``"X……"`` and ``"X，"`` share the same cached audio.
    """
    for src, dst in _TTS_PUNCT_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text


def _has_speakable_content(text: str) -> bool:
    """True if the text contains at least one character plausibly producing
    speech. Qwen3-TTS silently emits zero frames on pure-punctuation /
    whitespace / combining-mark inputs, which would otherwise bubble up
    as 500 Internal Server Error.

    Covers CJK ideographs + common scripts; punctuation, whitespace, and
    symbols don't count.
    """
    for ch in text:
        if ch.isalnum():
            return True
        if "一" <= ch <= "鿿":  # CJK Unified Ideographs
            return True
        if "㐀" <= ch <= "䶿":  # CJK Extension A
            return True
    return False


class TTSService:
    def __init__(
        self,
        client: TTSClient,
        cache: TTSCache,
    ):
        self.client = client
        self.cache = cache

    def synthesize(
        self,
        *,
        text: str,
        speaker: Speaker,
        tone: Tone,
    ) -> bytes:
        """Synthesize ``text`` with ``speaker``'s voice, at 1.0x. Cache
        key = (speaker_id, normalized_text) — tone is **not** in the
        key (per 2026-04-26 decision: LLM tone tags drift between
        re-analyses; prefer cache hit rate over per-tone fidelity).
        The first synthesized version of a (speaker, text) pair wins
        forever.
        """
        # Normalize before everything else so the cache, the speakable-
        # content check, and the model all see the same string. Strips
        # punctuation that destabilises Qwen3-TTS (notably U+2026 `…`,
        # which can make the model bleed reference-clip text into the
        # output — see ``_TTS_PUNCT_REPLACEMENTS``).
        text = normalize_for_tts(text)

        # Short-circuit: pure-punctuation / whitespace inputs trip Qwen3-TTS
        # into emitting zero frames, which would bubble up as a 500. Return
        # a tiny silence clip — the client plays it and advances.
        if not _has_speakable_content(text):
            log.info(
                "tts skipped (no speakable content): chars=%d speaker=%s text=%r",
                len(text), speaker.speaker_id, text[:40],
            )
            return _silence_wav()

        cached = self.cache.get(speaker.speaker_id, text)
        if cached is not None:
            log.info(
                "tts cache hit: chars=%d speaker=%s tone=%s bytes=%d",
                len(text), speaker.speaker_id, tone.value, len(cached),
            )
            return cached

        log.info(
            "tts synth start: chars=%d speaker=%s (gender=%s age=%s) tone=%s",
            len(text), speaker.speaker_id,
            speaker.gender.value, speaker.age.value, tone.value,
        )
        t0 = time.monotonic()
        try:
            audio = self.client.synthesize(
                text=text, speaker_id=speaker.speaker_id, tone=tone,
            )
        except RuntimeError as exc:
            # Qwen3-TTS occasionally yields zero audio chunks for inputs
            # it can't speak (decoration that survived
            # ``_has_speakable_content``, corner-case token sequences,
            # numeric tables, etc.). The backend already retries such a
            # failure several times internally (see ``Qwen3TTSClient``);
            # reaching here means every attempt still produced no audio.
            # Substitute silence so the player can advance instead of
            # erroring out the whole sentence — but **do not cache it**:
            # a transient backend failure (mlx-audio import error, model
            # load failure, GPU hiccup) would otherwise permanently
            # poison every (speaker, text) attempted during the outage,
            # forcing a manual cache wipe to recover. Re-synthesis on
            # the next request is cheap compared to that footgun.
            log.warning(
                "tts backend returned no audio; substituting silence (not cached): "
                "speaker=%s err=%s text=%r",
                speaker.speaker_id, exc, text[:120],
            )
            return _silence_wav()
        elapsed = time.monotonic() - t0
        self.cache.put(speaker.speaker_id, text, audio)
        log.info(
            "tts synth done: chars=%d speaker=%s bytes=%d wall=%.1fs",
            len(text), speaker.speaker_id, len(audio), elapsed,
        )
        return audio
