"""Silent-WAV TTS backend used for tests and local dev without MLX.

Produces a short silent 16 kHz mono WAV. Duration is a linear function of
input length so the iOS client's playback/preload timing code can be
exercised against realistic durations.
"""
from __future__ import annotations

import io
import wave

from ..core.enums import Tone


SAMPLE_RATE = 16000
# ~15 characters per second roughly matches Chinese TTS pacing at 1.0x speed.
CHARS_PER_SECOND = 15.0
MIN_DURATION = 0.3
MAX_DURATION = 30.0


class StubTTSClient:
    """Returns a valid silent WAV sized proportional to input length."""

    def synthesize(
        self,
        text: str,
        speaker_id: str,
        tone: Tone,
    ) -> bytes:
        seconds = len(text) / CHARS_PER_SECOND
        seconds = max(MIN_DURATION, min(MAX_DURATION, seconds))
        return _silent_wav(seconds)


def _silent_wav(duration_sec: float, sample_rate: int = SAMPLE_RATE) -> bytes:
    num_samples = int(duration_sec * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * num_samples)
    return buf.getvalue()
