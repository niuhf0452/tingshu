"""Tests for TTS audio quality control + the synthesis retry loop."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from app.core.enums import Tone
from app.services import audio_qc
from app.services.tts_qwen3 import Qwen3TTSClient

SR = 24000


def _tone(seconds: float) -> np.ndarray:
    """A steady voiced-like tone — dense, continuous energy."""
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


# --- longest_bad_span / is_defective --------------------------------------


class TestLongestBadSpan:
    def test_clean_clip_has_no_bad_span(self):
        assert audio_qc.longest_bad_span(_tone(4.0), SR) < 0.5

    def test_trailing_garbage_is_flagged(self):
        # 3 s of speech then 12 s of dead air — the canonical defect.
        clip = np.concatenate([_tone(3.0), _silence(12.0)])
        span = audio_qc.longest_bad_span(clip, SR)
        assert span > 9.0
        assert audio_qc.is_defective(clip, SR)

    def test_moderate_gap_is_not_flagged(self):
        # A 3 s internal pause — anomalous-ish but below the 5 s cutoff.
        clip = np.concatenate([_tone(3.0), _silence(3.0), _tone(3.0)])
        span = audio_qc.longest_bad_span(clip, SR)
        assert 1.0 < span < audio_qc.BAD_SPAN_MAX_S
        assert not audio_qc.is_defective(clip, SR)

    def test_fully_silent_clip_is_flagged(self):
        clip = _silence(8.0)
        assert audio_qc.longest_bad_span(clip, SR) == pytest.approx(8.0, abs=0.1)
        assert audio_qc.is_defective(clip, SR)

    def test_clip_shorter_than_a_frame_is_safe(self):
        assert audio_qc.longest_bad_span(np.zeros(100, np.float32), SR) == 0.0

    def test_accepts_non_flat_input(self):
        clip = _tone(2.0).reshape(-1, 1)  # column vector
        assert audio_qc.longest_bad_span(clip, SR) < 0.5


# --- synthesis retry loop --------------------------------------------------


class _Chunk:
    def __init__(self, audio: np.ndarray):
        self.audio = audio


class _ScriptedModel:
    """Stand-in for the MLX model: returns a scripted sample array per
    ``generate`` call (the last entry repeats if calls outrun the script)."""

    def __init__(self, takes: list[np.ndarray]):
        self._takes = takes
        self.calls = 0

    def _emit(self):
        arr = self._takes[min(self.calls, len(self._takes) - 1)]
        self.calls += 1
        return iter([_Chunk(arr)])

    def generate(self, **_kw):
        return self._emit()

    def generate_custom_voice(self, **_kw):
        return self._emit()


_DEFECT = np.concatenate([_tone(3.0), _silence(12.0)])  # bad_span ~11 s
_CLEAN = _tone(4.0)                                     # bad_span ~0
_NO_AUDIO = np.array([], dtype=np.float32)

afconvert = pytest.mark.skipif(
    shutil.which("afconvert") is None, reason="afconvert (macOS) required",
)


@afconvert
class TestSynthesisRetry:
    def _client(self, takes):
        model = _ScriptedModel(takes)
        return Qwen3TTSClient(model_dir="unused", model=model), model

    def test_clean_first_attempt_no_retry(self):
        client, model = self._client([_CLEAN])
        audio = client.synthesize("文本", "preset:voiceA", Tone.NEUTRAL)
        assert model.calls == 1
        assert len(audio) > 0

    def test_defective_then_clean_retries_once(self):
        client, model = self._client([_DEFECT, _CLEAN])
        client.synthesize("文本", "preset:voiceA", Tone.NEUTRAL)
        assert model.calls == 2  # retried, stopped as soon as it was clean

    def test_all_attempts_defective_stops_at_max(self):
        client, model = self._client([_DEFECT, _DEFECT, _DEFECT])
        audio = client.synthesize("文本", "preset:voiceA", Tone.NEUTRAL)
        assert model.calls == 3  # capped, does not loop forever
        assert len(audio) > 0    # still returns the least-bad clip

    def test_zero_audio_attempt_is_retried(self):
        client, model = self._client([_NO_AUDIO, _CLEAN])
        client.synthesize("文本", "preset:voiceA", Tone.NEUTRAL)
        assert model.calls == 2

    def test_all_attempts_zero_audio_raises(self):
        client, model = self._client([_NO_AUDIO, _NO_AUDIO, _NO_AUDIO])
        with pytest.raises(RuntimeError):
            client.synthesize("文本", "preset:voiceA", Tone.NEUTRAL)
        assert model.calls == 3
