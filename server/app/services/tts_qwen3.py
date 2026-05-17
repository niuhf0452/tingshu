"""Qwen3-TTS 0.6B (MLX) backend — production TTS for Apple Silicon.

Implements the ``TTSClient`` protocol; wired via ``tts.provider=qwen3_tts``
in config.

Implementation notes:

- **MLX native**: uses Apple's MLX framework for GPU/ANE acceleration.
  Model load + inference goes through ``mlx_audio.tts.utils.load_model``.

- **Speaker ID scheme** (engine-independent; the same speakers.json can
  be reused across any zero-shot TTS backend):

    ``zs:<prompt_id>``
        Voice cloning from ``prompts_dir/<prompt_id>.wav`` + ``.txt``.
        Fed into Qwen3-TTS's ``ref_audio`` + ``ref_text`` parameters of
        ``model.generate()``.

    ``preset:<speaker_name>``
        Use a built-in preset speaker (Qwen3-TTS ships its own).

- **Tone**: mapped to Qwen3-TTS's ``instruct`` parameter (natural-language
  style directive), which is more expressive than text-prefix prompting.

- **Output**: M4A (AAC) container, 48 kbps mono at the model's native
  24 kHz. WAV is the intermediate before the AAC encode pass — chosen
  over raw WAV output because a typical 5-15s sentence's WAV is
  300-700 KB while the same audio at 48 kbps AAC is 30-100 KB
  (~7-9× smaller, perceptually transparent for speech). Cache layer is
  format-agnostic; only the on-disk file extension changes (.wav → .m4a).
  Encode adds ~30ms (subprocess afconvert) — < 1% of the ~3s synthesis
  cost; trivial overhead for the size win.

- **Concurrency**: MLX inference holds the GPU; serialise via the
  process-wide ``mlx_gpu.gpu_guard`` so LLM and TTS take turns on Metal.
  Endpoint already uses ``run_in_threadpool`` so this doesn't block the
  asyncio loop.

- **Quality control**: every clip is checked by ``audio_qc`` for the
  "model failed to terminate → trailing garbage" defect; a defective
  clip is re-synthesized (sampling is stochastic, so a retry usually
  recovers), up to ``_MAX_SYNTH_ATTEMPTS`` attempts. See
  ``synthesize``.
"""
from __future__ import annotations

import io
import logging
import os
import re
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
from mlx_audio.tts.generate import generate_audio

from ..core.enums import Tone
from . import audio_qc
from .mlx_gpu import gpu_guard


log = logging.getLogger(__name__)


# Tone → Qwen3-TTS instruct natural-language directive (more expressive
# than text-prefix prepending and what the model was trained for).
_TONE_INSTRUCTS: dict[Tone, str] = {
    Tone.NEUTRAL: "",
    Tone.HAPPY: "请用开心、愉悦的语气朗读",
    Tone.SAD: "请用悲伤、低沉的语气朗读",
    Tone.ANGRY: "请用愤怒、激烈的语气朗读",
    Tone.FEARFUL: "请用紧张、恐惧的语气朗读",
    Tone.SURPRISED: "请用惊讶、意外的语气朗读",
    Tone.GENTLE: "请用温柔、轻声的语气朗读",
    Tone.SERIOUS: "请用严肃、郑重的语气朗读",
    Tone.PLAYFUL: "请用戏谑、轻松的语气朗读",
    Tone.WHISPER: "请用压低声音、低语的语气朗读",
}

OUTPUT_SAMPLE_RATE = 24000  # Qwen3-TTS native output rate
# AAC bitrate for the encode pass. 48 kbps mono at 24 kHz is comfortably
# above the perceptual transparency floor for speech (~32 kbps). Going
# higher (e.g. 64 kbps) gains nothing audible for narrated text.
OUTPUT_AAC_BITRATE = 48_000

# Quality control: a synthesized clip is checked with ``audio_qc`` and,
# if defective, re-synthesized. Sampling is stochastic (temperature 0.9)
# so a retry usually comes out clean. This is the **total** number of
# attempts, not extra retries.
_MAX_SYNTH_ATTEMPTS = 3

_SPEAKER_ID_RE = re.compile(r"^(zs|preset):(.+)$")


def parse_speaker_id(speaker_id: str) -> tuple[str, str]:
    m = _SPEAKER_ID_RE.match(speaker_id)
    if not m:
        raise ValueError(
            "speaker_id must be '<mode>:<target>' with mode in {zs, preset}, "
            f"got {speaker_id!r}"
        )
    return m.group(1), m.group(2)


def tone_instruct(tone: Tone) -> str:
    return _TONE_INSTRUCTS.get(tone, "")


class Qwen3TTSClient:
    def __init__(
        self,
        model_dir: str,
        prompts_dir: str | None = None,
        *,
        model: Any = None,
    ):
        """Load Qwen3-TTS MLX weights. Heavy (~2 GB); call once at startup.

        ``model`` is a test hatch — pass a stand-in that exposes the same
        ``generate()`` method as the real ``mlx_audio.tts.models.qwen3_tts.Model``.
        """
        if model is None:
            model_path = Path(model_dir).resolve()
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Qwen3-TTS model not found at {model_path}. "
                    "Download per the project README (Qwen3-TTS setup section) "
                    "before setting tts.provider=qwen3_tts."
                )
            try:
                from mlx_audio.tts.utils import load_model
            except ImportError as exc:  # pragma: no cover — depends on host env
                raise ImportError(
                    "mlx-audio not importable. `pip install mlx-audio` "
                    "(M-series Mac only)."
                ) from exc
            log.info("loading Qwen3-TTS from %s", model_path)
            t0 = time.monotonic()
            model = load_model(model_path)
            log.info("Qwen3-TTS loaded in %.1fs", time.monotonic() - t0)

        self._model = model
        self._prompts_dir = Path(prompts_dir).resolve() if prompts_dir else None
        # Per-prompt cache: prompt_id → (ref_audio_path_str, ref_text)
        self._prompt_cache: dict[str, tuple[str, str]] = {}

    def synthesize(
        self,
        text: str,
        speaker_id: str,
        tone: Tone,
    ) -> bytes:
        """Synthesize ``text`` at 1.0x (playback rate is applied client-
        side via ``AVAudioUnitTimePitch`` so the cache stays speed-free).

        Each clip is quality-checked with ``audio_qc``: Qwen3-TTS
        occasionally fails to terminate and trails off into a long span
        of garbage. Because sampling is stochastic (temperature 0.9), a
        re-synthesis usually comes out clean — so a defective clip is
        retried, up to ``_MAX_SYNTH_ATTEMPTS`` attempts in total. If
        every attempt is defective the least-bad one is returned (better
        than erroring the whole sentence). A zero-audio result still
        raises ``RuntimeError`` so ``TTSService`` can substitute silence.
        """
        mode, target = parse_speaker_id(speaker_id)
        instruct = tone_instruct(tone) or None

        def _generate():
            if mode == "zs":
                ref_audio_path, ref_text = self._load_prompt(target)
                return self._model.generate(
                    text=text, ref_audio=ref_audio_path, ref_text=ref_text,
                    instruct=instruct, speed=1.0, verbose=False,
                )
            if mode == "preset":
                return self._model.generate_custom_voice(
                    text=text, speaker=target, instruct=instruct, verbose=False,
                )
            raise ValueError(f"unknown speaker id mode: {mode}")  # pragma: no cover

        best_samples: np.ndarray | None = None
        best_span = float("inf")

        for attempt in range(1, _MAX_SYNTH_ATTEMPTS + 1):
            # ``gpu_guard`` serialises with the LLM on Metal — held only
            # for one generation, released between attempts.
            with gpu_guard():
                try:
                    samples = _collect_samples(_generate())
                except RuntimeError:
                    samples = None  # zero / no audio — a retry may recover

            span = (
                audio_qc.longest_bad_span(samples, OUTPUT_SAMPLE_RATE)
                if samples is not None else float("inf")
            )
            if span < best_span:
                best_span, best_samples = span, samples

            if samples is not None and span <= audio_qc.BAD_SPAN_MAX_S:
                if attempt > 1:
                    log.info(
                        "Qwen3-TTS clean on attempt %d/%d (bad_span=%.1fs)",
                        attempt, _MAX_SYNTH_ATTEMPTS, span,
                    )
                break

            log.warning(
                "Qwen3-TTS attempt %d/%d defective (%s) — %s; "
                "speaker=%s tone=%s text=%r",
                attempt, _MAX_SYNTH_ATTEMPTS,
                "no audio" if samples is None else f"bad_span={span:.1f}s",
                "retrying" if attempt < _MAX_SYNTH_ATTEMPTS else "giving up",
                speaker_id, tone.value, text[:120],
            )

        if best_samples is None:
            # Every attempt produced no audio — raise so TTSService
            # substitutes silence (the pre-existing zero-audio contract).
            raise RuntimeError(
                f"Qwen3-TTS produced no audio in {_MAX_SYNTH_ATTEMPTS} attempts"
            )
        if best_span > audio_qc.BAD_SPAN_MAX_S:
            log.warning(
                "Qwen3-TTS: all %d attempts defective; using least-bad "
                "(bad_span=%.1fs) speaker=%s text=%r",
                _MAX_SYNTH_ATTEMPTS, best_span, speaker_id, text[:120],
            )
        return _encode_aac(_samples_to_wav(best_samples))

    def _load_prompt(self, prompt_id: str) -> tuple[str, str]:
        cached = self._prompt_cache.get(prompt_id)
        if cached is not None:
            return cached
        if self._prompts_dir is None:
            raise ValueError(
                "tts.qwen3.prompts_dir must be configured to use zs:* speakers"
            )
        wav_path = self._prompts_dir / f"{prompt_id}.wav"
        txt_path = self._prompts_dir / f"{prompt_id}.txt"
        missing = [p for p in (wav_path, txt_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"voice clone prompt files missing for {prompt_id!r}: {missing}"
            )
        ref_text = txt_path.read_text(encoding="utf-8").strip()
        result = (str(wav_path), ref_text)
        self._prompt_cache[prompt_id] = result
        return result


def _collect_samples(generator) -> np.ndarray:
    """Collect MLX-audio's GenerationResult chunks → one concatenated
    float32 mono sample array.

    Raises ``RuntimeError`` if the generator yielded no audio (zero
    frames) — the caller decides whether to retry or substitute silence.
    """
    segments: list[Any] = []
    chunk_count = 0
    empty_audio_chunks = 0
    for chunk in generator:
        chunk_count += 1
        audio_arr = getattr(chunk, "audio", None)
        if audio_arr is None:
            empty_audio_chunks += 1
            continue
        np_audio = np.asarray(audio_arr).astype(np.float32, copy=False).flatten()
        if np_audio.size:
            segments.append(np_audio)
        else:
            empty_audio_chunks += 1

    if not segments:
        raise RuntimeError(
            f"Qwen3-TTS produced no audio output "
            f"(chunks={chunk_count} empty={empty_audio_chunks})"
        )
    return np.concatenate(segments)


def _samples_to_wav(samples: np.ndarray) -> bytes:
    """Float32 mono samples → 16-bit mono WAV bytes @ 24 kHz."""
    audio_int16 = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(OUTPUT_SAMPLE_RATE)
        w.writeframes(audio_int16.tobytes())
    return wav_buf.getvalue()


def _collect_audio_wav(generator) -> bytes:
    """Collect MLX-audio's GenerationResult chunks → 16-bit mono WAV bytes.

    Used by tooling that needs to **store** the audio for later
    re-loading (e.g. ``scripts/generate_voicedesign_voices.py`` writes
    voice-clone reference audio that the runtime reads back). WAV is
    losslessly decodable by every audio loader; AAC isn't (some loaders
    silently fail).
    """
    return _samples_to_wav(_collect_samples(generator))


def _encode_aac(wav_bytes: bytes) -> bytes:
    """Encode 16-bit mono WAV bytes → M4A (AAC) bytes via macOS afconvert.

    Bitrate is fixed at ``OUTPUT_AAC_BITRATE`` (48 kbps mono). Two-pass
    isn't worth it for short speech clips; the default CBR is already
    transparent at this bitrate for narrated text.

    Raises ``RuntimeError`` if afconvert isn't on PATH (non-macOS) or
    fails — caller may want to fall back to raw WAV.
    """
    with tempfile.TemporaryDirectory(prefix="tts-aac-") as td:
        wav_path = os.path.join(td, "in.wav")
        m4a_path = os.path.join(td, "out.m4a")
        with open(wav_path, "wb") as f:
            f.write(wav_bytes)
        try:
            subprocess.run(
                [
                    "afconvert",
                    "-f", "m4af",          # MPEG-4 Audio container (.m4a)
                    "-d", "aac",           # AAC codec
                    "-b", str(OUTPUT_AAC_BITRATE),  # bitrate in bits/sec
                    "-c", "1",             # mono channel layout
                    wav_path, m4a_path,
                ],
                check=True,
                capture_output=True,
                timeout=10,
            )
        except FileNotFoundError as exc:  # afconvert not present
            raise RuntimeError(
                "afconvert not found — AAC encoding requires macOS"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"afconvert failed: {exc.stderr.decode('utf-8', errors='replace')[:300]}"
            ) from exc
        with open(m4a_path, "rb") as f:
            return f.read()
