"""Quality control for synthesized TTS clips.

Detects the one Qwen3-TTS failure mode that matters in practice: the
model finishes the real sentence, then fails to terminate and trails
off into a long span of garbage — noise, stuttering / repetition, dead
silence or a fade to near-silence, usually mixed together.

What unifies that garbage span, whatever it is made of, is its texture:
clean narration is **dense and continuous**, the garbage span is
**choppy** — sparse energy bursts separated by silence. So detection
measures the local *temporal density* of voiced frames, not any
per-frame spectral property (harmonicity / flatness were tried and do
not separate — the garbage can itself be periodic, e.g. a stutter loop
or a held tone)::

    voiced[f]   = rms[f] > VOICED_REL_FLOOR × (95th-pct rms)
    density[f]  = fraction of voiced frames in a ±DENS_WIN_S/2 window
    bad_span_s  = longest contiguous run with density < DENS_THR

Calibrated offline against user-confirmed clips (the calibration tool is
``scripts/detect_tts_defects.py``): every cache clip with
``bad_span_s ≥ 5 s`` was a real defect, with no false positives. Below
~5 s the metric cannot tell a short defect from an ordinary pause, so
only the gross "failed to terminate" failure is flagged — but that is
the high-impact one, where the whole sentence is ruined.

Pure-numpy on purpose: this runs inside the synthesis path, so it must
not pull in a heavy audio stack.
"""
from __future__ import annotations

import numpy as np


# Frame geometry for the 24 kHz Qwen3-TTS output.
_HOP = 256
_RMS_WIN = 512
# A frame counts as "voiced" when its RMS clears this fraction of the
# clip's 95th-percentile RMS (a robust, level-adaptive loudness ref).
_VOICED_REL_FLOOR = 0.15
_DENS_WIN_S = 1.5          # window the voiced-density is averaged over
_DENS_THR = 0.45           # local density below this ⇒ a "garbage" frame
_FULLY_SILENT_PEAK = 1e-3  # 95th-pct RMS below this ⇒ digital silence

# A clip whose longest garbage span exceeds this many seconds is treated
# as defective. See the module docstring for how this was calibrated.
BAD_SPAN_MAX_S = 5.0


def longest_bad_span(samples: np.ndarray, sample_rate: int) -> float:
    """Longest low-voiced-density span, in seconds — the defect metric.

    ``0.0`` for a clean clip; large for one that trailed off into
    garbage. ``samples`` is a mono float waveform (any shape, flattened).
    """
    y = np.asarray(samples, dtype=np.float32).reshape(-1)
    if y.size < _RMS_WIN or sample_rate <= 0:
        return 0.0

    rms = _frame_rms(y)
    peak = float(np.percentile(rms, 95))
    sec_per_frame = _HOP / sample_rate

    if peak < _FULLY_SILENT_PEAK:
        return y.size / sample_rate  # the whole clip is digital silence

    voiced = (rms > max(_VOICED_REL_FLOOR * peak, 1e-4)).astype(np.float32)
    win = max(1, int(_DENS_WIN_S / sec_per_frame))
    density = np.convolve(voiced, np.ones(win) / win, mode="same")

    low = density < _DENS_THR
    if not low.any():
        return 0.0
    # Longest contiguous run of ``low``.
    edges = np.diff(np.concatenate(([0], low.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return float((ends - starts).max() * sec_per_frame)


def is_defective(samples: np.ndarray, sample_rate: int) -> bool:
    """True if the clip's longest garbage span exceeds ``BAD_SPAN_MAX_S``."""
    return longest_bad_span(samples, sample_rate) > BAD_SPAN_MAX_S


def _frame_rms(y: np.ndarray) -> np.ndarray:
    """Per-frame RMS energy (window ``_RMS_WIN``, hop ``_HOP``)."""
    n_frames = 1 + (y.size - _RMS_WIN) // _HOP
    idx = np.arange(_RMS_WIN)[None, :] + _HOP * np.arange(n_frames)[:, None]
    frames = y[idx]
    return np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
