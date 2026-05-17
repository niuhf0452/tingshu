"""Visualise TTS cache audio: waveform + mel-spectrogram + RMS energy.

READ-ONLY diagnostic tool. Decodes one or more .m4a clips and writes a
PNG per clip (waveform on top, mel-spectrogram below, RMS energy strip
at the bottom — all sharing one time axis).

Usage (from server/):
    python3 scripts/viz_audio.py data/tts_cache/<hash>.m4a [more.m4a ...]
    python3 scripts/viz_audio.py --outdir scripts/output/viz  <files...>
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import librosa  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")  # noqa: E402  — headless, write PNG only
import matplotlib.pyplot as plt  # noqa: E402
import soundfile as sf  # noqa: E402

HOP = 256
N_FFT = 1024


def decode(path: Path) -> tuple[np.ndarray, int]:
    """m4a → mono float32 via afconvert (auto-detects container)."""
    with tempfile.TemporaryDirectory(prefix="viz-") as td:
        wav = Path(td) / "o.wav"
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16", str(path), str(wav)],
            check=True, capture_output=True, timeout=30,
        )
        data, sr = sf.read(wav, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def render(path: Path, outpath: Path) -> None:
    y, sr = decode(path)
    dur = len(y) / sr

    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=HOP)[0]
    rms_t = np.arange(len(rms)) * HOP / sr
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=128,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, (ax_w, ax_s, ax_e) = plt.subplots(
        3, 1, figsize=(min(24, max(6, dur * 0.6)), 7),
        sharex=True, gridspec_kw={"height_ratios": [2, 3, 1]},
    )

    # waveform
    ax_w.plot(np.arange(len(y)) / sr, y, lw=0.4, color="#1f77b4")
    ax_w.set_ylabel("amplitude")
    ax_w.set_title(f"{path.name}   dur={dur:.1f}s   sr={sr}")
    ax_w.set_ylim(-1.05, 1.05)

    # mel-spectrogram
    librosa.display.specshow(
        mel_db, sr=sr, hop_length=HOP, x_axis="time", y_axis="mel",
        ax=ax_s, cmap="magma",
    )
    ax_s.set_ylabel("mel freq")

    # RMS energy envelope
    ax_e.plot(rms_t, rms, lw=0.7, color="#d62728")
    ax_e.fill_between(rms_t, rms, color="#d62728", alpha=0.3)
    ax_e.set_ylabel("RMS")
    ax_e.set_xlabel("time (s)")
    ax_e.set_xlim(0, dur)

    fig.tight_layout()
    fig.savefig(outpath, dpi=110)
    plt.close(fig)
    print(f"  {outpath}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", help=".m4a clip path(s)")
    ap.add_argument("--outdir", default="scripts/output/viz")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for f in args.files:
        p = Path(f)
        render(p, outdir / f"{p.stem}.png")


if __name__ == "__main__":
    main()
