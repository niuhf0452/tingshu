"""Offline detector for defective Qwen3-TTS cache audio.

Scans the on-disk TTS cache (``data/tts_cache/*.m4a``) and flags clips
where synthesis went wrong. Detection is purely from the audio signal —
no ASR, no text, no clip-duration verdict.

READ-ONLY. Never edits the cache or any production code; only prints a
report and writes a CSV.

Usage (run from the server/ directory):
    python3 scripts/detect_tts_defects.py                 # scan everything
    python3 scripts/detect_tts_defects.py --limit 1000    # quick sample
    python3 scripts/detect_tts_defects.py --workers 10 --report out.csv

THE DEFECT & HOW IT IS DETECTED
-------------------------------
Confirmed defects all share one failure mode: the model synthesises the
real sentence, then fails to terminate and trails off into a long span
of garbage — a *mix* of noise, stuttering/repetition and silence that
cannot be (and need not be) told apart.

What unifies that garbage span, whatever it is made of, is its texture:
clean narration is **dense and continuous**, the garbage span is
**choppy** — sparse energy bursts separated by silence. So the detector
measures the local *temporal density* of voiced frames, not any
per-frame spectral property:

    voiced[f]   = rms[f] > VOICED_REL_FLOOR × (95th-pct rms)
    density[f]  = fraction of voiced frames in a ±DENS_WIN/2 window
    bad_span_s  = longest contiguous run with density < DENS_THR

Calibrated against user-confirmed clips. The metric only separates
cleanly at the gross end: every clip scanned with ``bad_span_s`` ≥ 5 s
was a real defect (the model froze / trailed off / faded to
near-silence), with zero false positives. Below ~5 s the metric cannot
separate a short defect from an ordinary pause or quiet passage — both
produce low-density spans of 2–3 s — so detection there is unreliable
and is intentionally not attempted.

A clip is flagged when ``bad_span_s`` exceeds BAD_SPAN_MAX_S (5 s):
that catches the gross "model failed to terminate" failures, which are
the high-impact ones (the whole sentence is ruined).
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import librosa  # noqa: E402  (after warning filter — librosa is chatty)
import soundfile as sf  # noqa: E402


# --- frame geometry --------------------------------------------------------
HOP = 256                     # ~10.7 ms hop @ 24 kHz
RMS_WIN = 512

# --- voiced-density detector ----------------------------------------------
VOICED_REL_FLOOR = 0.15       # frame is "voiced" if RMS > this × clip RMS peak
DENS_WIN_S = 1.5              # window the voiced-density is averaged over
DENS_THR = 0.45               # local density below this ⇒ "garbage" frame
BAD_SPAN_MAX_S = 5.0          # longest garbage run longer than this ⇒ flag
FULLY_SILENT_PEAK = 1e-3      # 95th-pct frame RMS below this ⇒ digital silence


@dataclass
class Result:
    path: str
    ok: bool = True
    error: str = ""
    duration_s: float = 0.0
    voiced_ratio: float = 0.0
    bad_span_s: float = 0.0       # longest low-density (garbage) run
    bad_span_start_s: float = 0.0
    flags: list[str] = field(default_factory=list)


# --- decoding --------------------------------------------------------------

def decode(path: Path) -> tuple[np.ndarray, int]:
    """m4a → mono float32 waveform via afconvert (macOS; the same tool the
    production encode path uses)."""
    with tempfile.TemporaryDirectory(prefix="tts-detect-") as td:
        wav = Path(td) / "o.wav"
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16", str(path), str(wav)],
            check=True, capture_output=True, timeout=60,
        )
        data, sr = sf.read(wav, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


# --- analysis --------------------------------------------------------------

def analyze(path: Path) -> Result:
    res = Result(path=str(path))
    try:
        y, sr = decode(path)
    except Exception as exc:  # noqa: BLE001 — report, don't crash the sweep
        res.ok = False
        res.error = f"decode failed: {exc}"
        return res

    res.duration_s = len(y) / sr if sr else 0.0
    if len(y) < RMS_WIN:
        res.flags.append("empty")
        return res

    rms = librosa.feature.rms(y=y, frame_length=RMS_WIN, hop_length=HOP)[0]
    spf = HOP / sr
    peak = float(np.percentile(rms, 95))

    if peak < FULLY_SILENT_PEAK:
        # No signal at all — the whole clip is the garbage span.
        res.bad_span_s = res.duration_s
        res.flags.append("defect")
        return res

    voiced = (rms > max(VOICED_REL_FLOOR * peak, 1e-4)).astype(np.float32)
    res.voiced_ratio = float(voiced.mean())

    # local voiced-density: fraction of voiced frames in a sliding window
    win = max(1, int(DENS_WIN_S / spf))
    density = np.convolve(voiced, np.ones(win) / win, mode="same")

    # longest contiguous run where density stays below DENS_THR
    res.bad_span_s, res.bad_span_start_s = _longest_low_run(density, DENS_THR, spf)

    if res.bad_span_s > BAD_SPAN_MAX_S:
        res.flags.append("defect")
    return res


def _longest_low_run(curve: np.ndarray, thr: float,
                     spf: float) -> tuple[float, float]:
    """Longest contiguous run with ``curve < thr`` → (length_s, start_s)."""
    bad = curve < thr
    if not bad.any():
        return 0.0, 0.0
    edges = np.diff(np.concatenate(([0], bad.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    k = int(np.argmax(ends - starts))
    return (ends[k] - starts[k]) * spf, starts[k] * spf


# --- driver ----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/tts_cache", help="cache directory")
    ap.add_argument("--limit", type=int, default=0,
                    help="scan at most N files (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--report", default="scripts/output/tts_defect_report.csv")
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob("*.m4a"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"no .m4a files under {args.dir}", file=sys.stderr)
        sys.exit(1)
    print(f"scanning {len(files)} file(s) from {args.dir} "
          f"with {args.workers} worker(s)...")

    results: list[Result] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(analyze, files), start=1):
            results.append(res)
            if i % 1000 == 0 or i == len(files):
                print(f"  {i}/{len(files)}")

    _report(results, Path(args.report))


def _report(results: list[Result], report_path: Path) -> None:
    decoded = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "file", "duration_s", "voiced_ratio", "bad_span_s",
            "bad_span_start_s", "flags", "error",
        ])
        for r in results:
            w.writerow([
                r.path, f"{r.duration_s:.2f}", f"{r.voiced_ratio:.3f}",
                f"{r.bad_span_s:.2f}", f"{r.bad_span_start_s:.2f}",
                "|".join(r.flags), r.error,
            ])

    suspects = [r for r in decoded if r.flags]
    print("\n" + "=" * 72)
    print(f"SUMMARY  —  {len(results)} file(s),  {len(decoded)} decoded,  "
          f"{len(failed)} decode-failed")
    print(f"clean: {len(decoded) - len(suspects)}   defect: {len(suspects)}")
    print("=" * 72)

    suspects.sort(key=lambda r: -r.bad_span_s)
    print(f"\n--- DEFECT (longest garbage span > {BAD_SPAN_MAX_S}s), "
          f"worst {min(40, len(suspects))} of {len(suspects)} ---")
    for r in suspects[:40]:
        print(f"  bad_span={r.bad_span_s:5.1f}s  from {r.bad_span_start_s:5.1f}s"
              f"  dur={r.duration_s:5.1f}s   {r.path}")

    if failed:
        print(f"\n--- DECODE FAILURES / MISSING ({len(failed)}) ---")
        for r in failed[:8]:
            print(f"  {r.path}: {r.error}")

    print(f"\nfull per-file CSV → {report_path}")
    print("listen to a flagged clip with:  afplay <path>")


if __name__ == "__main__":
    main()
