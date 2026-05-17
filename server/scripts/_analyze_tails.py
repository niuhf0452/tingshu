"""Test the 'voiced-density' detector against the 18 labelled tail clips.

Insight from the boundary-marked spectrograms: clean narration is dense
and continuous; the garbage tail (noise / stutter / freeze, mixed) is
choppy — sparse energy bursts separated by silence. So the discriminator
is the *local temporal density* of voiced frames, not any per-frame
spectral property.

Algorithm tested here:
  voiced[f]   = rms[f] > 0.15 * (95th-pct rms)
  density[f]  = fraction of voiced frames in a ±DENS_WIN/2 window
  garbage[f]  = density[f] < DENS_THR
  metric      = longest contiguous garbage run (seconds)

For each labelled clip we check the detected longest garbage run lines
up with the user's [T, end] annotation; for good clips it should be ~0.
"""
from __future__ import annotations
import os, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
import librosa

HOP = 256
RMS_WIN = 512
DENS_WIN_S = 1.5      # density smoothing window
DENS_THR = 0.45       # below this local voiced-density ⇒ "garbage"

ANN = {
 "5e9ac3a8848639fbbc4c6ad907b4de63e0e9ea72": 13.30,
 "1fae5b7888039700c743ab3330131929a212f813": 8.99,
 "563807ccb07ee45c4760daf747ecbcbfc21b094a": 7.35,
 "8431c911546623028f2e28c944b4adf1a21768c4": 7.89,
 "558854af25b767f2938481ddc159bbdd3faf9e96": 9.92,
 "695495954d470ee1e1ad8470f0d72e60b2adbc9a": 6.22,
 "61f8e94780bafcd2a62465d135dfe33f3c43781a": 8.57,
 "1bf09b0640d93bd373de181f8a93ef8c84a99224": 5.83,
 "5c546fa99cadde180ab1d7bf578e80750c098872": 7.45,
 "536817b40e20bbba85b10155e2e802738776bb69": 4.64,
 "321d3796e9fafed572d23b4c7e04b249c5ae20e6": 3.91,
 "5e94d51b5e0257a5e8f02556cc2ce89b16f02fb0": 4.43,
 "5aa60a543169f243f914b9abdbedcd59bfa0e82e": 5.31,
 "97c576c0d85db176795d199a3bcd6e387dc6355b": 1.30,
 "3883cabd582a3b8c28de64e812ef1dda578c6eee": 1.42,
 "a65466684228571aaa0157c33e7c1a5fd5bc0010": 2.83,
 "754a4f439f9ad52fe2a803357a84e5d04926550d": 3.82,
 "f71c9ea22ffe08b88d99b960338d4ef1335c18e0": 3.46,
}


def density_curve(y, sr):
    """Return (density per frame, sec_per_frame)."""
    rms = librosa.feature.rms(y=y, frame_length=RMS_WIN, hop_length=HOP)[0]
    peak = float(np.percentile(rms, 95))
    voiced = (rms > max(0.15 * peak, 1e-4)).astype(np.float32)
    win = max(1, int(DENS_WIN_S * sr / HOP))
    kern = np.ones(win) / win
    dens = np.convolve(voiced, kern, mode="same")
    return dens, HOP / sr


def longest_low_run(dens, spf):
    """Longest contiguous run with density < DENS_THR → (seconds, start_s)."""
    bad = dens < DENS_THR
    best_len, best_start = 0, 0
    i = 0
    while i < len(bad):
        if bad[i]:
            j = i
            while j < len(bad) and bad[j]:
                j += 1
            if j - i > best_len:
                best_len, best_start = j - i, i
            i = j
        else:
            i += 1
    return best_len * spf, best_start * spf


def main():
    print("LABELLED defect clips — does the longest low-density run match [T,end]?")
    print(f"{'file':10s} {'dur':>5s} {'annT':>5s}  "
          f"{'detected run (start..end)':>26s}  {'len':>6s}")
    print("-" * 64)
    for h, T in ANN.items():
        p = f"data/tts_cache/{h}.m4a"
        if not os.path.exists(p):
            print(f"{h[:9]:10s}  (cache file gone — skipped)")
            continue
        y, sr = librosa.load(p, sr=None, mono=True)
        dur = len(y) / sr
        dens, spf = density_curve(y, sr)
        run_len, run_start = longest_low_run(dens, spf)
        print(f"{h[:9]:10s} {dur:5.1f} {T:5.1f}  "
              f"{run_start:10.1f}..{run_start + run_len:<6.1f}"
              f"(true {T:.0f}..{dur:.0f})  {run_len:6.1f}s")

    # good reference clips
    print("\nGOOD reference clips — longest low-density run should be small:")
    import csv, random
    rows = [r for r in csv.DictReader(open("scripts/output/tts_defect_report.csv"))
            if not r["flags"]]
    random.seed(11)
    runs = []
    for r in random.sample(rows, 40):
        p = r["file"]
        if not os.path.exists(p):
            continue
        y, sr = librosa.load(p, sr=None, mono=True)
        dens, spf = density_curve(y, sr)
        run_len, _ = longest_low_run(dens, spf)
        runs.append(run_len)
    runs = np.array(runs)
    print(f"  n={len(runs)}  p50={np.percentile(runs,50):.2f}s  "
          f"p90={np.percentile(runs,90):.2f}s  max={runs.max():.2f}s")


if __name__ == "__main__":
    main()
