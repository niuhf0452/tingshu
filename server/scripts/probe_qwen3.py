"""Standalone probe — calls mlx_audio.generate directly, no server stack.

Use it to bisect "TTS returns silent audio" failures: skips uvicorn,
threadpool, gpu_guard, _collect_samples, retry loop — so any failure
surfaces in raw form with full traceback. Run from server/:

    uv run python scripts/probe_qwen3.py
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

MODEL_DIR = Path("pretrained_models/Qwen3-TTS-0.6B").resolve()
PROMPT_WAV = Path("data/voices/prompts/vd_narrator_male_mature.wav").resolve()
PROMPT_TXT = Path("data/voices/prompts/vd_narrator_male_mature.txt").resolve()
TEXT = (
    "如今便完全换了一幅场景，水府之内处处热火朝天，"
    "一个个小家伙奔跑不停，欢天喜地，任劳任怨，乐在其中。"
)


def main() -> int:
    for p in (MODEL_DIR, PROMPT_WAV, PROMPT_TXT):
        if not p.exists():
            print(f"MISSING: {p}", file=sys.stderr)
            return 2

    print(f"loading model from {MODEL_DIR} …")
    t0 = time.monotonic()
    from mlx_audio.tts.utils import load_model
    model = load_model(str(MODEL_DIR))
    print(f"model loaded in {time.monotonic() - t0:.1f}s")
    print(f"model class: {type(model).__module__}.{type(model).__name__}")

    ref_text = PROMPT_TXT.read_text(encoding="utf-8").strip()
    print(f"ref_text: {ref_text!r}")
    print(f"target text ({len(TEXT)} chars): {TEXT!r}")

    print("\ninvoking model.generate() …")
    t0 = time.monotonic()
    try:
        gen = model.generate(
            text=TEXT,
            ref_audio=str(PROMPT_WAV),
            ref_text=ref_text,
            speed=1.0,
            verbose=False,  # match the server's call
        )
        chunks = 0
        empty = 0
        total_samples = 0
        for chunk in gen:
            chunks += 1
            audio = getattr(chunk, "audio", None)
            if audio is None:
                empty += 1
                print(f"  chunk {chunks}: NO AUDIO  attrs={dir(chunk)[:8]}…")
                continue
            n = getattr(audio, "size", None)
            if n is None:
                n = len(list(audio))
            if n == 0:
                empty += 1
                print(f"  chunk {chunks}: size=0")
                continue
            total_samples += n
            print(f"  chunk {chunks}: size={n}")
        wall = time.monotonic() - t0
        print(
            f"\ndone in {wall:.1f}s — chunks={chunks} "
            f"empty={empty} total_samples={total_samples} "
            f"(~{total_samples / 24000:.2f}s of audio)"
        )
        return 0 if total_samples > 0 else 1
    except Exception:  # noqa: BLE001
        print("\n!!! exception during generate !!!", file=sys.stderr)
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
