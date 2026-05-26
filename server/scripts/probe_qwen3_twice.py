"""Minimal: two sequential generate() calls on main thread."""
from __future__ import annotations
import sys, time
from pathlib import Path

MODEL_DIR = Path("pretrained_models/Qwen3-TTS-0.6B").resolve()
PROMPT_WAV = Path("data/voices/prompts/vd_narrator_male_mature.wav").resolve()
PROMPT_TXT = Path("data/voices/prompts/vd_narrator_male_mature.txt").resolve()
TEXTS = [
    "如今便完全换了一幅场景，水府之内处处热火朝天，一个个小家伙奔跑不停，欢天喜地，任劳任怨，乐在其中。",
    "他抬起头，望向远处的山峦，眼神中流露出一丝复杂的情绪。",
    "夜已深，整座城市渐渐沉入梦乡，只有几盏路灯还在静静地守候。",
    "你好，今天天气真好。",
]

from mlx_audio.tts.utils import load_model
print("loading …")
model = load_model(str(MODEL_DIR))
print("loaded")
ref_text = PROMPT_TXT.read_text(encoding="utf-8").strip()

for i, txt in enumerate(TEXTS):
    print(f"\n=== call {i+1} ===", flush=True)
    t = time.monotonic()
    try:
        total = 0
        for chunk in model.generate(
            text=txt, ref_audio=str(PROMPT_WAV), ref_text=ref_text,
            instruct=None, speed=1.0, verbose=False,
        ):
            audio = getattr(chunk, "audio", None)
            if audio is not None:
                total += getattr(audio, "size", 0)
        print(f"call {i+1}: samples={total} wall={time.monotonic()-t:.2f}s", flush=True)
    except Exception as e:
        import traceback
        print(f"call {i+1}: EXC {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
