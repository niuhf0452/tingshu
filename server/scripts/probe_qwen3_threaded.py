"""Reproduce server conditions: run generate() N times,
some on the main thread, some in a worker thread (like FastAPI's
threadpool), holding the same lock the server uses.
"""
from __future__ import annotations

import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL_DIR = Path("pretrained_models/Qwen3-TTS-0.6B").resolve()
PROMPT_WAV = Path("data/voices/prompts/vd_narrator_male_mature.wav").resolve()
PROMPT_TXT = Path("data/voices/prompts/vd_narrator_male_mature.txt").resolve()
TEXTS = [
    "如今便完全换了一幅场景，水府之内处处热火朝天，一个个小家伙奔跑不停，欢天喜地，任劳任怨，乐在其中。",
    "他抬起头，望向远处的山峦，眼神中流露出一丝复杂的情绪。",
    "夜已深，整座城市渐渐沉入梦乡，只有几盏路灯还在静静地守候。",
]

_GPU_LOCK = threading.Lock()


def main() -> int:
    from mlx_audio.tts.utils import load_model
    print("loading model …")
    t0 = time.monotonic()
    model = load_model(str(MODEL_DIR))
    print(f"loaded in {time.monotonic() - t0:.1f}s")

    ref_text = PROMPT_TXT.read_text(encoding="utf-8").strip()

    def synth_once(label: str, text: str) -> int:
        t = time.monotonic()
        with _GPU_LOCK:
            try:
                gen = model.generate(
                    text=text, ref_audio=str(PROMPT_WAV), ref_text=ref_text,
                    instruct=None, speed=1.0, verbose=False,
                )
                total = 0
                chunks = 0
                for chunk in gen:
                    chunks += 1
                    audio = getattr(chunk, "audio", None)
                    if audio is not None:
                        n = getattr(audio, "size", None) or len(list(audio))
                        total += n
            except Exception as e:  # noqa: BLE001
                wall = time.monotonic() - t
                print(f"  [{label}] EXC ({type(e).__name__}: {e}) wall={wall:.2f}s thread={threading.current_thread().name}")
                traceback.print_exc()
                return -1
        wall = time.monotonic() - t
        thread = threading.current_thread().name
        print(f"  [{label}] chunks={chunks} samples={total} wall={wall:.2f}s thread={thread}")
        return total

    print("\n--- Phase 1: 3 sequential calls on main thread ---")
    for i, txt in enumerate(TEXTS):
        synth_once(f"main-{i}", txt)

    print("\n--- Phase 2: 3 sequential calls on worker thread (like FastAPI) ---")
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="worker") as ex:
        for i, txt in enumerate(TEXTS):
            ex.submit(synth_once, f"worker-{i}", txt).result()

    print("\n--- Phase 3: alternate main / worker ---")
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="worker") as ex:
        for i, txt in enumerate(TEXTS):
            synth_once(f"main-mix-{i}", txt)
            ex.submit(synth_once, f"worker-mix-{i}", txt).result()

    return 0


if __name__ == "__main__":
    sys.exit(main())
