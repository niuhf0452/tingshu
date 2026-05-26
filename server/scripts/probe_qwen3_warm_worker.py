"""Reproduce server scenario: a worker thread that was created BEFORE
the model was loaded (matching uvicorn lifecycle).
"""
from __future__ import annotations
import sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import mlx.core as mx

MODEL_DIR = Path("pretrained_models/Qwen3-TTS-0.6B").resolve()
PROMPT_WAV = Path("data/voices/prompts/vd_narrator_male_mature.wav").resolve()
PROMPT_TXT = Path("data/voices/prompts/vd_narrator_male_mature.txt").resolve()
TEXT = "你好，今天天气真好。"

# Start the worker thread FIRST — keep it alive (like uvicorn's threadpool)
ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="warm-worker")
# Force the worker to actually run something so the thread is materialised
ex.submit(lambda: None).result()
print("warm worker pool ready", flush=True)

# Then load model on main thread (like FastAPI lifespan)
from mlx_audio.tts.utils import load_model
print("loading model on main thread (with worker already alive) …", flush=True)
model = load_model(str(MODEL_DIR))
print("loaded", flush=True)
ref_text = PROMPT_TXT.read_text(encoding="utf-8").strip()


def synth(label, strategy=None):
    t = time.monotonic()
    try:
        if strategy == "mx_stream":
            ctx = mx.stream(mx.gpu)
            ctx.__enter__()
        elif strategy == "set_default":
            mx.set_default_stream(mx.default_stream(mx.gpu))
        total = 0
        for chunk in model.generate(
            text=TEXT, ref_audio=str(PROMPT_WAV), ref_text=ref_text,
            instruct=None, speed=1.0, verbose=False,
        ):
            audio = getattr(chunk, "audio", None)
            if audio is not None:
                total += getattr(audio, "size", 0)
        print(f"  [{label}] samples={total} wall={time.monotonic()-t:.2f}s thr={threading.current_thread().name}", flush=True)
        return total
    except Exception as e:
        print(f"  [{label}] EXC {type(e).__name__}: {e} wall={time.monotonic()-t:.2f}s thr={threading.current_thread().name}", flush=True)
        return -1
    finally:
        if strategy == "mx_stream":
            ctx.__exit__(None, None, None)


def synth_after(label, prelude):
    def task():
        try:
            prelude()
        except Exception as e:
            print(f"  [{label}] prelude EXC {type(e).__name__}: {e}", flush=True)
            return
        synth(label)
    ex.submit(task).result()


print("\n=== E: warmup with mx.eval(mx.array([1.0])) first ===")
synth_after("warmup-eval", lambda: mx.eval(mx.array([1.0])))

print("\n=== F: new_thread_local_stream ===")
def make_tls():
    mx.new_thread_local_stream(mx.gpu)
synth_after("new-tls", make_tls)

print("\n=== G: set_default_stream(new_stream(gpu)) ===")
def fresh_stream():
    s = mx.new_stream(mx.gpu)
    mx.set_default_stream(s)
synth_after("new-stream", fresh_stream)

print("\n=== H: load model on this worker thread ===")
def load_here():
    global model
    from mlx_audio.tts.utils import load_model
    model = load_model(str(MODEL_DIR))
synth_after("reload-on-worker", load_here)

ex.shutdown(wait=True)
