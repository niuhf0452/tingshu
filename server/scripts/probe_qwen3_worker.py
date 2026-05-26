"""Reproduce the exact server scenario:
- Load model on main thread
- Run generate() inside a FastAPI-style worker thread
- Try several stream-binding strategies
"""
from __future__ import annotations
import sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import mlx.core as mx

MODEL_DIR = Path("pretrained_models/Qwen3-TTS-0.6B").resolve()
PROMPT_WAV = Path("data/voices/prompts/vd_narrator_male_mature.wav").resolve()
PROMPT_TXT = Path("data/voices/prompts/vd_narrator_male_mature.txt").resolve()
TEXT = "你好，今天天气真好。"

from mlx_audio.tts.utils import load_model
print("loading on main thread …", flush=True)
model = load_model(str(MODEL_DIR))
print("loaded", flush=True)
ref_text = PROMPT_TXT.read_text(encoding="utf-8").strip()


def call(label: str, setup=None):
    t = time.monotonic()
    try:
        if setup:
            setup()
        total = 0
        for chunk in model.generate(
            text=TEXT, ref_audio=str(PROMPT_WAV), ref_text=ref_text,
            instruct=None, speed=1.0, verbose=False,
        ):
            audio = getattr(chunk, "audio", None)
            if audio is not None:
                total += getattr(audio, "size", 0)
        print(f"  [{label}] samples={total} wall={time.monotonic()-t:.2f}s", flush=True)
    except Exception as e:
        print(f"  [{label}] EXC {type(e).__name__}: {e} wall={time.monotonic()-t:.2f}s", flush=True)


def main_thread_baseline():
    call("main")

def worker_no_setup():
    with ThreadPoolExecutor(max_workers=1) as ex:
        ex.submit(call, "worker(no setup)").result()

def worker_with_mx_stream_gpu():
    def setup_ctx_mgr():
        # this is a no-op outside `with` — we need to enter the context
        pass
    def task():
        try:
            with mx.stream(mx.gpu):
                total = 0
                for chunk in model.generate(
                    text=TEXT, ref_audio=str(PROMPT_WAV), ref_text=ref_text,
                    instruct=None, speed=1.0, verbose=False,
                ):
                    audio = getattr(chunk, "audio", None)
                    if audio is not None:
                        total += getattr(audio, "size", 0)
                print(f"  [worker mx.stream(gpu)] samples={total}", flush=True)
        except Exception as e:
            print(f"  [worker mx.stream(gpu)] EXC {type(e).__name__}: {e}", flush=True)
    with ThreadPoolExecutor(max_workers=1) as ex:
        ex.submit(task).result()

def worker_with_set_default_stream():
    def task():
        try:
            mx.set_default_stream(mx.default_stream(mx.gpu))
            total = 0
            for chunk in model.generate(
                text=TEXT, ref_audio=str(PROMPT_WAV), ref_text=ref_text,
                instruct=None, speed=1.0, verbose=False,
            ):
                audio = getattr(chunk, "audio", None)
                if audio is not None:
                    total += getattr(audio, "size", 0)
            print(f"  [worker set_default_stream] samples={total}", flush=True)
        except Exception as e:
            print(f"  [worker set_default_stream] EXC {type(e).__name__}: {e}", flush=True)
    with ThreadPoolExecutor(max_workers=1) as ex:
        ex.submit(task).result()


if __name__ == "__main__":
    print("\n=== A: main thread (baseline) ===")
    main_thread_baseline()
    print("\n=== B: worker, NO setup ===")
    worker_no_setup()
    print("\n=== C: worker, with mx.stream(mx.gpu) ===")
    worker_with_mx_stream_gpu()
    print("\n=== D: worker, with set_default_stream ===")
    worker_with_set_default_stream()
