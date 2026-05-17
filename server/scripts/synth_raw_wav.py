"""Generate a sentence's audio as both raw WAV (Qwen3 native PCM) and
M4A (AAC-encoded), from a SINGLE generation pass.

Why one pass: Qwen3-TTS is non-deterministic, so two ``synthesize``
calls on the same text produce different audio. To attribute noise to
the AAC encode pass (versus the model itself), we have to share the
underlying float32 stream — collect it once into WAV, then encode that
identical WAV into M4A. If WAV is clean and M4A is dirty, AAC is the
culprit; if both are dirty, the Qwen3 output already has the artifact.

Usage::

    cd server
    source .venv/bin/activate
    python -m scripts.synth_raw_wav
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.deps import get_tts_service, get_voice_library
from app.core.enums import Tone
from app.core.voice import resolve_speaker
from app.services.tts_qwen3 import (
    Qwen3TTSClient,
    _collect_audio_wav,
    _encode_aac,
    parse_speaker_id,
    tone_instruct,
)


SENTENCE = (
    "除此之外，陈平安还凭空取出那根在倒悬山炼制而成的缚妖索，"
    "以蛟龙沟元婴老蛟的金色龙须作为法宝根本，"
    "在世间千奇百怪的法宝当中，品相也算极高。"
)
NARRATOR_ID = 0


def main() -> None:
    library = get_voice_library()
    speaker = resolve_speaker(
        character_id=NARRATOR_ID, library=library, book_characters=[],
    )
    print(f"speaker: {speaker.speaker_id}")
    print(f"text   : {len(SENTENCE)} chars")

    # We need direct access to the Qwen3TTSClient so we can intercept
    # between the generator and the AAC encode pass. The deps factory
    # builds a TTSService wrapping the client; reach inside.
    service = get_tts_service()
    client = service.client
    if not isinstance(client, Qwen3TTSClient):
        raise SystemExit(
            f"need qwen3_tts backend; got {type(client).__name__}"
        )

    # Mirror Qwen3TTSClient.synthesize but split the WAV / AAC steps.
    mode, target = parse_speaker_id(speaker.speaker_id)
    instruct = tone_instruct(Tone.NEUTRAL) or None
    if mode != "zs":
        raise SystemExit("expected zs:* speaker for narrator")
    ref_audio_path, ref_text = client._load_prompt(target)

    print("\nrunning Qwen3 generation…")
    from app.services.mlx_gpu import gpu_guard
    with gpu_guard():
        generator = client._model.generate(
            text=SENTENCE,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            instruct=instruct,
            speed=1.0,
            verbose=False,
        )
        wav_bytes = _collect_audio_wav(generator)

    wav_path = Path("/tmp/tingshu_raw.wav")
    wav_path.write_bytes(wav_bytes)
    print(f"raw WAV  -> {wav_path}  ({len(wav_bytes):,} bytes)")

    m4a_bytes = _encode_aac(wav_bytes)
    m4a_path = Path("/tmp/tingshu_raw_encoded.m4a")
    m4a_path.write_bytes(m4a_bytes)
    print(f"AAC M4A  -> {m4a_path}  ({len(m4a_bytes):,} bytes)")

    print()
    print("Listen to both:")
    print(f"  open {wav_path}")
    print(f"  open {m4a_path}")
    print()
    print("If only M4A has the tail noise → AAC encode is to blame.")
    print("If WAV also has it → the noise is in Qwen3's output before "
          "compression.")


if __name__ == "__main__":
    main()
