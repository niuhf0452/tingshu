"""Validate that the 50-char-soft-split rule actually fixes Qwen3-TTS
quality on a long-segment problem case.

Synthesises the same sentence twice:
- ``baseline`` — the full 68-char sentence in one shot (current cached
  audio, known to garble at the end).
- ``split``    — the same sentence cut at a comma boundary into two
  ~30-40-char halves and synthesised separately.

User listens to all three files (full / first-half / second-half) and
confirms whether each split half sounds clean. If yes, the prompt's
new 50-char rule is the right fix and we just need to re-analyse the
affected chapters to apply it.

Usage::

    cd server
    source .venv/bin/activate
    python -m scripts.synth_split_compare
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.deps import get_repository, get_tts_service, get_voice_library
from app.core.enums import Tone
from app.core.voice import resolve_speaker


BOOK_ID = "6b7e0d3c95e0"
CHAPTER_ID = 395
NARRATOR_ID = 0  # iOS default

FULL = (
    "除此之外，陈平安还凭空取出那根在倒悬山炼制而成的缚妖索，"
    "以蛟龙沟元婴老蛟的金色龙须作为法宝根本，"
    "在世间千奇百怪的法宝当中，品相也算极高。"
)
# Split at the second comma — leaves first half 28 chars, second 40 chars.
# Both well under 50 so quality should be uniform across each.
SPLIT_A = "除此之外，陈平安还凭空取出那根在倒悬山炼制而成的缚妖索，"
SPLIT_B = (
    "以蛟龙沟元婴老蛟的金色龙须作为法宝根本，"
    "在世间千奇百怪的法宝当中，品相也算极高。"
)
assert SPLIT_A + SPLIT_B == FULL, "split must concatenate to original"


def synth(label: str, text: str) -> Path:
    repo = get_repository()
    library = get_voice_library()
    speaker = resolve_speaker(
        character_id=NARRATOR_ID, library=library, book_characters=[],
    )
    tts = get_tts_service()
    print(f"[{label}] {len(text)}字  speaker={speaker.speaker_id}")
    audio = tts.synthesize(text=text, speaker=speaker, tone=Tone.NEUTRAL)
    out = Path("/tmp") / f"tingshu_synth_split_{label}.m4a"
    out.write_bytes(audio)
    print(f"[{label}] wrote {len(audio):,} bytes -> {out}")
    return out


def main() -> None:
    print("=== baseline (one 68-char shot — known bad) ===")
    full = synth("full", FULL)
    print()
    print(f"=== split A ({len(SPLIT_A)}字) ===")
    a = synth("a", SPLIT_A)
    print()
    print(f"=== split B ({len(SPLIT_B)}字) ===")
    b = synth("b", SPLIT_B)
    print()
    print("Listen to all three:")
    print(f"  open {full}")
    print(f"  open {a}")
    print(f"  open {b}")
    print()
    print("If A and B are clean and the full has degraded tail, the "
          "50-char rule (already in the prompt) is the right fix; "
          "we just need to re-analyse affected chapters.")


if __name__ == "__main__":
    main()
