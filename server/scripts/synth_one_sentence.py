"""Synthesise a single sentence from a real chapter meta to A/B-test
TTS output quality.

Loads the saved ``ChapterMeta`` for ``(book_id, chapter_id)``, locates
the sentence whose verbatim ``text`` starts with the given prefix,
resolves its speaker the same way the production endpoint does, calls
the configured TTS backend (qwen3_tts per ``config.yaml``), and writes
the resulting bytes to disk for listening.

Usage::

    cd server
    source .venv/bin/activate
    python -m scripts.synth_one_sentence \\
        --book-id 6b7e0d3c95e0 \\
        --chapter-id 392 \\
        --prefix '除此之外，陈平安还凭空取出'

The output file is written to ``/tmp/tingshu_synth_<sha>.<ext>`` and
the path is printed at the end. ``ext`` is ``.m4a`` for the qwen3
backend, ``.wav`` for the stub.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/...` and `python -m scripts...` both.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.deps import get_repository, get_tts_service, get_voice_library
from app.config import get_settings
from app.core.voice import resolve_speaker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-id", required=True)
    parser.add_argument("--chapter-id", type=int, required=True)
    parser.add_argument(
        "--prefix", required=True,
        help="Verbatim leading text of the target sentence "
             "(must match what's stored in chapter meta).",
    )
    parser.add_argument(
        "--narrator-id", type=int, default=0,
        help="Which narrator voice to use when the sentence's speaker "
             "is the narrator (id 0). 0 = male, 1 = female. Mirrors "
             "iOS settings.narratorCharacterId.",
    )
    args = parser.parse_args()

    settings = get_settings()
    print(f"[config] tts.provider = {settings.tts.provider}")
    print(f"[config] qwen3.model_dir = {settings.qwen3_model_dir}")

    repo = get_repository()
    chapter_meta = repo.load_chapter_meta(args.book_id, args.chapter_id)
    if chapter_meta is None:
        print(
            f"ERROR: no chapter meta for book={args.book_id} "
            f"chapter={args.chapter_id}",
            file=sys.stderr,
        )
        sys.exit(2)

    target = None
    target_idx = -1
    for i, s in enumerate(chapter_meta.sentences):
        text = s.text or ""
        if text.startswith(args.prefix):
            target = s
            target_idx = i
            break
    if target is None:
        print(
            f"ERROR: no sentence in chapter starts with {args.prefix!r}",
            file=sys.stderr,
        )
        sys.exit(2)

    text = target.text or ""
    print()
    print(f"[sentence] idx={target_idx}  chars={len(text)}")
    print(f"[sentence] character_id={target.character_id} tone={target.tone.value}")
    print(f"[sentence] lines {target.start_line}-{target.end_line}")
    print(f"[sentence] text={text!r}")

    # Mirror PlaybackService.effectiveCharacterId: narrator (id 0) is
    # substituted with the user-selected narrator id before TTS lookup.
    effective_id = target.character_id
    if effective_id == 0:
        effective_id = args.narrator_id
        print(f"[narrator] substituting character_id 0 -> {effective_id}")

    # Mirror api/tts.py speaker-resolution logic.
    library = get_voice_library()
    if not library:
        print("ERROR: voice library is empty", file=sys.stderr)
        sys.exit(2)
    book_chars = []
    if effective_id > 15:
        book_chars = repo.load_characters(args.book_id)
    elif effective_id < 0:
        # Incidental chars resolve via this same chapter's snapshot.
        book_chars = chapter_meta.characters

    speaker = resolve_speaker(
        character_id=effective_id, library=library, book_characters=book_chars,
    )
    print()
    print(f"[speaker] id={speaker.speaker_id}")
    print(f"[speaker] gender={speaker.gender.value} age={speaker.age.value}")
    print(f"[speaker] personality={[p.value for p in speaker.personality]}")

    # Synthesise. TTSService applies its own normalize + cache, same as
    # the live endpoint.
    print()
    print("[synth] calling backend… (qwen3_tts is ~3-8 s per sentence on M4)")
    tts = get_tts_service()
    audio = tts.synthesize(text=text, speaker=speaker, tone=target.tone)
    ext = ".m4a" if (
        len(audio) >= 8 and audio[4:8] == b"ftyp"
    ) else ".wav"

    out = Path("/tmp") / f"tingshu_synth_{args.book_id}_ch{args.chapter_id}_s{target_idx}{ext}"
    out.write_bytes(audio)
    print(f"[synth] wrote {len(audio):,} bytes -> {out}")
    print()
    print(f"open {out}   # to play it")


if __name__ == "__main__":
    main()
