"""Ad-hoc verification: re-run segment_chapter on 剑来 ch.439 (file 0441)
with the new line-aligned batching and check the tail is no longer lost.

Does NOT touch the stored 0441.json — it only re-runs segmentation in
memory and prints a coverage report. Run from the server/ directory:

    python3 scripts/test_segment_0441.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.core.models import Character
from app.core.nlp.reconcile import reconcile_chapter_speakers
from app.core.nlp.sentences import locate_sentences
from app.services.llm_factory import create_llm_client
from app.services.llm_prompts import split_chapter_for_segmentation


CHAPTER_DIR = Path("data/books/6b7e0d3c95e0/chapters")
CHAPTER_TXT = CHAPTER_DIR / "0441.txt"
CHARACTERS_JSON = Path("data/books/6b7e0d3c95e0/characters.json")
MAX_TOKENS_CHAPTER = 16384


def main() -> None:
    chapter_text = CHAPTER_TXT.read_text(encoding="utf-8")
    lines = chapter_text.split("\n")
    print(f"chapter: {CHAPTER_TXT}  ({len(chapter_text)} chars, {len(lines)} lines)")

    roster = [Character(**c) for c in json.loads(CHARACTERS_JSON.read_text("utf-8"))]
    print(f"roster: {len(roster)} characters")

    batches = split_chapter_for_segmentation(chapter_text, MAX_TOKENS_CHAPTER)
    print(f"batching: {len(batches)} batch(es) -> "
          + ", ".join(f"{len(b)} chars" for b in batches))

    llm = create_llm_client(get_settings().llm)
    print("calling segment_chapter (new batched path)...")
    analyzed = llm.segment_chapter(chapter_text, roster)
    print(f"segment_chapter returned {len(analyzed)} segments")

    speaker_to_id, _ = reconcile_chapter_speakers(
        known=roster, speakers=[s.speaker for s in analyzed],
    )
    sentences = locate_sentences(chapter_text, analyzed, speaker_to_id)
    print(f"locate_sentences resolved {len(sentences)} sentences")

    # --- coverage report (same check used to find the original bug) ---
    covered: set[int] = set()
    for s in sentences:
        for ln in range(s.start_line - 1, s.end_line):
            covered.add(ln)
    nonblank = [i for i, ln in enumerate(lines) if ln.strip()]
    missing = [i for i in nonblank if i not in covered]

    max_end = max((s.end_line for s in sentences), default=0)
    print()
    print(f"nonblank lines: {len(nonblank)}")
    print(f"covered:        {len(nonblank) - len(missing)}")
    print(f"missing:        {len(missing)}")
    print(f"last covered line: {max_end} / {len(lines)}")

    # --- segment length report (long-sentence split should keep these short) ---
    vlens = [sum(1 for c in s.text if not c.isspace()) for s in sentences]
    if vlens:
        over = [v for v in vlens if v > 60]
        print()
        print(f"segment length (visible chars): "
              f"min={min(vlens)} max={max(vlens)} avg={sum(vlens) / len(vlens):.1f}")
        print(f"segments over 60 chars: {len(over)}")
    print()
    if missing:
        print("missing lines (line# : text):")
        for i in missing:
            print(f"  {i + 1}: {lines[i][:80]!r}")
    else:
        print("every nonblank line is covered.")


if __name__ == "__main__":
    main()
