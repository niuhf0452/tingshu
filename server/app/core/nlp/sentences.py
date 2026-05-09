"""Map LLM-produced sentence texts back to (line, col) positions in the source.

Production flow:

    Python sees only ``chapter_text`` and the LLM's ``list[AnalyzedSentence]``.
    ``locate_sentences`` does a forward substring search (cursor advances
    monotonically) to assign each sentence a ``(start_line, start_col,
    end_line, end_col)`` range.

Why this is robust:
- LLMs are bad at exact character offsets but fine at copying text verbatim.
- The cursor is monotonic so a repeated phrase (e.g. a character's catch-phrase
  said twice) still resolves to distinct ranges — the first call consumes the
  first occurrence, the second call starts from there.
- Whitespace-tolerant matching lets us recover even if the LLM slightly
  reformats whitespace between spans.
"""
from __future__ import annotations

import bisect
import logging

from ..models import AnalyzedSentence, Sentence


log = logging.getLogger(__name__)


def locate_sentences(
    chapter_text: str,
    analyzed: list[AnalyzedSentence],
    speaker_to_id: dict[str, int] | None = None,
) -> list[Sentence]:
    """Fill in (line, col) for each analyzed sentence via forward match.

    ``speaker_to_id`` maps each ``AnalyzedSentence.speaker`` to the
    character id assigned by ``core.nlp.reconcile``. Speakers not in the
    map default to ``0`` (narrator) — a safe fall-through in case a caller
    passes in partially-reconciled data.

    Sentences that can't be located are dropped with a warning — better to
    surface fewer sentences than to produce wrong highlight ranges.
    """
    if not chapter_text or not analyzed:
        return []

    mapping = speaker_to_id or {}
    line_starts = _build_line_starts(chapter_text)
    cursor = 0
    results: list[Sentence] = []

    for entry in analyzed:
        raw = entry.text
        if not raw or not raw.strip():
            continue

        span = _find_span(chapter_text, raw, cursor)
        if span is None:
            log.debug("could not locate sentence starting %r from cursor=%d", raw[:20], cursor)
            continue

        start, end = span
        start_line, start_col = _offset_to_line_col(start, line_starts)
        end_line, end_col = _offset_to_line_col(end, line_starts)
        character_id = mapping.get(entry.speaker.strip(), 0)
        # Capture the actual text from chapter_text (not entry.text) ——
        # they differ when the whitespace-fallback match was used, and
        # what's actually stored at the position is what should appear
        # in debug output.
        results.append(Sentence(
            start_line=start_line,
            start_col=start_col,
            end_line=end_line,
            end_col=end_col,
            character_id=character_id,
            tone=entry.tone,
            text=chapter_text[start:end],
        ))
        cursor = end

    return results


def _build_line_starts(text: str) -> list[int]:
    """Offsets where each line begins in the linear string.

    ``line_starts[i]`` is the start offset of line ``i+1`` (1-based lines).
    An extra trailing entry is appended for bisect convenience.
    """
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    starts.append(len(text) + 1)  # sentinel so bisect_right always finds a line
    return starts


def _offset_to_line_col(offset: int, line_starts: list[int]) -> tuple[int, int]:
    """Convert a linear offset to (line 1-based, col 0-based)."""
    # ``bisect_right`` returns the insertion point to the right of equal
    # entries — so for offset == line_starts[k] we get k+1, hence -1.
    idx = bisect.bisect_right(line_starts, offset) - 1
    return idx + 1, offset - line_starts[idx]


def _find_span(chapter_text: str, needle: str, cursor: int) -> tuple[int, int] | None:
    """Locate ``needle`` in ``chapter_text[cursor:]``.

    First try an exact match. If the LLM normalised whitespace, fall back
    to a whitespace-collapsed match that still reports the original span.
    """
    idx = chapter_text.find(needle, cursor)
    if idx >= 0:
        return idx, idx + len(needle)

    # Whitespace-insensitive fallback. Build a compact-index map so we can
    # translate match offsets in the compact string back to original offsets.
    compact_needle = _strip_ws(needle)
    if not compact_needle:
        return None

    compact_text, back_map = _compact_with_map(chapter_text, cursor)
    pos = compact_text.find(compact_needle)
    if pos < 0:
        return None

    start = back_map[pos]
    end = back_map[pos + len(compact_needle) - 1] + 1
    return start, end


def _strip_ws(s: str) -> str:
    return "".join(c for c in s if not c.isspace())


def _compact_with_map(text: str, offset: int) -> tuple[str, list[int]]:
    """Return (text-with-whitespace-removed, original-offset-per-compact-char)."""
    compact_chars: list[str] = []
    back_map: list[int] = []
    for i in range(offset, len(text)):
        ch = text[i]
        if ch.isspace():
            continue
        compact_chars.append(ch)
        back_map.append(i)
    return "".join(compact_chars), back_map
