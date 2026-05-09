"""TXT chapter detection (§2.4.1).

LLM-driven detector with two modes plus a TOC-pagination loop:

1. Take a 200-line batch from the book's opening and call
   ``LLMClient.detect_chapters``. The LLM classifies the batch as TOC or
   non-TOC and returns either a partial TOC title list with a
   ``toc_complete`` flag, or a ``first_chapter_title`` + ``chapter_pattern``.
2. TOC mode only: if ``toc_complete=False``, feed the next 200-line batch
   (passing accumulated titles as a hint) and merge results. Repeat up to
   ``MAX_TOC_BATCHES``.
3. Python then locates chapter boundaries in the full text:
   - TOC mode: forward ``find(title, cursor)`` with monotonic cursor.
   - Pattern mode: compile the LLM-supplied regex and scan.

Single failure fallback: wrap the whole text as one chapter named after
the file. No hardcoded regex.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from ..models import ChapterDetection
from ..parsers.txt import ParsedChapter, _trim_blank_lines
from ...services.llm import LLMClient


log = logging.getLogger(__name__)

# How much of the book we feed the LLM per batch. 200 lines is large
# enough to comfortably hold a preface (楔子 / 序章) + first chapter for
# pattern mode, and also a reasonable slice of a long TOC for pagination.
OPENING_LINE_LIMIT = 200

# Cap on TOC-pagination rounds. 10 batches × 200 lines = 2000 lines — well
# beyond any realistic TOC. Past this cap we assume the LLM is stuck and
# drop to the single-chapter fallback.
MAX_TOC_BATCHES = 10

# Minimum chapters to accept LLM output. Below this we assume detection
# misfired and drop to the single-chapter fallback.
MIN_CHAPTERS_FROM_LLM = 2


@dataclass
class DetectionOutcome:
    """Structured result for introspection + logging.

    ``path`` is one of: ``"toc"``, ``"pattern"``, ``"fallback_single"``.
    """

    chapters: list[ParsedChapter]
    path: str
    llm_detection: ChapterDetection | None = None


def detect_and_split_chapters(
    text: str,
    llm: LLMClient,
    *,
    fallback_title: str,
) -> DetectionOutcome:
    """Split ``text`` into chapters using the LLM + single fallback strategy.

    Always returns at least one chapter (the fallback_single path wraps the
    full text when LLM detection produces fewer than 2 chapters).
    """
    if not text.strip():
        return _single_chapter(text, fallback_title)

    t0 = time.monotonic()
    detection = _detect_with_pagination(text, llm)
    chapters = _split_via_detection(text, detection) if detection is not None else None
    elapsed = time.monotonic() - t0

    if chapters is not None and len(chapters) >= MIN_CHAPTERS_FROM_LLM:
        path = "toc" if detection.has_toc else "pattern"
        log.info(
            "chapter detection done: path=%s chapters=%d wall=%.1fs",
            path, len(chapters), elapsed,
        )
        return DetectionOutcome(chapters=chapters, path=path, llm_detection=detection)

    log.info(
        "chapter detection: LLM produced %s chapters (min %d); falling back to single chapter "
        "(wall=%.1fs)",
        0 if chapters is None else len(chapters), MIN_CHAPTERS_FROM_LLM, elapsed,
    )
    return _single_chapter(text, fallback_title, llm_detection=detection)


# --- pagination loop -----------------------------------------------------


def _detect_with_pagination(text: str, llm: LLMClient) -> ChapterDetection | None:
    """Drive ``LLMClient.detect_chapters`` across 200-line batches.

    TOC mode may span multiple batches — we accumulate ``chapter_titles``
    until the LLM signals ``toc_complete=True`` or we hit
    ``MAX_TOC_BATCHES``. Pattern mode returns after the first batch.
    """
    lines = text.split("\n")
    accumulated: list[str] = []
    first_detection: ChapterDetection | None = None
    completed = False
    batches_done = 0

    for batch_idx in range(MAX_TOC_BATCHES):
        start = batch_idx * OPENING_LINE_LIMIT
        batch = "\n".join(lines[start : start + OPENING_LINE_LIMIT])
        if not batch.strip():
            break
        batches_done = batch_idx + 1

        detection = llm.detect_chapters(
            batch,
            known_titles=list(accumulated) if batch_idx > 0 else None,
        )

        if batch_idx == 0:
            first_detection = detection
            if not detection.has_toc:
                # Pattern mode — single-shot.
                return detection
        elif not detection.has_toc:
            log.warning(
                "toc pagination batch %d: LLM flipped has_toc=False mid-stream; stopping",
                batch_idx,
            )
            break

        for title in detection.chapter_titles:
            if title and title not in accumulated:
                accumulated.append(title)

        if detection.toc_complete:
            completed = True
            break

    if not completed and accumulated:
        log.warning(
            "toc pagination: did not complete (batches=%d, titles=%d)",
            batches_done, len(accumulated),
        )

    if accumulated:
        return ChapterDetection(
            has_toc=True,
            chapter_titles=accumulated,
            toc_complete=completed,
        )
    return first_detection


# --- strategy implementations --------------------------------------------


def _split_via_detection(
    text: str,
    detection: ChapterDetection,
) -> list[ParsedChapter] | None:
    """Return chapters according to the LLM detection, or None on failure."""
    if detection.has_toc:
        return _split_by_toc(text, detection.chapter_titles)
    if detection.first_chapter_title and detection.chapter_pattern:
        return _split_by_pattern(
            text,
            first_title=detection.first_chapter_title,
            pattern=detection.chapter_pattern,
            preface_titles=detection.preface_titles,
        )
    return None


def _split_by_toc(text: str, titles: list[str]) -> list[ParsedChapter] | None:
    """Forward-find each TOC title in the body and slice between matches.

    The TOC block itself counts as one occurrence of each title; the body
    contains a second. A monotonic cursor walks forward so the same title
    in the TOC can't be matched twice.
    """
    if len(titles) < MIN_CHAPTERS_FROM_LLM:
        return None

    first_toc = text.find(titles[0])
    if first_toc < 0:
        return None
    cursor = first_toc + len(titles[0])
    first_body = text.find(titles[0], cursor)
    if first_body < 0:
        # No TOC after all (LLM misfired): treat first_toc as body anchor.
        first_body = first_toc
        cursor = first_toc + len(titles[0])

    anchors: list[tuple[int, str]] = [(first_body, titles[0])]
    cursor = first_body + len(titles[0])
    for title in titles[1:]:
        idx = _find_normalised(text, title, cursor)
        if idx < 0:
            log.debug("_split_by_toc: title %r not found after cursor=%d", title, cursor)
            continue
        anchors.append((idx, title))
        cursor = idx + len(title)

    if len(anchors) < MIN_CHAPTERS_FROM_LLM:
        return None
    return _slice_into_chapters(text, anchors)


def _split_by_pattern(
    text: str,
    *,
    first_title: str,
    pattern: str,
    preface_titles: list[str],
) -> list[ParsedChapter] | None:
    """Locate the first chapter by exact title, then use the regex pattern
    for subsequent chapters. Prepend preface sections if any.
    """
    try:
        compiled = re.compile(pattern, re.MULTILINE)
    except re.error as exc:
        log.warning("_split_by_pattern: invalid regex %r: %s", pattern, exc)
        return None

    first_idx = text.find(first_title)
    if first_idx < 0:
        log.warning("_split_by_pattern: first chapter title %r not found", first_title)
        return None

    # The LLM-supplied pattern usually only matches the title prefix (e.g.
    # ``第X章``), so we extend each match to end-of-line to capture the
    # full title (e.g. ``第二章 对决``).
    heading_anchors: list[tuple[int, str]] = [(first_idx, first_title)]
    for m in compiled.finditer(text, first_idx + len(first_title)):
        eol = text.find("\n", m.end())
        line_end = eol if eol >= 0 else len(text)
        title_line = text[m.start():line_end].strip()
        if title_line:
            heading_anchors.append((m.start(), title_line))

    if len(heading_anchors) < MIN_CHAPTERS_FROM_LLM:
        return None

    chapters = _slice_into_chapters(text, heading_anchors)

    pre_text = text[:first_idx]
    if pre_text.strip():
        preface_chapters = _split_preface(pre_text, preface_titles)
        chapters = preface_chapters + chapters

    return chapters


def _split_preface(
    preface_text: str,
    titles: list[str],
) -> list[ParsedChapter]:
    """Slice the preface block by the provided title anchors.

    ``titles`` comes from the LLM — e.g. ``["序章", "楔子"]``. Missing
    titles are silently skipped; any leading content before the first
    preface title is discarded (typically a publisher blurb).
    """
    if not titles:
        body = _trim_blank_lines(preface_text)
        return [ParsedChapter(title="序言", text=body)] if body else []

    anchors: list[tuple[int, str]] = []
    cursor = 0
    for title in titles:
        idx = preface_text.find(title, cursor)
        if idx < 0:
            continue
        anchors.append((idx, title))
        cursor = idx + len(title)

    if not anchors:
        body = _trim_blank_lines(preface_text)
        return [ParsedChapter(title="序言", text=body)] if body else []

    return _slice_into_chapters(preface_text, anchors)


# --- helpers --------------------------------------------------------------


def _single_chapter(
    text: str,
    fallback_title: str,
    *,
    llm_detection: ChapterDetection | None = None,
) -> DetectionOutcome:
    return DetectionOutcome(
        chapters=[ParsedChapter(title=fallback_title, text=_trim_blank_lines(text))],
        path="fallback_single",
        llm_detection=llm_detection,
    )


def _slice_into_chapters(
    text: str,
    anchors: list[tuple[int, str]],
) -> list[ParsedChapter]:
    """Build chapters from sorted (offset, title) anchor list."""
    anchors = sorted(anchors, key=lambda x: x[0])
    out: list[ParsedChapter] = []
    for i, (start, title) in enumerate(anchors):
        body_start = start + len(title)
        body_end = anchors[i + 1][0] if i + 1 < len(anchors) else len(text)
        body = _trim_blank_lines(text[body_start:body_end])
        out.append(ParsedChapter(title=title, text=body))
    return out


def _normalise_title(s: str) -> str:
    """Light normalisation for tolerant matching between TOC and body titles."""
    compact = "".join(ch for ch in s if not ch.isspace())
    return _zh_digits_to_ascii(compact).lower()


_ZH_DIGITS = {
    "零": "0", "〇": "0",
    "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
    "六": "6", "七": "7", "八": "8", "九": "9",
    # Keep composites that matter for chapter numbers — "十" handled specially below.
}


def _zh_digits_to_ascii(s: str) -> str:
    """Convert Chinese digits to ASCII where it preserves chapter numbering.

    Handles simple cases (一-九 and common 十X / X十 / X十Y patterns). More
    elaborate numbers (百, 千) are left alone — they are rare in chapter
    titles and better handled by falling back to raw-text match.
    """
    out: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "十":
            prev = out[-1] if out and out[-1].isdigit() else ""
            nxt = _ZH_DIGITS.get(s[i + 1]) if i + 1 < len(s) else None
            if prev and nxt:
                out[-1] = prev + nxt  # X十Y → XY
                i += 2
                continue
            if prev:
                out[-1] = prev + "0"  # X十 → X0
                i += 1
                continue
            if nxt:
                out.append("1" + nxt)  # 十X → 1X
                i += 2
                continue
            out.append("10")
            i += 1
            continue
        out.append(_ZH_DIGITS.get(ch, ch))
        i += 1
    return "".join(out)


def _find_normalised(text: str, needle: str, cursor: int) -> int:
    """Find ``needle`` in ``text[cursor:]`` with tolerant normalisation.

    Tries exact match first; on miss, retries against a
    whitespace-stripped + zh-digit-normalised view. Returns the offset in
    the ORIGINAL text. Returns -1 on total failure.
    """
    idx = text.find(needle, cursor)
    if idx >= 0:
        return idx

    compact_text_chars: list[str] = []
    back_map: list[int] = []
    for i in range(cursor, len(text)):
        ch = text[i]
        if ch.isspace():
            continue
        compact_text_chars.append(ch)
        back_map.append(i)
    compact_text = _zh_digits_to_ascii("".join(compact_text_chars)).lower()
    compact_needle = _normalise_title(needle)
    if not compact_needle:
        return -1

    pos = compact_text.find(compact_needle)
    if pos < 0:
        return -1
    original_pos = back_map[min(pos, len(back_map) - 1)]
    return original_pos
