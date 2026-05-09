"""TXT book parser: encoding detection + dev-scaffold chapter splitting.

The regex-based ``split_chapters`` is a dev-time convenience for tests
that don't want to spin up an LLM. Production import flow injects the
LLM-driven splitter from ``core.nlp.chapters.detect_and_split_chapters``
via the ``chapter_splitter`` argument to ``parse_txt``. Per §2.4.1 the
production pipeline no longer uses this regex as a fallback — when the
LLM fails, the whole file is wrapped as a single chapter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

import chardet


# Matches:  第一章 / 第1章 / 第 十二 回 / 第 42 节 / 第123卷 ...
# Dev scaffolding only — production chapter detection is LLM-driven
# (see core.nlp.chapters). Kept strict (``\s+`` before the optional title)
# so body lines like ``第二章开始。`` don't false-match as headings.
_CHAPTER_TITLE_RE = re.compile(
    r"""^\s*
        第\s*[零〇一二三四五六七八九十百千0-9]+\s*
        [章回节卷篇]
        (?:\s+\S.*)?
        \s*$
    """,
    re.VERBOSE,
)


@dataclass
class ParsedChapter:
    title: str
    text: str  # UTF-8 text, LF line endings, no trailing blank lines


@dataclass
class ParsedBook:
    title: str
    author: str
    chapters: list[ParsedChapter]


def decode_text(raw: bytes) -> str:
    """Decode with a priority chain. Chinese TXT novels are almost always
    UTF-8 or GB18030 — try those strictly before falling back to chardet
    (which misdetects short Chinese samples as TIS-620/EUC-TW).
    Normalises line endings to LF.
    """
    if not raw:
        return ""

    # BOM shortcuts.
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
        text = raw.decode("utf-8", errors="replace")
        return _normalise_newlines(text)

    for enc in ("utf-8", "gb18030"):
        try:
            return _normalise_newlines(raw.decode(enc))
        except UnicodeDecodeError:
            continue

    # Last resort: chardet + replace errors.
    detection = chardet.detect(raw)
    encoding = (detection.get("encoding") or "utf-8").lower()
    if encoding in {"gb2312", "gbk"}:
        encoding = "gb18030"
    try:
        text = raw.decode(encoding, errors="replace")
    except LookupError:
        text = raw.decode("utf-8", errors="replace")
    return _normalise_newlines(text)


def _normalise_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def split_chapters(text: str, fallback_title: str) -> list[ParsedChapter]:
    lines = text.split("\n")
    heading_idx: list[tuple[int, str]] = [
        (i, line.strip()) for i, line in enumerate(lines)
        if _CHAPTER_TITLE_RE.match(line)
    ]

    if not heading_idx:
        body = _trim_blank_lines(text)
        return [ParsedChapter(title=fallback_title, text=body)]

    chapters: list[ParsedChapter] = []
    # Content before the first heading is treated as preamble and attached to
    # the first chapter (common pattern: author's note → 第一章).
    for idx, (line_no, title) in enumerate(heading_idx):
        body_start = line_no + 1
        body_end = heading_idx[idx + 1][0] if idx + 1 < len(heading_idx) else len(lines)
        body_lines = lines[body_start:body_end]
        body = _trim_blank_lines("\n".join(body_lines))
        chapters.append(ParsedChapter(title=title, text=body))

    return chapters


def _trim_blank_lines(text: str) -> str:
    return text.strip("\n")


ChapterSplitter = Callable[[str, str], list[ParsedChapter]]


def parse_txt(
    raw: bytes,
    fallback_title: str,
    author: str = "",
    *,
    chapter_splitter: Optional[ChapterSplitter] = None,
) -> ParsedBook:
    """Decode bytes and split into chapters.

    ``chapter_splitter`` accepts ``(text, fallback_title)`` and returns a
    list of ``ParsedChapter``. Defaults to the hardcoded-regex splitter;
    production code injects the LLM-backed splitter from
    ``core.nlp.chapters.detect_and_split_chapters``.
    """
    text = decode_text(raw)
    splitter = chapter_splitter or split_chapters
    chapters = splitter(text, fallback_title)
    return ParsedBook(title=fallback_title, author=author, chapters=chapters)
