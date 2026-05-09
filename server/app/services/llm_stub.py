"""Deterministic LLM stub used for tests and local development without MLX.

Behaviour:

- ``detect_chapters`` either returns a caller-pinned override or applies
  a small regex heuristic over the opening.
- ``classify_chapter_characters`` walks the chapter text, identifies any
  speaker name that isn't in ``known_characters``, and lists them in
  ``new_names``. ``evolved`` is always empty (the stub doesn't try to
  detect character growth).
- ``profile_new_characters`` returns a deterministic profile (SHA-1
  based) per name so tests can exercise the discovery + merge path
  without a real LLM.
- ``segment_chapter`` runs a rule-based segmenter and assigns a free-form
  speaker to each segment: dialogue is attributed to a known character
  name mentioned in the prefix, falling back to the most recently
  mentioned speaker. Narration uses ``"旁白"``.

The methods are independent — same shape as the production LLMClient —
so test code paths exercise the sequential A1→A3→B1 pipeline.
"""
from __future__ import annotations

import hashlib
import re as _re
from dataclasses import dataclass

from ..core.enums import Age, Gender, Personality, Tone
from ..core.models import (
    AnalyzedSentence,
    ChapterDetection,
    Character,
    ClassifiedCharacters,
)
from .llm import LLMClient


# Canonical Chinese chapter heading regex. The stub returns this as
# ``chapter_pattern`` so the chapter detector's non-TOC path works.
_STUB_HEADING_PATTERN = (
    r"^\s*第\s*[零〇一二三四五六七八九十百千0-9]+\s*[章回节卷篇]"
    r"(?:\s*\S.*)?\s*$"
)
_STUB_HEADING_RE = _re.compile(_STUB_HEADING_PATTERN, _re.MULTILINE)


@dataclass
class StubLLMClient(LLMClient):
    """Configurable stub for tests."""

    chapter_detection_override: ChapterDetection | None = None

    # --- LLMClient ---

    def detect_chapters(
        self,
        opening_text: str,
        *,
        known_titles: list[str] | None = None,  # noqa: ARG002 — single-shot
    ) -> ChapterDetection:
        if self.chapter_detection_override is not None:
            return self.chapter_detection_override
        matches = list(_STUB_HEADING_RE.finditer(opening_text))
        if len(matches) >= 2:
            return ChapterDetection(
                has_toc=False,
                preface_titles=[],
                first_chapter_title=matches[0].group(0).strip(),
                chapter_pattern=_STUB_HEADING_PATTERN,
            )
        return ChapterDetection()

    def classify_chapter_characters(
        self,
        chapter_text: str,
        known_characters: list[Character],
    ) -> ClassifiedCharacters:
        """Walk the chapter, find any non-narrator speaker name not
        already in ``known_characters``, and return them as ``new_names``.
        ``evolved`` is always empty — the stub doesn't try to detect
        character growth (real LLM does)."""
        known_names = {c.name for c in known_characters if c.id != 0}
        new_speakers = self._discover_new_speakers(chapter_text, known_names)
        return ClassifiedCharacters(new_names=list(new_speakers), evolved=[])

    def profile_new_characters(
        self,
        name_to_contexts: dict[str, list[str]],
        known_characters: list[Character],  # noqa: ARG002 — stub ignores
    ) -> list[Character]:
        """Emit a deterministic profile (SHA-1 of name) for each
        new character. Context windows are accepted but ignored —
        the stub's profile is name-derived for test stability."""
        return [_deterministic_character(name) for name in name_to_contexts]

    def segment_chapter(
        self,
        chapter_text: str,
        known_characters: list[Character],
    ) -> list[AnalyzedSentence]:
        """Rule-based segmenter. Speakers resolve to known character
        names by prefix matching; segments without a confident match
        fall through to the last seen speaker (or 旁白)."""
        sentences: list[AnalyzedSentence] = []
        known_names = {c.name for c in known_characters if c.id != 0}
        last_speaker = "旁白"

        for piece in _rule_based_segment(chapter_text):
            is_dialogue = _looks_like_dialogue(piece)
            if not is_dialogue:
                for name in known_names:
                    if name in piece:
                        last_speaker = name
                sentences.append(AnalyzedSentence(
                    text=piece, speaker="旁白", tone=Tone.NEUTRAL,
                ))
                continue
            prefix_speaker = _speaker_in_dialogue_prefix(piece, list(known_names))
            speaker = prefix_speaker or last_speaker
            last_speaker = speaker
            sentences.append(AnalyzedSentence(
                text=piece, speaker=speaker, tone=Tone.NEUTRAL,
            ))

        return sentences

    def _discover_new_speakers(
        self, chapter_text: str, known_names: set[str],
    ) -> dict[str, None]:
        """Shared helper for ``classify_chapter_characters``: walk the
        chapter, attribute dialogue, return ordered set of speaker
        names not in ``known_names``."""
        last_speaker = "旁白"
        new_speakers: dict[str, None] = {}  # ordered set
        for piece in _rule_based_segment(chapter_text):
            is_dialogue = _looks_like_dialogue(piece)
            if not is_dialogue:
                for name in known_names:
                    if name in piece:
                        last_speaker = name
                continue
            prefix_speaker = _speaker_in_dialogue_prefix(piece, list(known_names))
            speaker = prefix_speaker or last_speaker
            last_speaker = speaker
            if speaker not in known_names and speaker != "旁白":
                new_speakers[speaker] = None
        return new_speakers


# --- rule-based chapter segmenter (stub-only) ---

_TERMINATORS = frozenset("。！？…!?")
_BRACKET_PAIRS = {"「": "」", "『": "』", "“": "”"}
_OPENS = frozenset(_BRACKET_PAIRS.keys())
_CLOSES = frozenset(_BRACKET_PAIRS.values())


def _rule_based_segment(text: str) -> list[str]:
    if not text:
        return []
    pieces: list[str] = []
    for line in text.split("\n"):
        pieces.extend(_split_line(line))
    return pieces


def _split_line(line: str) -> list[str]:
    n = len(line)
    if n == 0:
        return []
    out: list[str] = []
    depth = 0
    start = 0
    i = 0

    def emit(a: int, b: int) -> None:
        if a >= b:
            return
        slice_ = line[a:b]
        if any(c not in _TERMINATORS and not c.isspace() for c in slice_):
            out.append(slice_.strip())

    while i < n:
        ch = line[i]
        if ch in _OPENS:
            depth += 1
            i += 1
            continue
        if ch in _CLOSES:
            depth = max(0, depth - 1)
            if depth == 0 and i > 0 and line[i - 1] in _TERMINATORS:
                emit(start, i + 1)
                start = i + 1
            i += 1
            continue
        if ch in _TERMINATORS and depth == 0:
            j = i + 1
            while j < n and line[j] in _TERMINATORS:
                j += 1
            if j < n and line[j] in _CLOSES:
                depth = max(0, depth - 1)
                j += 1
            emit(start, j)
            start = j
            i = j
            continue
        i += 1
    emit(start, n)
    return out


def _looks_like_dialogue(text: str) -> bool:
    return bool(text) and (text[0] in _OPENS or text[-1] in _CLOSES)


_SPEECH_VERBS = ("说道", "说", "道", "问道", "问", "喊", "叫", "笑道", "叹道", "开口")


def _speaker_in_dialogue_prefix(text: str, known_names: list[str]) -> str | None:
    bracket_idx = next((i for i, c in enumerate(text) if c in _OPENS), -1)
    if bracket_idx <= 0:
        return None
    prefix = text[:bracket_idx]

    for name in known_names:
        if name in prefix:
            return name

    for verb in _SPEECH_VERBS:
        pos = prefix.find(verb)
        if pos < 0:
            continue
        start = pos
        while start > 0 and _is_cjk(prefix[start - 1]) and pos - start < 3:
            start -= 1
        candidate = prefix[start:pos]
        if 2 <= len(candidate) <= 3 and all(_is_cjk(c) for c in candidate):
            return candidate
    return None


def _is_cjk(ch: str) -> bool:
    return "一" <= ch <= "鿿"


_GENDERS = list(Gender)
_AGES = list(Age)
_PERSONALITIES = list(Personality)


def _deterministic_character(name: str) -> Character:
    """Pick attributes based on SHA-1 of the name — stable across runs.

    ``id`` is left as 0; the BookService merge step assigns the real id
    when adding to the global character table.
    """
    digest = hashlib.sha1(name.encode("utf-8")).digest()
    gender = _GENDERS[digest[0] % len(_GENDERS)]
    age = _AGES[digest[1] % len(_AGES)]
    count = 1 + (digest[2] % 2)
    tags: list[Personality] = []
    for i in range(count):
        tags.append(_PERSONALITIES[digest[3 + i] % len(_PERSONALITIES)])
    seen: set[Personality] = set()
    unique_tags = [t for t in tags if not (t in seen or seen.add(t))]  # type: ignore[func-returns-value]
    return Character(
        id=0,
        name=name,
        identity=f"stub-identity-of-{name}",
        gender=gender,
        age=age,
        personality=unique_tags,
    )
