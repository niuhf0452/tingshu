"""High-level book service: ingest + lazy chapter metadata.

The chapter-meta flow (see docs/technical-plan.md §2.3):

Phase A — character profile analysis (must complete first):
  A1. ``classify_chapter_characters`` (LLM) lists every non-narrator
      character this chapter touches and tags each as new / evolved.
      Evolved characters carry a full new profile inline; new
      characters carry only their name.
  A2. For each new name, **code** scans every chapter of the book in
      order to find the first 3 line-occurrences of the name and
      gathers ±3 lines of context around each. This guards against
      non-linear reading — a reader who jumps to chapter 50 may land
      on a character whose introduction lives in chapter 5; we want
      the original introduction text feeding profile generation.
  A3. ``profile_new_characters`` (LLM) generates full profiles for the
      new characters using the gathered context windows.
  A4. Merge evolved + new profiles into the cumulative roster
      (``characters.json``). Save.

Phase B — reading-segment analysis:
  B1. ``segment_chapter`` (LLM) splits the chapter into single-speaker
      reading segments using the **post-Phase-A** roster, so segments'
      speaker names should resolve to known characters.
  B2. Reconcile maps speakers to ids; any leftover unknown name is
      mapped to the narrator (per spec — Phase A is supposed to have
      registered everyone; an unknown name here is a glitch and the
      safe fallback is to keep the line audible in narrator voice).
  B3. ``locate_sentences`` resolves each segment to a position range
      in the chapter text. Save chapter meta.

The cumulative character roster lives in a server-only
``characters.json`` next to ``meta.json`` and is never exposed via API;
the App reads characters from each chapter's snapshot embedded in
``ChapterMeta``.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from ..services.llm import LLMClient
from .enums import BookStatus
from .models import (
    BookListItem,
    BookListResponse,
    BookMeta,
    Character,
    CharacterUpdate,
    ChapterEntry,
    ChapterMeta,
)
from .narrator import NARRATOR_ID_MAX
from .nlp.chapters import detect_and_split_chapters
from .nlp.reconcile import merge_character_updates, reconcile_chapter_speakers
from .nlp.sentences import locate_sentences
from .parsers.epub import EpubParseError, parse_epub
from .parsers.txt import ParsedBook, parse_txt
from .storage import BookRepository, chapter_meta_name, chapter_text_name, new_book_id


log = logging.getLogger(__name__)

NARRATOR_CHARACTER_ID = 0
NARRATOR_NAME = "旁白"

# Cross-chapter introduction context for new characters: how many
# line-occurrences to find and how many lines on either side to include.
# 3 occurrences × (1 + 3 + 3) = up to 21 lines per character — enough
# for the LLM to nail down gender / age / identity.
_INTRO_MAX_OCCURRENCES = 3
_INTRO_LINES_AROUND = 3


class BookService:
    def __init__(self, repo: BookRepository, llm: LLMClient):
        self.repo = repo
        self.llm = llm
        # Per-book lock guarding characters.json reads/writes only.
        # **Not** held during LLM calls — different chapters of the
        # same book run their LLM analyses in parallel and only take
        # turns on the brief (~ms) characters.json merge step.
        # Different books → different locks → fully parallel.
        # Same-chapter coalescing is handled by the SSE
        # ``ChapterMetaStreamManager``, not here.
        self._book_locks: dict[str, threading.Lock] = {}
        self._book_locks_guard = threading.Lock()

    def _book_lock(self, book_id: str) -> threading.Lock:
        """Lazily create / fetch the per-book characters.json lock.
        ``_book_locks_guard`` only protects insertion into the dict —
        held for nanoseconds, never under load."""
        with self._book_locks_guard:
            lock = self._book_locks.get(book_id)
            if lock is None:
                lock = threading.Lock()
                self._book_locks[book_id] = lock
            return lock

    # --- phase 1: parse + write chapters, status=processing ---

    def ingest_txt(self, raw: bytes, source_filename: str) -> BookMeta:
        fallback_title = _stem_without_ext(source_filename)

        def _split(text: str, fb_title: str):
            line_count = text.count("\n") + 1
            log.info(
                "txt decoded: filename=%s chars=%d lines=%d — running LLM chapter detection…",
                source_filename, len(text), line_count,
            )
            outcome = detect_and_split_chapters(text, self.llm, fallback_title=fb_title)
            log.info(
                "chapter detection for %s: path=%s chapters=%d",
                source_filename, outcome.path, len(outcome.chapters),
            )
            return outcome.chapters

        parsed = parse_txt(raw, fallback_title=fallback_title, chapter_splitter=_split)
        return self._persist_parsed(parsed, source_filename)

    def ingest_epub(self, raw: bytes, source_filename: str) -> BookMeta:
        """Ingest an EPUB. No LLM call needed — EPUB ships its own TOC."""
        fallback_title = _stem_without_ext(source_filename)
        try:
            parsed = parse_epub(raw, fallback_title=fallback_title)
        except EpubParseError as exc:
            raise ValueError(f"invalid EPUB: {exc}") from exc
        log.info(
            "epub ingest %s: chapters=%d  title=%r  author=%r",
            source_filename, len(parsed.chapters), parsed.title, parsed.author,
        )
        return self._persist_parsed(parsed, source_filename)

    def _persist_parsed(self, parsed: ParsedBook, source_filename: str) -> BookMeta:
        """Shared tail of ingest paths: allocate id, write chapters, persist meta."""
        book_id = new_book_id()
        chapter_entries = [
            ChapterEntry(
                id=idx,
                title=ch.title,
                text_file=chapter_text_name(idx),
                meta_file=chapter_meta_name(idx),
            )
            for idx, ch in enumerate(parsed.chapters, start=1)
        ]
        meta = BookMeta(
            book_id=book_id,
            title=parsed.title,
            author=parsed.author,
            summary="",
            chapters=chapter_entries,
            status=BookStatus.PROCESSING,
            source_filename=source_filename,
        )
        self.repo.create(meta)

        for idx, ch in enumerate(parsed.chapters, start=1):
            self.repo.write_chapter_text(book_id, idx, ch.text)

        # Initialise the cumulative character table with just the narrator.
        self.repo.save_characters(book_id, [
            Character(id=NARRATOR_CHARACTER_ID, name=NARRATOR_NAME),
        ])

        self.repo.save_meta(meta)
        return meta

    # --- phase 2: mark ready ---

    def analyze_book(self, book_id: str) -> BookMeta:
        """Mark the book as ``ready``. No upfront character analysis —
        the cumulative roster grows lazily during chapter analysis."""
        with self._book_lock(book_id):
            meta = self.repo.load_meta(book_id)
            meta.status = BookStatus.READY
            self.repo.save_meta(meta)
        return meta

    # --- phase 3: per-chapter metadata (lazy) ---

    def get_or_generate_chapter_meta(
        self,
        book_id: str,
        chapter_id: int,
    ) -> ChapterMeta:
        cached = self.repo.load_chapter_meta(book_id, chapter_id)
        if cached is not None:
            return cached
        return self.generate_chapter_meta(book_id, chapter_id)

    def generate_chapter_meta(
        self,
        book_id: str,
        chapter_id: int,
    ) -> ChapterMeta:
        """Sequential A1→A2→A3→B1 pipeline. See module docstring."""
        book_lock = self._book_lock(book_id)
        with book_lock:
            meta = self.repo.load_meta(book_id)
            if not any(c.id == chapter_id for c in meta.chapters):
                raise FileNotFoundError(f"chapter {chapter_id} not found in book {book_id}")
            known = self.repo.load_characters(book_id)

        chapter_text = self.repo.read_chapter_text(book_id, chapter_id)
        log.info(
            "analyze_chapter: book=%s ch=%d chars=%d known_chars=%d",
            book_id, chapter_id, len(chapter_text), len(known),
        )
        t0 = time.monotonic()

        # ---- Phase A: character analysis ----
        # A1: classify characters this chapter touches.
        classified = self.llm.classify_chapter_characters(chapter_text, known)
        a1_elapsed = time.monotonic() - t0

        # A2: gather cross-chapter introduction context for new names.
        name_to_contexts: dict[str, list[str]] = {}
        if classified.new_names:
            book_meta = meta  # already loaded under lock above
            for name in classified.new_names:
                name_to_contexts[name] = self._find_character_introductions(
                    book_id=book_id,
                    book_meta=book_meta,
                    name=name,
                )

        # A3: generate profiles for new characters.
        t_a3 = time.monotonic()
        new_profiles = self.llm.profile_new_characters(name_to_contexts, known)
        a3_elapsed = time.monotonic() - t_a3

        # A4: merge evolved + new into the roster, save. Incidentals are
        # **not** merged — they're chapter-local one-offs.
        with book_lock:
            # Reload in case another concurrent chapter job updated the
            # global table while our LLM calls were in flight.
            known = self.repo.load_characters(book_id)
            updated_known, new_count, evolved_count = merge_character_updates(
                known=known,
                updates=list(classified.evolved) + list(new_profiles),
            )
            self.repo.save_characters(book_id, updated_known)

        # Build chapter-local incidentals: descriptor-named one-off
        # speakers from A1, with chapter-local **negative** ids. The
        # ``id < 0`` test is the canonical way to identify an
        # incidental anywhere in the system — never the name. The
        # name is just whatever descriptor the LLM emitted (妇人, 仆人,
        # 店小二, …) and is purely user-facing.
        #
        # Defence: if the LLM emits an incidental whose name collides
        # with a real character (just-merged into the roster), drop the
        # incidental — the real character wins, since it has a stable
        # cross-chapter id.
        roster_names = {c.name for c in updated_known}
        chapter_incidentals: list[Character] = []
        for profile in classified.incidentals:
            if not profile.name or profile.name in roster_names:
                continue
            chapter_incidentals.append(Character(
                id=-(len(chapter_incidentals) + 1),
                name=profile.name,
                identity=profile.identity,
                gender=profile.gender,
                age=profile.age,
                personality=list(profile.personality),
            ))

        # ---- Phase B: segmentation ----
        t_b = time.monotonic()
        analyzed_sentences = self.llm.segment_chapter(
            chapter_text, updated_known + chapter_incidentals,
        )
        b1_elapsed = time.monotonic() - t_b

        with book_lock:
            # Reload once more so the speaker→id resolution sees the
            # latest roster (concurrent chapters may have added more).
            # Incidentals are not in the roster — pass them alongside.
            roster = self.repo.load_characters(book_id)
            speaker_to_id, chapter_chars = reconcile_chapter_speakers(
                known=roster + chapter_incidentals,
                speakers=[s.speaker for s in analyzed_sentences],
            )
            sentences = locate_sentences(
                chapter_text, analyzed_sentences, speaker_to_id,
            )
            chapter_meta = ChapterMeta(sentences=sentences, characters=chapter_chars)
            self.repo.save_chapter_meta(book_id, chapter_id, chapter_meta)

        incidental_count = sum(1 for c in chapter_chars if c.id < 0)
        log.info(
            "analyze_chapter done: book=%s ch=%d sentences=%d "
            "chapter_chars=%d new=%d evolved=%d incidental=%d "
            "wall_a1=%.1fs wall_a3=%.1fs wall_b1=%.1fs total=%.1fs",
            book_id, chapter_id, len(sentences), len(chapter_chars),
            new_count, evolved_count, incidental_count,
            a1_elapsed, a3_elapsed, b1_elapsed, time.monotonic() - t0,
        )
        return chapter_meta

    def chapter_meta_exists(self, book_id: str, chapter_id: int) -> bool:
        return self.repo.chapter_meta_path(book_id, chapter_id).exists()

    def _find_character_introductions(
        self,
        *,
        book_id: str,
        book_meta: BookMeta,
        name: str,
    ) -> list[str]:
        """Scan all chapters in order for the first
        ``_INTRO_MAX_OCCURRENCES`` lines mentioning ``name``. For each
        match return a context window of ±``_INTRO_LINES_AROUND`` lines,
        prefixed with ``[第N章, 行M]`` so the LLM sees source attribution.

        Stops as soon as enough occurrences are collected — does NOT
        read every chapter for a common name.
        """
        if not name:
            return []
        windows: list[str] = []
        for chapter in book_meta.chapters:
            if len(windows) >= _INTRO_MAX_OCCURRENCES:
                break
            try:
                text = self.repo.read_chapter_text(book_id, chapter.id)
            except FileNotFoundError:
                continue
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if name not in line:
                    continue
                start = max(0, i - _INTRO_LINES_AROUND)
                end = min(len(lines), i + _INTRO_LINES_AROUND + 1)
                window = "\n".join(lines[start:end])
                windows.append(f"[第{chapter.id}章, 行{i + 1}]\n{window}")
                if len(windows) >= _INTRO_MAX_OCCURRENCES:
                    break
        return windows

    # --- read APIs -----------------------------------------------------------

    def list_books(self) -> BookListResponse:
        items: list[BookListItem] = []
        for book_id in self.repo.list_book_ids():
            try:
                meta = self.repo.load_meta(book_id)
            except Exception:
                continue
            items.append(BookListItem(
                book_id=meta.book_id,
                title=meta.title,
                author=meta.author,
                status=meta.status,
                chapter_count=len(meta.chapters),
            ))
        return BookListResponse(books=items)

    def get_meta(self, book_id: str) -> BookMeta:
        return self.repo.load_meta(book_id)

    # --- character roster (book-level, user-editable) ----------------------

    def list_book_characters(self, book_id: str) -> list[Character]:
        """Return the cumulative roster, narrator slots filtered out.

        Narrator-range ids (≤15) aren't real characters — they're system
        slots whose voice is selected via the app's separate "旁白音色"
        picker. The app's character-voice screen is for book characters
        only, so we only expose ``id > NARRATOR_ID_MAX`` here.

        Read does not take the book lock — readers seeing a partially-
        merged roster mid-analysis is fine (the next refresh shows the
        finished state) and skipping the lock means the list call never
        waits behind a multi-second LLM analysis. Writers use the lock
        and an atomic file swap, so a reader still sees a consistent
        snapshot, just possibly a stale one.
        """
        roster = self.repo.load_characters(book_id)
        return [c for c in roster if c.id > NARRATOR_ID_MAX]

    def update_book_character(
        self,
        book_id: str,
        character_id: int,
        update: CharacterUpdate,
    ) -> Character:
        """Apply a partial update to one character in ``characters.json``.

        Holds the per-book lock during read-modify-write so concurrent
        chapter analysis can't lose either side's edit. Per spec the
        write **waits** on contention but never fails: chapter analysis
        also uses ``_book_lock(book_id)`` (see ``generate_chapter_meta``)
        and only holds it briefly for the merge step, so the user's tap
        delays at most one merge cycle. Different chapters' analyses
        already serialise on this same lock.

        Raises:
            ValueError: ``character_id`` is in the narrator-reserved
                range (≤ 15) — those aren't editable here.
            KeyError: ``character_id`` not present in the roster.
        """
        if character_id <= NARRATOR_ID_MAX:
            raise ValueError(
                f"character_id {character_id} is reserved for narrator slots; "
                "edit narrator voice via the app settings instead",
            )
        with self._book_lock(book_id):
            roster = self.repo.load_characters(book_id)
            for i, ch in enumerate(roster):
                if ch.id != character_id:
                    continue
                # Build the new Character from the existing one + only
                # the fields the client sent. ``model_copy(update=…)``
                # silently ignores ``None`` only when the field already
                # accepts None — it doesn't here, so filter explicitly.
                changes = update.model_dump(exclude_unset=True)
                if not changes:
                    return ch  # no-op patch — return unchanged
                roster[i] = ch.model_copy(update=changes)
                self.repo.save_characters(book_id, roster)
                log.info(
                    "character updated: book=%s id=%d changes=%s",
                    book_id, character_id, sorted(changes.keys()),
                )
                return roster[i]
        raise KeyError(f"character {character_id} not in book {book_id}")

    # --- delete --------------------------------------------------------------

    def delete_book(self, book_id: str) -> None:
        """Recursively remove the book directory (meta.json, characters.json,
        chapters/*). See docs/technical-plan.md §2.2.1.
        """
        with self._book_lock(book_id):
            self.repo.delete(book_id)
        # Drop the lock entry too — the book is gone.
        with self._book_locks_guard:
            self._book_locks.pop(book_id, None)
        log.info("deleted book %s", book_id)


def _stem_without_ext(filename: str) -> str:
    return Path(filename).stem or "Untitled"
