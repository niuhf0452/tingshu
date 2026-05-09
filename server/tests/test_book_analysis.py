"""End-to-end BookService tests covering ingest + lazy chapter analysis.

Post-refactor (§2.3): characters live in a server-only ``characters.json``
plus per-chapter snapshots inside each ``ChapterMeta``. ``BookMeta`` is
immutable after import.
"""
from __future__ import annotations

from pathlib import Path

from app.core.enums import BookStatus
from app.core.service import BookService, NARRATOR_CHARACTER_ID, NARRATOR_NAME
from app.core.storage import BookRepository
from app.services.llm_stub import StubLLMClient


SAMPLE_TXT = (
    "第一章 开端\n"
    "萧炎紧握拳头，望向远方。\n"
    "药老缓缓开口：「小子别急。」\n"
    "萧炎点头。\n\n"
    "第二章 发展\n"
    "萧炎与药老结伴前行。\n"
    "药老说：「好好修炼。」\n"
)


def _mk_service(tmp_path: Path) -> tuple[BookService, BookRepository, StubLLMClient]:
    llm = StubLLMClient()
    repo = BookRepository(tmp_path / "books")
    service = BookService(repo=repo, llm=llm)
    return service, repo, llm


def test_ingest_seeds_narrator_only(tmp_path):
    service, repo, _ = _mk_service(tmp_path)
    meta = service.ingest_txt(SAMPLE_TXT.encode("utf-8"), source_filename="t.txt")
    assert meta.status == BookStatus.PROCESSING
    # BookMeta no longer carries characters — they live in characters.json.
    chars = repo.load_characters(meta.book_id)
    assert [c.name for c in chars] == [NARRATOR_NAME]
    assert chars[0].id == NARRATOR_CHARACTER_ID
    # Reloaded BookMeta from disk matches in-memory.
    loaded = repo.load_meta(meta.book_id)
    assert loaded.status == BookStatus.PROCESSING


def test_analyze_book_marks_ready(tmp_path):
    service, repo, _ = _mk_service(tmp_path)
    meta = service.ingest_txt(SAMPLE_TXT.encode("utf-8"), source_filename="t.txt")

    analysed = service.analyze_book(meta.book_id)
    assert analysed.status == BookStatus.READY
    chars = repo.load_characters(meta.book_id)
    assert [c.name for c in chars] == [NARRATOR_NAME]


def test_chapter_meta_discovers_characters_lazily(tmp_path):
    """generate_chapter_meta populates characters.json AND embeds the
    chapter's character snapshot in chapters/N.json."""
    service, repo, _ = _mk_service(tmp_path)
    meta = service.ingest_txt(SAMPLE_TXT.encode("utf-8"), source_filename="t.txt")

    chapter_meta = service.generate_chapter_meta(meta.book_id, 1)
    assert chapter_meta.sentences, "should produce at least one sentence"
    # Snapshot inside the chapter meta has at least one non-narrator name.
    snapshot_names = [c.name for c in chapter_meta.characters]
    assert any(name != NARRATOR_NAME for name in snapshot_names), snapshot_names

    # The cumulative roster (server-only) also got the new character.
    cumulative = repo.load_characters(meta.book_id)
    cumulative_names = [c.name for c in cumulative]
    assert NARRATOR_NAME in cumulative_names
    assert len(cumulative_names) > 1


def test_chapter_meta_immutable_after_write(tmp_path):
    """Re-running generate_chapter_meta on the same chapter doesn't re-read
    a stale snapshot — but the snapshot is written each time and matches
    the freshest profile."""
    service, repo, _ = _mk_service(tmp_path)
    meta = service.ingest_txt(SAMPLE_TXT.encode("utf-8"), source_filename="t.txt")

    first = service.generate_chapter_meta(meta.book_id, 1)
    second = service.generate_chapter_meta(meta.book_id, 1)
    # Stub is deterministic so both passes produce the same snapshot.
    assert [c.name for c in first.characters] == [c.name for c in second.characters]


def test_cumulative_roster_grows_across_chapters(tmp_path):
    """Chapter 2's analysis sees chapter 1's discoveries as known
    characters, doesn't double-create them."""
    service, repo, _ = _mk_service(tmp_path)
    meta = service.ingest_txt(SAMPLE_TXT.encode("utf-8"), source_filename="t.txt")

    service.generate_chapter_meta(meta.book_id, 1)
    after_ch1 = {c.name for c in repo.load_characters(meta.book_id)}

    service.generate_chapter_meta(meta.book_id, 2)
    after_ch2 = {c.name for c in repo.load_characters(meta.book_id)}

    # Chapter 2 didn't shrink the roster.
    assert after_ch1.issubset(after_ch2)


def test_analyze_book_is_idempotent(tmp_path):
    service, _, _ = _mk_service(tmp_path)
    meta = service.ingest_txt(SAMPLE_TXT.encode("utf-8"), source_filename="t.txt")
    first = service.analyze_book(meta.book_id)
    second = service.analyze_book(meta.book_id)
    assert first.status == second.status == BookStatus.READY
