"""File-system book repository.

Layout (see docs/technical-plan.md §2.1.1):

    data/books/<book_id>/
        meta.json              (book-level static info; immutable after import)
        characters.json        (server-only: cumulative character roster
                                with name → id + current profile across
                                all chapters analysed so far)
        cover.jpg              (optional)
        chapters/
            0001.txt
            0001.json           (lazy-generated; embeds per-chapter
                                 ``characters`` snapshot)
            ...

The ``characters.json`` is **internal**: never bundled into the download
zip, never served via the API. It exists so the LLM can be told the
known character table when analysing each chapter (cross-chapter alias
recognition + id stability). The App reads characters from each chapter
meta's snapshot.
"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from .models import BookMeta, Character, ChapterMeta


CHAPTER_ID_WIDTH = 4  # 0001, 0002, ... up to 9999


def new_book_id() -> str:
    # Short, URL-safe, collision-resistant enough for personal library scale.
    return uuid.uuid4().hex[:12]


def chapter_text_name(chapter_id: int) -> str:
    return f"chapters/{chapter_id:0{CHAPTER_ID_WIDTH}d}.txt"


def chapter_meta_name(chapter_id: int) -> str:
    return f"chapters/{chapter_id:0{CHAPTER_ID_WIDTH}d}.json"


class BookRepository:
    """Thin wrapper over the books root directory. No locking yet — single-writer."""

    def __init__(self, books_dir: Path):
        self.books_dir = books_dir
        self.books_dir.mkdir(parents=True, exist_ok=True)

    # --- paths ---

    def book_dir(self, book_id: str) -> Path:
        return self.books_dir / book_id

    def meta_path(self, book_id: str) -> Path:
        return self.book_dir(book_id) / "meta.json"

    def chapters_dir(self, book_id: str) -> Path:
        return self.book_dir(book_id) / "chapters"

    def chapter_text_path(self, book_id: str, chapter_id: int) -> Path:
        return self.book_dir(book_id) / chapter_text_name(chapter_id)

    def chapter_meta_path(self, book_id: str, chapter_id: int) -> Path:
        return self.book_dir(book_id) / chapter_meta_name(chapter_id)

    # --- existence / enumeration ---

    def exists(self, book_id: str) -> bool:
        return self.meta_path(book_id).exists()

    def list_book_ids(self) -> list[str]:
        return sorted(p.name for p in self.books_dir.iterdir()
                      if p.is_dir() and (p / "meta.json").exists())

    # --- read / write ---

    def create(self, meta: BookMeta) -> None:
        self.book_dir(meta.book_id).mkdir(parents=True, exist_ok=True)
        self.chapters_dir(meta.book_id).mkdir(parents=True, exist_ok=True)
        self.save_meta(meta)

    def load_meta(self, book_id: str) -> BookMeta:
        path = self.meta_path(book_id)
        if not path.exists():
            raise FileNotFoundError(f"book meta not found: {book_id}")
        return BookMeta.model_validate_json(path.read_text(encoding="utf-8"))

    def save_meta(self, meta: BookMeta) -> None:
        path = self.meta_path(meta.book_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")

    def write_chapter_text(self, book_id: str, chapter_id: int, text: str) -> None:
        path = self.chapter_text_path(book_id, chapter_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def read_chapter_text(self, book_id: str, chapter_id: int) -> str:
        return self.chapter_text_path(book_id, chapter_id).read_text(encoding="utf-8")

    def load_chapter_meta(self, book_id: str, chapter_id: int) -> ChapterMeta | None:
        path = self.chapter_meta_path(book_id, chapter_id)
        if not path.exists():
            return None
        return ChapterMeta.model_validate_json(path.read_text(encoding="utf-8"))

    def save_chapter_meta(self, book_id: str, chapter_id: int, meta: ChapterMeta) -> None:
        path = self.chapter_meta_path(book_id, chapter_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")

    # --- characters (server-only cumulative roster) ---

    def characters_path(self, book_id: str) -> Path:
        return self.book_dir(book_id) / "characters.json"

    def load_characters(self, book_id: str) -> list[Character]:
        """Return the cumulative character roster for the book. Empty list
        if the file doesn't exist yet (e.g. fresh import — only narrator
        will be added when the first chapter is analysed)."""
        path = self.characters_path(book_id)
        if not path.exists():
            return []
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return []
        data = json.loads(raw)
        return [Character.model_validate(item) for item in data]

    def save_characters(self, book_id: str, characters: list[Character]) -> None:
        path = self.characters_path(book_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [c.model_dump(mode="json") for c in characters]
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # --- delete ---

    def delete(self, book_id: str) -> None:
        """Recursively remove the book directory.

        Raises ``FileNotFoundError`` if the book doesn't exist so callers
        can map that to HTTP 404 instead of silently succeeding.
        """
        path = self.book_dir(book_id)
        if not path.exists():
            raise FileNotFoundError(f"book not found: {book_id}")
        shutil.rmtree(path)
