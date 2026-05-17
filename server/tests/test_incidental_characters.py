"""Tests for the chapter-local incidental character feature.

Background:
- Some chapter speakers don't have proper names — "妇人 said xxx",
  "仆人 reported", "路人 yelled". The LLM identifies these as
  "incidentals" in classify_chapter_characters (Phase A1).
- Service.generate_chapter_meta assigns each a **negative** chapter-
  local id, renames the user-facing ``name`` to "路人1/2/...", and
  stores them ONLY in the chapter snapshot (never characters.json).
- TTS resolution for negative ids reads chapter meta instead of the
  global roster.

This file covers the full path with a stub LLM that emits incidentals,
and the TTS-endpoint negative-id branch with a hand-rolled chapter
meta.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_repository, get_tts_service, get_voice_library
from app.core.enums import Age, Gender, Personality, Tone
from app.core.models import (
    AnalyzedSentence,
    BookMeta,
    Character,
    ChapterMeta,
    ClassifiedCharacters,
    Sentence,
    Speaker,
)
from app.core.narrator import NARRATOR_SPEAKERS
from app.core.service import BookService
from app.core.storage import BookRepository
from app.core.tts_cache import TTSCache
from app.main import create_app
from app.services.llm_stub import StubLLMClient
from app.services.tts import TTSService
from app.services.tts_stub import StubTTSClient


# ---------------------------------------------------------------------------
# A stub LLM that emits a hardcoded incidental + a sentence attributed to it
# ---------------------------------------------------------------------------


class _IncidentalLLM(StubLLMClient):
    """Stub LLM that always emits a hardcoded incidental (``妇人``).
    Segmentation is left to the base stub — once 妇人 is on the list
    of known characters (the service merges in chapter_incidentals
    before B1), the rule-based prefix matcher in
    ``_speaker_in_dialogue_prefix`` picks it up naturally from a
    "妇人说：「...」" line in the chapter text.
    """

    def classify_chapter_characters(self, chapter_text, known):
        base = super().classify_chapter_characters(chapter_text, known)
        # Pull 妇人 out of new_names — in this test it's an incidental,
        # not a real new character. The base stub doesn't know that
        # distinction (it just lists every speaker not in known); we
        # impose the bucket choice here.
        filtered_new = [n for n in base.new_names if n != "妇人"]
        return ClassifiedCharacters(
            new_names=filtered_new,
            incidentals=[
                Character(
                    id=0,  # placeholder — service.py assigns negative id
                    name="妇人",
                    identity="路边妇人",
                    gender=Gender.FEMALE,
                    age=Age.ADULT,
                    personality=[Personality.GENTLE],
                ),
            ],
        )


@pytest.fixture
def chapter_text_with_incidental():
    """Sample chapter where 妇人 has a single dialogue line. The "妇人说：
    「...」" form is what the base stub's prefix matcher recognises, so
    we don't need to override segment_chapter."""
    return (
        "第一章 偶遇\n"
        "萧炎走在街头。\n"
        "萧炎说：「这是哪里？」\n"
        "妇人说：「这里是青阳镇。」\n"
    )


@pytest.fixture
def repo_with_book(tmp_path: Path, chapter_text_with_incidental: str):
    repo = BookRepository(tmp_path / "books")
    book_id = "incident_book"
    meta = BookMeta(book_id=book_id, title="测试书")
    repo.create(meta)
    repo.save_meta(meta)
    repo.write_chapter_text(book_id, 1, chapter_text_with_incidental)
    # Recreate the meta with one chapter entry pointing at the file we
    # just wrote (BookService normally does this during ingest; we're
    # bypassing ingest to avoid the full TXT pipeline in tests).
    from app.core.models import ChapterEntry
    from app.core.storage import chapter_meta_name, chapter_text_name

    meta = BookMeta(
        book_id=book_id,
        title="测试书",
        chapters=[ChapterEntry(
            id=1, title="第一章 偶遇",
            text_file=chapter_text_name(1),
            meta_file=chapter_meta_name(1),
        )],
    )
    repo.save_meta(meta)
    repo.save_characters(book_id, [Character(id=0, name="旁白")])
    return repo, book_id


@pytest.fixture
def service(repo_with_book):
    repo, _ = repo_with_book
    return BookService(repo=repo, llm=_IncidentalLLM())


# ---------------------------------------------------------------------------
# Service-level: full pipeline produces incidentals correctly
# ---------------------------------------------------------------------------


def test_chapter_meta_includes_incidental_with_negative_id(
    service: BookService, repo_with_book,
):
    repo, book_id = repo_with_book
    meta = service.generate_chapter_meta(book_id, 1)

    incidentals = [c for c in meta.characters if c.id < 0]
    assert len(incidentals) == 1
    inc = incidentals[0]
    # First incidental gets id -1. Identification across the system
    # uses ``id < 0`` — the name is just the LLM-emitted descriptor,
    # not parsed for any logic.
    assert inc.id == -1
    # The LLM-given descriptor passes through as-is.
    assert inc.name == "妇人"
    assert inc.identity == "路边妇人"
    # Profile (matcher inputs) survives intact.
    assert inc.gender == Gender.FEMALE
    assert inc.age == Age.ADULT
    assert inc.personality == [Personality.GENTLE]


def test_incidentals_NOT_persisted_to_characters_json(
    service: BookService, repo_with_book,
):
    """Chapter-local — the cumulative roster must never grow with
    one-off speakers. Otherwise the iOS "修改角色音色" list would fill
    up with anonymous 妇人/仆人/店小二 entries that don't survive
    re-analysis. The id-based filter (``c.id > NARRATOR_ID_MAX`` in
    ``list_book_characters``) is what keeps them out — never name
    parsing."""
    repo, book_id = repo_with_book
    service.generate_chapter_meta(book_id, 1)

    roster = repo.load_characters(book_id)
    for c in roster:
        assert c.id >= 0, f"incidental {c.name} (id={c.id}) leaked into roster"


def test_sentence_attributed_to_incidental_uses_negative_id(
    service: BookService, repo_with_book,
):
    """The sentence whose speaker matched the incidental should carry
    its chapter-local negative id, not the narrator fallback (0)."""
    repo, book_id = repo_with_book
    meta = service.generate_chapter_meta(book_id, 1)
    text = repo.read_chapter_text(book_id, 1)
    lines = text.split("\n")

    incidental_id = next(c.id for c in meta.characters if c.id < 0)
    incidental_sentences = [
        s for s in meta.sentences if s.character_id == incidental_id
    ]
    assert incidental_sentences, "no sentence routed to the incidental"

    # Verify the segment text actually came from the chapter line we
    # tagged — guards against locate_sentences silently mismatching.
    s = incidental_sentences[0]
    raw = lines[s.start_line - 1][s.start_col:s.end_col]
    assert "妇人" in raw


def test_collision_with_real_character_drops_incidental(
    repo_with_book, chapter_text_with_incidental: str,
):
    """If A1 emits an incidental whose name collides with a just-merged
    real character (rare LLM glitch), the real character wins and the
    incidental is dropped — otherwise the same name would resolve to
    two ids and reconcile would coin-flip."""
    repo, book_id = repo_with_book

    class _CollidingLLM(_IncidentalLLM):
        def classify_chapter_characters(self, chapter_text, known):
            return ClassifiedCharacters(
                new_names=["妇人"],  # treat 妇人 as a real new character
                incidentals=[
                    Character(  # ALSO emit 妇人 as incidental — collision
                        id=0, name="妇人", identity="路边妇人",
                        gender=Gender.FEMALE, age=Age.ADULT,
                        personality=[Personality.GENTLE],
                    ),
                ],
            )

    svc = BookService(repo=repo, llm=_CollidingLLM())
    meta = svc.generate_chapter_meta(book_id, 1)
    # 妇人 ended up as a real character (positive id), not an incidental.
    assert any(c.name == "妇人" and c.id >= 16 for c in meta.characters)
    assert not any(c.id < 0 for c in meta.characters)


# ---------------------------------------------------------------------------
# TTS endpoint: character_id < 0 resolves via chapter meta
# ---------------------------------------------------------------------------


_LIBRARY = [
    Speaker(
        speaker_id=NARRATOR_SPEAKERS[0],
        gender=Gender.MALE, age=Age.ADULT,
        personality=[Personality.CALM],
    ),
    Speaker(
        speaker_id=NARRATOR_SPEAKERS[1],
        gender=Gender.FEMALE, age=Age.ADULT,
        personality=[Personality.GENTLE],
    ),
    Speaker(
        speaker_id="adult_female_gentle",
        gender=Gender.FEMALE, age=Age.ADULT,
        personality=[Personality.GENTLE, Personality.KIND],
    ),
]


@pytest.fixture
def tts_endpoint_setup(tmp_path: Path):
    """Hand-roll a book + chapter meta containing one incidental, no
    real book characters. The TTS endpoint should resolve a request
    targeting the incidental's negative id by reading the chapter meta
    snapshot."""
    repo = BookRepository(tmp_path / "books")
    book_id = "tts_incident"
    repo.book_dir(book_id).mkdir(parents=True, exist_ok=True)
    (repo.book_dir(book_id) / "meta.json").write_text("{}", encoding="utf-8")
    repo.save_characters(book_id, [Character(id=0, name="旁白")])

    chapter_meta = ChapterMeta(
        sentences=[
            Sentence(
                start_line=1, start_col=0, end_line=1, end_col=10,
                character_id=-1, tone=Tone.NEUTRAL,
            ),
        ],
        characters=[
            Character(
                id=-1, name="妇人", identity="路边妇人",
                gender=Gender.FEMALE, age=Age.ADULT,
                personality=[Personality.GENTLE],
            ),
        ],
    )
    repo.save_chapter_meta(book_id, 1, chapter_meta)

    tts_service = TTSService(
        client=StubTTSClient(), cache=TTSCache(tmp_path / "cache"),
    )
    app = create_app()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_voice_library] = lambda: list(_LIBRARY)
    app.dependency_overrides[get_tts_service] = lambda: tts_service
    with TestClient(app) as client:
        yield client, book_id


def test_tts_endpoint_resolves_negative_id_via_chapter_meta(tts_endpoint_setup):
    client, book_id = tts_endpoint_setup
    r = client.post("/api/tts", json={
        "book_id": book_id,
        "chapter_id": 1,
        "character_id": -1,
        "text": "你好。",
        "tone": "neutral",
    })
    assert r.status_code == 200, r.text
    # Matcher picked the female adult gentle speaker — both the gendered
    # narrator and the dedicated speaker are valid candidates; whichever
    # wins, gender must be female (the incidental's profile).
    assert r.headers.get("X-Speaker-Gender") == "female"


def test_tts_endpoint_negative_id_with_missing_chapter_meta_returns_404(
    tts_endpoint_setup,
):
    client, book_id = tts_endpoint_setup
    r = client.post("/api/tts", json={
        "book_id": book_id,
        "chapter_id": 999,  # no meta saved for this chapter
        "character_id": -1,
        "text": "你好。",
        "tone": "neutral",
    })
    assert r.status_code == 404
    assert "incidental" in r.json()["detail"].lower()


def test_tts_endpoint_negative_id_unknown_in_chapter_meta_returns_503(
    tmp_path: Path,
):
    """Chapter meta exists but doesn't contain the requested negative
    id — voice resolution can't find the character. Should bubble up
    as 503 like other unresolved characters."""
    repo = BookRepository(tmp_path / "books")
    book_id = "stale_neg"
    repo.book_dir(book_id).mkdir(parents=True, exist_ok=True)
    (repo.book_dir(book_id) / "meta.json").write_text("{}", encoding="utf-8")
    repo.save_chapter_meta(book_id, 1, ChapterMeta(sentences=[], characters=[]))

    tts_service = TTSService(
        client=StubTTSClient(), cache=TTSCache(tmp_path / "cache"),
    )
    app = create_app()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_voice_library] = lambda: list(_LIBRARY)
    app.dependency_overrides[get_tts_service] = lambda: tts_service
    with TestClient(app) as client:
        r = client.post("/api/tts", json={
            "book_id": book_id,
            "chapter_id": 1,
            "character_id": -1,
            "text": "你好。",
            "tone": "neutral",
        })
    assert r.status_code == 503


# ---------------------------------------------------------------------------
# list_book_characters filters narrator AND negative ids
# ---------------------------------------------------------------------------


def test_list_book_characters_excludes_incidentals(
    service: BookService, repo_with_book,
):
    """The iOS character-edit screen uses ``list_book_characters``;
    incidentals must not appear there (they're chapter-local, not
    user-editable in the global roster sense)."""
    repo, book_id = repo_with_book
    service.generate_chapter_meta(book_id, 1)

    listed = service.list_book_characters(book_id)
    for c in listed:
        assert c.id >= 16, (
            f"list_book_characters returned id={c.id} ({c.name}); "
            "narrator slots and incidentals should be filtered out"
        )
