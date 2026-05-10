"""TTS service + endpoint: cache, stub synthesis, character_id routing."""
from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_repository, get_tts_service, get_voice_library
from app.core.enums import Age, Gender, Personality, Tone
from app.core.models import Character, Speaker
from app.core.narrator import NARRATOR_SPEAKERS
from app.core.storage import BookRepository
from app.core.tts_cache import TTSCache
from app.main import create_app
from app.services.tts import TTSService, normalize_for_tts
from app.services.tts_stub import StubTTSClient


# Library that includes BOTH predefined narrator voices (so the narrator
# path can resolve) AND a few generic voices for book-character matching.
LIBRARY = [
    Speaker(
        speaker_id=NARRATOR_SPEAKERS[0],  # male narrator
        gender=Gender.MALE, age=Age.ADULT,
        personality=[Personality.CALM, Personality.MATURE],
    ),
    Speaker(
        speaker_id=NARRATOR_SPEAKERS[1],  # female narrator
        gender=Gender.FEMALE, age=Age.ADULT,
        personality=[Personality.GENTLE, Personality.CALM],
    ),
    Speaker(
        speaker_id="youth_female_kind", gender=Gender.FEMALE, age=Age.TEEN,
        personality=[Personality.GENTLE, Personality.KIND],
    ),
    Speaker(
        speaker_id="adult_male_brave", gender=Gender.MALE, age=Age.ADULT,
        personality=[Personality.BRAVE],
    ),
]


@pytest.fixture
def tts_service(tmp_path: Path):
    return TTSService(
        client=StubTTSClient(),
        cache=TTSCache(tmp_path / "tts_cache"),
    )


@pytest.fixture
def repo_with_book(tmp_path: Path):
    """A BookRepository containing one fake book with a character roster
    so the API endpoint's book-character path can resolve."""
    repo = BookRepository(tmp_path / "books")
    book_id = "test_book"
    book_dir = repo.book_dir(book_id)
    book_dir.mkdir(parents=True, exist_ok=True)
    (book_dir / "meta.json").write_text("{}", encoding="utf-8")
    characters = [
        Character(id=0, name="旁白"),
        Character(
            id=16, name="萧炎",
            gender=Gender.MALE, age=Age.ADULT,
            personality=[Personality.BRAVE],
        ),
        Character(
            id=17, name="美杜莎",
            gender=Gender.FEMALE, age=Age.TEEN,
            personality=[Personality.GENTLE, Personality.KIND],
        ),
    ]
    (book_dir / "characters.json").write_text(
        json.dumps([c.model_dump(mode="json") for c in characters]),
        encoding="utf-8",
    )
    return repo, book_id


@pytest.fixture
def api_client(tts_service, repo_with_book):
    repo, _ = repo_with_book
    app = create_app()
    app.dependency_overrides[get_tts_service] = lambda: tts_service
    app.dependency_overrides[get_voice_library] = lambda: list(LIBRARY)
    app.dependency_overrides[get_repository] = lambda: repo
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# TTSService unit tests (speaker resolution lives at the API layer; this
# layer just synthesizes + caches given a Speaker).
# ---------------------------------------------------------------------------


def test_stub_produces_valid_wav(tts_service):
    audio = tts_service.synthesize(
        text="你好，世界。",
        speaker=LIBRARY[0],
        tone=Tone.NEUTRAL,
    )
    import io
    wf = wave.open(io.BytesIO(audio))
    try:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() > 0
    finally:
        wf.close()


def test_cache_hit_skips_client(tmp_path: Path):
    class CountingClient:
        def __init__(self):
            self.calls = 0

        def synthesize(self, **_kwargs):
            self.calls += 1
            return StubTTSClient().synthesize(**_kwargs)

    client = CountingClient()
    service = TTSService(
        client=client, cache=TTSCache(tmp_path / "c"),
    )

    for _ in range(3):
        service.synthesize(
            text="同一句话", speaker=LIBRARY[0], tone=Tone.NEUTRAL,
        )
    assert client.calls == 1


def test_cache_key_ignores_tone(tmp_path: Path):
    """Per design: tone is NOT in the cache key. Same (speaker, text)
    with different tones returns the cached audio of the first call."""
    class CountingClient:
        def __init__(self):
            self.calls = 0

        def synthesize(self, **_kwargs):
            self.calls += 1
            return StubTTSClient().synthesize(**_kwargs)

    client = CountingClient()
    service = TTSService(client=client, cache=TTSCache(tmp_path / "c"))
    service.synthesize(text="x", speaker=LIBRARY[0], tone=Tone.NEUTRAL)
    service.synthesize(text="x", speaker=LIBRARY[0], tone=Tone.HAPPY)
    service.synthesize(text="x", speaker=LIBRARY[0], tone=Tone.ANGRY)
    assert client.calls == 1


class TestNormalizeForTTS:
    def test_horizontal_ellipsis_stripped(self):
        # The bug that motivated this normalisation: U+2026 ×2 in
        # target text caused Qwen3-TTS zero-shot to bleed reference
        # text into the output.
        assert normalize_for_tts("不要跑太远……") == "不要跑太远，"

    def test_normal_punct_untouched(self):
        original = "他说，今天很好。明天呢？"
        assert normalize_for_tts(original) == original


def test_cache_key_normalized_so_variants_share_audio(tmp_path: Path):
    """A sentence ending with `……` and one ending with `，` (the
    normalised form) must hit the same cache entry — the whole point
    of normalising is to maximise reuse across LLM analyses that
    produce slightly different punctuation."""
    class CountingClient:
        def __init__(self):
            self.calls = 0

        def synthesize(self, **_kwargs):
            self.calls += 1
            return StubTTSClient().synthesize(**_kwargs)

    client = CountingClient()
    service = TTSService(client=client, cache=TTSCache(tmp_path / "c"))
    service.synthesize(text="不要跑太远……", speaker=LIBRARY[0], tone=Tone.NEUTRAL)
    service.synthesize(text="不要跑太远，", speaker=LIBRARY[0], tone=Tone.NEUTRAL)
    assert client.calls == 1


def test_cache_miss_on_speaker_or_text_change(tmp_path: Path):
    class CountingClient:
        def __init__(self):
            self.calls = 0

        def synthesize(self, **_kwargs):
            self.calls += 1
            return StubTTSClient().synthesize(**_kwargs)

    client = CountingClient()
    service = TTSService(client=client, cache=TTSCache(tmp_path / "c"))
    service.synthesize(text="x", speaker=LIBRARY[0], tone=Tone.NEUTRAL)
    service.synthesize(text="y", speaker=LIBRARY[0], tone=Tone.NEUTRAL)  # different text
    service.synthesize(text="x", speaker=LIBRARY[1], tone=Tone.NEUTRAL)  # different speaker
    assert client.calls == 3


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


def test_endpoint_narrator_path_resolves_to_predefined_voice(
    api_client, repo_with_book,
):
    """character_id=0 → male narrator (no characters.json lookup needed)."""
    _, book_id = repo_with_book
    r = api_client.post("/api/tts", json={
        "book_id": book_id,
        "chapter_id": 1,
        "character_id": 0,
        "text": "他心中暗道。",
        "tone": "neutral",
    })
    assert r.status_code == 200, r.text
    assert r.headers["x-speaker-id"] == NARRATOR_SPEAKERS[0]
    assert r.headers["content-type"] == "audio/wav"
    assert r.content.startswith(b"RIFF")


def test_endpoint_female_narrator(api_client, repo_with_book):
    _, book_id = repo_with_book
    r = api_client.post("/api/tts", json={
        "book_id": book_id,
        "chapter_id": 1,
        "character_id": 1,
        "text": "她说。",
        "tone": "neutral",
    })
    assert r.status_code == 200
    assert r.headers["x-speaker-id"] == NARRATOR_SPEAKERS[1]


def test_endpoint_book_character_path(api_client, repo_with_book):
    """character_id ≥ 16 looks up the book's characters.json and runs
    attribute matching against the voice library."""
    _, book_id = repo_with_book
    # 萧炎 = MALE / ADULT / [BRAVE] → adult_male_brave wins.
    r = api_client.post("/api/tts", json={
        "book_id": book_id,
        "chapter_id": 1,
        "character_id": 16,
        "text": "我绝不再退缩。",
        "tone": "angry",
    })
    assert r.status_code == 200
    assert r.headers["x-speaker-id"] == "adult_male_brave"


def test_endpoint_book_character_unknown_id_returns_503(
    api_client, repo_with_book,
):
    _, book_id = repo_with_book
    r = api_client.post("/api/tts", json={
        "book_id": book_id,
        "chapter_id": 1,
        "character_id": 999,
        "text": "x",
        "tone": "neutral",
    })
    assert r.status_code == 503
    assert "999" in r.json()["detail"]


def test_endpoint_empty_text_400(api_client, repo_with_book):
    _, book_id = repo_with_book
    r = api_client.post("/api/tts", json={
        "book_id": book_id,
        "chapter_id": 1,
        "character_id": 0,
        "text": "   ",
        "tone": "neutral",
    })
    assert r.status_code == 400


def test_endpoint_empty_voice_library_503(tts_service, repo_with_book):
    repo, book_id = repo_with_book
    app = create_app()
    app.dependency_overrides[get_tts_service] = lambda: tts_service
    app.dependency_overrides[get_voice_library] = lambda: []
    app.dependency_overrides[get_repository] = lambda: repo
    with TestClient(app) as client:
        r = client.post("/api/tts", json={
            "book_id": book_id,
            "chapter_id": 1,
            "character_id": 0,
            "text": "x",
            "tone": "neutral",
        })
    assert r.status_code == 503


# ---------------------------------------------------------------------------
# Cache wipe endpoint
# ---------------------------------------------------------------------------


def test_endpoint_clear_cache_removes_all_files(api_client, tts_service):
    """``DELETE /api/tts/cache`` wipes every ``.m4a`` plus stale tmp
    files; the directory itself stays. After the wipe a follow-up
    synth re-populates it (no permissions corrupted, etc.)."""
    cache_root = tts_service.cache.root
    # Seed the cache with a real synthesis + a stray ``.tmp`` left over
    # from a hypothetical interrupted ``put``.
    tts_service.synthesize(
        text="你好", speaker=LIBRARY[2], tone=Tone.NEUTRAL,
    )
    (cache_root / "stale.m4a.tmp").write_bytes(b"partial")
    files_before = list(cache_root.iterdir())
    assert any(f.suffix == ".m4a" for f in files_before)
    assert any(f.name.endswith(".tmp") for f in files_before)

    r = api_client.delete("/api/tts/cache")
    assert r.status_code == 204

    files_after = list(cache_root.iterdir())
    assert files_after == []
    assert cache_root.exists()  # directory itself survives

    # Subsequent synth still works — directory is reusable.
    audio = tts_service.synthesize(
        text="你好", speaker=LIBRARY[2], tone=Tone.NEUTRAL,
    )
    assert len(audio) > 0
    assert any(f.suffix == ".m4a" for f in cache_root.iterdir())


def test_endpoint_clear_cache_idempotent_on_empty(api_client, tts_service):
    """Wiping an already-empty cache is a no-op, not an error."""
    r = api_client.delete("/api/tts/cache")
    assert r.status_code == 204
    r = api_client.delete("/api/tts/cache")
    assert r.status_code == 204
