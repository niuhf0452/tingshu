"""Voice library: load speakers.json + match speakers to character attributes.

Matching strategy (see docs/technical-plan.md §2.5):

- Gender and age are hard constraints in spec; in practice we relax to a
  distance metric so the service always returns *some* speaker rather than
  failing hard on an incomplete library. The ranking is deterministic.
- Personality is a soft constraint — intersection size with the character's
  tags breaks ties.

Ranking key (lower is better, first-mismatch wins):

    (gender_distance, age_distance, -personality_overlap, speaker_id)
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from .enums import Age, Gender, Personality
from .models import Character, Speaker
from .narrator import is_narrator_id, speaker_id_for_narrator


_AGE_ORDER = [Age.CHILD, Age.TEEN, Age.YOUTH, Age.ADULT, Age.ELDER]
_AGE_INDEX = {a: i for i, a in enumerate(_AGE_ORDER)}

_SPEAKER_LIST_ADAPTER = TypeAdapter(list[Speaker])


def load_voice_library(path: Path) -> list[Speaker]:
    """Load ``speakers.json`` from disk. Missing file returns an empty list
    (callers can decide whether to treat that as a hard error).
    """
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _SPEAKER_LIST_ADAPTER.validate_python(raw)


def gender_distance(target: Gender, candidate: Gender) -> int:
    if target == candidate:
        return 0
    if Gender.NEUTRAL in (target, candidate):
        return 1
    return 2  # male vs female


def age_distance(target: Age, candidate: Age) -> int:
    return abs(_AGE_INDEX[target] - _AGE_INDEX[candidate])


def personality_overlap(
    target: list[Personality] | set[Personality],
    candidate: list[Personality] | set[Personality],
) -> int:
    return len(set(target) & set(candidate))


def match_speaker(
    library: list[Speaker],
    gender: Gender,
    age: Age,
    personality: list[Personality],
) -> Speaker | None:
    """Return the best-scoring speaker, or ``None`` if the library is empty."""
    if not library:
        return None
    target_tags = set(personality)

    def rank_key(sp: Speaker) -> tuple[int, int, int, str]:
        return (
            gender_distance(gender, sp.gender),
            age_distance(age, sp.age),
            -personality_overlap(target_tags, sp.personality),
            sp.speaker_id,
        )

    return min(library, key=rank_key)


class SpeakerResolutionError(Exception):
    """Raised when a character_id cannot be mapped to a Speaker —
    either it's a narrator slot with no configured voice, or a book
    character id missing from the roster."""


def resolve_speaker(
    *,
    character_id: int,
    library: list[Speaker],
    book_characters: list[Character],
) -> Speaker:
    """Map a sentence's ``character_id`` to a concrete Speaker.

    Narrator path (``character_id`` ∈ [0, 15]): direct lookup against
    the predefined narrator → speaker_id table. The speaker_id must
    exist in ``library`` (otherwise the voice library is misconfigured).

    Book character path (``character_id`` ≥ 16): look up the character
    in ``book_characters`` (which the caller supplies from
    ``characters.json``), then run attribute-based matching as before.

    Raises ``SpeakerResolutionError`` on misconfiguration / missing
    character. Callers (the TTS endpoint) should turn this into an
    HTTP 503 so the client surfaces something actionable.
    """
    if is_narrator_id(character_id):
        speaker_id = speaker_id_for_narrator(character_id)
        if speaker_id is None:
            raise SpeakerResolutionError(
                f"narrator id {character_id} is reserved but not mapped to a voice"
            )
        for sp in library:
            if sp.speaker_id == speaker_id:
                return sp
        raise SpeakerResolutionError(
            f"narrator voice {speaker_id!r} (for character_id {character_id}) "
            "not found in voice library"
        )

    character = next((c for c in book_characters if c.id == character_id), None)
    if character is None:
        raise SpeakerResolutionError(
            f"book character_id {character_id} not in characters.json"
        )
    matched = match_speaker(
        library, character.gender, character.age, character.personality,
    )
    if matched is None:
        raise SpeakerResolutionError("voice library is empty")
    return matched
