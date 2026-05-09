from __future__ import annotations

import json
from pathlib import Path

from app.core.enums import Age, Gender, Personality
from app.core.models import Speaker
from app.core.voice import (
    age_distance,
    gender_distance,
    load_voice_library,
    match_speaker,
    personality_overlap,
)


LIBRARY = [
    Speaker(speaker_id="f_adult_gentle", gender=Gender.FEMALE, age=Age.ADULT,
            personality=[Personality.GENTLE, Personality.KIND]),
    Speaker(speaker_id="m_teen_brave", gender=Gender.MALE, age=Age.TEEN,
            personality=[Personality.BRAVE, Personality.DETERMINED]),
    Speaker(speaker_id="m_elder_wise", gender=Gender.MALE, age=Age.ELDER,
            personality=[Personality.WISE, Personality.CALM]),
    Speaker(speaker_id="n_adult_calm", gender=Gender.NEUTRAL, age=Age.ADULT,
            personality=[Personality.CALM]),
]


def test_gender_distance():
    assert gender_distance(Gender.MALE, Gender.MALE) == 0
    assert gender_distance(Gender.MALE, Gender.NEUTRAL) == 1
    assert gender_distance(Gender.MALE, Gender.FEMALE) == 2


def test_age_distance():
    assert age_distance(Age.TEEN, Age.TEEN) == 0
    assert age_distance(Age.CHILD, Age.TEEN) == 1
    assert age_distance(Age.CHILD, Age.ELDER) == 4


def test_personality_overlap():
    assert personality_overlap([Personality.BRAVE, Personality.KIND],
                               [Personality.KIND, Personality.WISE]) == 1


def test_match_exact_gender_age_breaks_tie_on_personality():
    # Both candidates could be elder male; but m_elder_wise is the only elder
    # in the library, so it wins on age before personality matters.
    sp = match_speaker(LIBRARY, Gender.MALE, Age.ELDER,
                       [Personality.WISE, Personality.CALM])
    assert sp is not None and sp.speaker_id == "m_elder_wise"


def test_match_personality_breaks_tie():
    extra = Speaker(speaker_id="m_elder_cold", gender=Gender.MALE, age=Age.ELDER,
                    personality=[Personality.COLD])
    sp = match_speaker([*LIBRARY, extra], Gender.MALE, Age.ELDER,
                       [Personality.CALM])
    # Both candidates are equidistant on gender+age; m_elder_wise has CALM in
    # its tags, m_elder_cold does not — overlap tie-breaker picks wise.
    assert sp is not None and sp.speaker_id == "m_elder_wise"


def test_match_falls_back_when_no_exact_match():
    # No female teen in the library — gender/age distance kicks in.
    sp = match_speaker(LIBRARY, Gender.FEMALE, Age.TEEN, [Personality.KIND])
    assert sp is not None
    # f_adult_gentle wins: gender_dist=0, age_dist=2 vs m_teen_brave's
    # gender_dist=2, age_dist=0 — gender distance dominates.
    assert sp.speaker_id == "f_adult_gentle"


def test_match_tie_broken_by_speaker_id_for_determinism():
    a = Speaker(speaker_id="zzz", gender=Gender.MALE, age=Age.ADULT,
                personality=[])
    b = Speaker(speaker_id="aaa", gender=Gender.MALE, age=Age.ADULT,
                personality=[])
    sp = match_speaker([a, b], Gender.MALE, Age.ADULT, [])
    assert sp is not None and sp.speaker_id == "aaa"


def test_match_empty_library_returns_none():
    assert match_speaker([], Gender.MALE, Age.ADULT, []) is None


def test_load_voice_library_missing_file(tmp_path: Path):
    assert load_voice_library(tmp_path / "nope.json") == []


def test_load_voice_library_parses_json(tmp_path: Path):
    path = tmp_path / "speakers.json"
    path.write_text(json.dumps([
        {
            "speaker_id": "s1",
            "gender": "male",
            "age": "adult",
            "personality": ["calm", "wise"],
        }
    ]), encoding="utf-8")

    lib = load_voice_library(path)
    assert len(lib) == 1
    assert lib[0].speaker_id == "s1"
    assert lib[0].personality == [Personality.CALM, Personality.WISE]
