"""Tests for chapter character reconciliation (post-refactor).

The reconcile module is split into two functions:
- ``merge_new_characters``: appends newly-discovered character profiles
  to the cumulative roster (already-known names are skipped).
- ``reconcile_chapter_speakers``: maps post-segmentation speaker names
  to ids using the **already-updated** roster. Unknown speakers fall
  back to the narrator (no auto-allocation).
"""
from __future__ import annotations

from app.core.enums import Age, Gender, Personality
from app.core.models import Character
from app.core.nlp.reconcile import (
    FIRST_BOOK_CHARACTER_ID,
    NARRATOR_ID,
    merge_new_characters,
    reconcile_chapter_speakers,
)


def _profile(name: str, **kw) -> Character:
    return Character(
        id=0,
        name=name,
        identity=kw.pop("identity", f"id-of-{name}"),
        gender=kw.pop("gender", Gender.MALE),
        age=kw.pop("age", Age.YOUTH),
        personality=kw.pop("personality", [Personality.CALM]),
    )


class TestMergeNewCharacters:
    def test_new_character_appended(self):
        known = [Character(id=0, name="旁白")]
        updated, new = merge_new_characters(
            known=known, new_characters=[_profile("破军", identity="反派")],
        )
        assert new == 1
        assert [c.name for c in updated] == ["旁白", "破军"]
        # First book character lands at FIRST_BOOK_CHARACTER_ID (16),
        # leaving 1..15 reserved for additional narrator slots.
        assert next(c for c in updated if c.name == "破军").id == FIRST_BOOK_CHARACTER_ID

    def test_fresh_roster_never_allocates_into_narrator_range(self):
        """Even with no known characters at all, new ids must skip the
        reserved range so a book character can't accidentally get
        routed to a narrator voice."""
        updated, _ = merge_new_characters(
            known=[], new_characters=[_profile("a"), _profile("b"), _profile("c")],
        )
        ids = [c.id for c in updated]
        assert all(i >= FIRST_BOOK_CHARACTER_ID for i in ids)
        assert ids == sorted(set(ids))  # contiguous + unique

    def test_known_character_profile_not_overwritten(self):
        """A character's profile is fixed once established — a candidate
        with an already-known name is skipped, not merged."""
        known = [
            Character(id=0, name="旁白"),
            Character(
                id=16, name="萧炎", identity="少年弟子，主角",
                gender=Gender.MALE, age=Age.TEEN,
                personality=[Personality.BRAVE],
            ),
        ]
        candidate = Character(
            id=0, name="萧炎", identity="青年高手，主角",
            gender=Gender.MALE, age=Age.YOUTH,
            personality=[Personality.BRAVE, Personality.WISE],
        )
        updated, new = merge_new_characters(
            known=known, new_characters=[candidate],
        )
        assert new == 0
        xy = next(c for c in updated if c.name == "萧炎")
        assert xy.id == 16  # untouched
        assert xy.age == Age.TEEN  # original profile preserved
        assert xy.identity == "少年弟子，主角"

    def test_narrator_candidate_ignored(self):
        known = [Character(id=0, name="旁白")]
        bogus = Character(id=0, name="旁白", identity="should not be applied")
        updated, new = merge_new_characters(
            known=known, new_characters=[bogus],
        )
        assert new == 0
        assert updated[0].identity == ""

    def test_new_ids_continue_from_max_above_reserved(self):
        known = [
            Character(id=0, name="旁白"),
            Character(id=42, name="萧炎"),  # arbitrary high existing id
        ]
        updated, _ = merge_new_characters(
            known=known, new_characters=[_profile("药老")],
        )
        # Continue from max(known)+1 since that's already past the
        # narrator-reserved range.
        assert next(c for c in updated if c.name == "药老").id == 43

    def test_empty_input_is_noop(self):
        known = [Character(id=0, name="旁白"), Character(id=1, name="萧炎")]
        updated, new = merge_new_characters(known=known, new_characters=[])
        assert new == 0
        assert updated == known


class TestReconcileChapterSpeakers:
    def test_narrator_resolves_to_zero(self):
        known = [Character(id=0, name="旁白")]
        mapping, chapter_chars = reconcile_chapter_speakers(
            known=known, speakers=["旁白"],
        )
        assert mapping["旁白"] == NARRATOR_ID
        # narrator never appears in chapter_characters
        assert chapter_chars == []

    def test_empty_string_resolves_to_narrator(self):
        known = [Character(id=0, name="旁白")]
        mapping, _ = reconcile_chapter_speakers(known=known, speakers=[""])
        assert mapping[""] == NARRATOR_ID

    def test_known_speaker_resolves_to_existing_id(self):
        known = [
            Character(id=0, name="旁白"),
            Character(id=1, name="萧炎", identity="主角"),
        ]
        mapping, chapter_chars = reconcile_chapter_speakers(
            known=known, speakers=["萧炎"],
        )
        assert mapping["萧炎"] == 1
        assert [c.name for c in chapter_chars] == ["萧炎"]

    def test_unknown_speaker_falls_back_to_narrator(self):
        """Phase A is supposed to register every character before
        segmentation runs. If an unknown name still appears, the safe
        fallback is to map it to the narrator (line stays audible in
        narrator voice rather than fabricating a default character)."""
        known = [Character(id=0, name="旁白")]
        mapping, chapter_chars = reconcile_chapter_speakers(
            known=known, speakers=["路人甲"],
        )
        assert mapping["路人甲"] == NARRATOR_ID
        # unknown speaker doesn't show up in chapter_characters
        assert chapter_chars == []

    def test_multiple_speakers_deduped_in_chapter_snapshot(self):
        known = [
            Character(id=0, name="旁白"),
            Character(id=1, name="萧炎"),
            Character(id=2, name="药老"),
        ]
        speakers = ["萧炎", "药老", "萧炎", "旁白", "药老"]
        mapping, chapter_chars = reconcile_chapter_speakers(
            known=known, speakers=speakers,
        )
        assert mapping == {"萧炎": 1, "药老": 2, "旁白": NARRATOR_ID}
        # snapshot dedups, narrator excluded
        assert [c.name for c in chapter_chars] == ["萧炎", "药老"]

    def test_speaker_whitespace_trimmed(self):
        known = [
            Character(id=0, name="旁白"),
            Character(id=1, name="萧炎"),
        ]
        mapping, _ = reconcile_chapter_speakers(
            known=known, speakers=["  萧炎  "],
        )
        assert mapping["萧炎"] == 1

    def test_no_speakers_is_noop(self):
        known = [Character(id=0, name="旁白")]
        mapping, chapter_chars = reconcile_chapter_speakers(
            known=known, speakers=[],
        )
        assert mapping == {}
        assert chapter_chars == []
