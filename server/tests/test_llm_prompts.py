"""Unit tests for the shared LLM prompt / parser module."""
from __future__ import annotations

import json

from app.core.enums import Age, Gender, Personality, Tone
from app.core.models import Character
from app.services.llm_prompts import (
    _parse_json_object,
    format_known_characters_brief,
    format_known_characters_full,
    parse_character_updates,
    parse_chapter_detection,
    parse_segmented_chapter,
)


# --- _parse_json_object ---


class TestParseJsonObject:
    def test_plain_json(self):
        assert _parse_json_object('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert _parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_fenced_without_lang(self):
        assert _parse_json_object('```\n{"a": 1}\n```') == {"a": 1}

    def test_with_surrounding_prose(self):
        assert _parse_json_object('here it is: {"a": 1}. done.') == {"a": 1}

    def test_invalid_returns_none(self):
        assert _parse_json_object("not json at all") is None

    def test_empty_returns_none(self):
        assert _parse_json_object("") is None

    def test_malformed_brace(self):
        assert _parse_json_object('{"a": }') is None


# --- parse_segmented_chapter ---


class TestParseSegmentedChapter:
    def test_two_segments(self):
        raw = (
            '{"t":"句子一","s":"旁白","o":"neutral"}\n'
            '{"t":"句子二","s":"萧炎","o":"angry"}\n'
        )
        result = parse_segmented_chapter(raw)
        assert len(result) == 2
        assert result[0].text == "句子一"
        assert result[1].speaker == "萧炎"
        assert result[1].tone == Tone.ANGRY

    def test_long_keys_accepted(self):
        raw = '{"text":"x","speaker":"旁白","tone":"happy"}\n'
        result = parse_segmented_chapter(raw)
        assert result[0].tone == Tone.HAPPY

    def test_blank_lines_and_fences_ignored(self):
        raw = (
            '```ndjson\n'
            '{"t":"a","s":"旁白","o":"neutral"}\n'
            '\n'
            '{"t":"b","s":"旁白","o":"neutral"}\n'
            '```\n'
        )
        result = parse_segmented_chapter(raw)
        assert [s.text for s in result] == ["a", "b"]

    def test_unknown_tone_becomes_neutral(self):
        raw = '{"t":"x","s":"旁白","o":"ecstatic"}\n'
        assert parse_segmented_chapter(raw)[0].tone == Tone.NEUTRAL

    def test_empty_text_dropped(self):
        raw = (
            '{"t":"","s":"旁白","o":"neutral"}\n'
            '{"t":"keep","s":"旁白","o":"neutral"}\n'
        )
        result = parse_segmented_chapter(raw)
        assert [s.text for s in result] == ["keep"]

    def test_bad_line_skipped_others_kept(self):
        raw = (
            '{"t":"first","s":"旁白","o":"neutral"}\n'
            '{this is broken json\n'
            '{"t":"third","s":"旁白","o":"neutral"}\n'
        )
        result = parse_segmented_chapter(raw)
        assert [s.text for s in result] == ["first", "third"]

    def test_garbage_returns_empty(self):
        assert parse_segmented_chapter("garbage") == []


# --- parse_character_updates ---


class TestParseCharacterUpdates:
    def test_one_profile(self):
        raw = '{"c":"萧炎","g":"male","a":"teen","p":["brave","determined"],"i":"少年弟子，主角"}\n'
        result = parse_character_updates(raw)
        assert len(result) == 1
        upd = result[0]
        assert upd.name == "萧炎"
        assert upd.age == Age.TEEN
        assert upd.gender == Gender.MALE
        assert upd.personality == [Personality.BRAVE, Personality.DETERMINED]
        assert upd.identity == "少年弟子，主角"
        assert upd.id == 0  # placeholder; reconcile assigns the real id

    def test_long_keys_accepted(self):
        raw = '{"name":"破军","gender":"male","age":"adult","personality":["fierce"],"identity":"反派"}\n'
        result = parse_character_updates(raw)
        assert result[0].name == "破军"

    def test_blank_lines_and_fences_ignored(self):
        raw = (
            '```ndjson\n'
            '\n'
            '{"c":"x","g":"male","a":"adult","p":["calm"],"i":"y"}\n'
            '```\n'
        )
        result = parse_character_updates(raw)
        assert result[0].name == "x"

    def test_invalid_gender_becomes_neutral(self):
        raw = '{"c":"破军","g":"alien","a":"adult","p":["calm"],"i":"x"}\n'
        result = parse_character_updates(raw)
        # bogus gender → fall back to NEUTRAL (don't drop the update)
        assert result[0].gender == Gender.NEUTRAL

    def test_empty_input_returns_empty(self):
        assert parse_character_updates("") == []
        assert parse_character_updates("\n\n\n") == []

    def test_garbage_returns_empty(self):
        assert parse_character_updates("garbage") == []

    def test_without_personality_defaults_to_calm(self):
        raw = '{"c":"y","g":"male","a":"adult","p":[],"i":"z"}\n'
        result = parse_character_updates(raw)
        assert result[0].personality == [Personality.CALM]

    def test_drops_invalid_personality_tags(self):
        raw = '{"c":"y","g":"male","a":"adult","p":["calm","BOGUS","wise"],"i":"z"}\n'
        result = parse_character_updates(raw)
        assert result[0].personality == [Personality.CALM, Personality.WISE]

    def test_identity_capped_at_80(self):
        raw = f'{{"c":"y","g":"male","a":"adult","p":["calm"],"i":"{"x"*200}"}}\n'
        result = parse_character_updates(raw)
        assert len(result[0].identity) == 80


# --- parse_chapter_detection ---


class TestParseChapterDetection:
    def test_toc(self):
        raw = json.dumps({
            "has_toc": True,
            "chapter_titles": ["第一章", "第二章"],
        })
        d = parse_chapter_detection(raw)
        assert d is not None
        assert d.has_toc is True
        assert d.chapter_titles == ["第一章", "第二章"]
        assert d.toc_complete is True

    def test_toc_partial(self):
        raw = json.dumps({
            "has_toc": True,
            "chapter_titles": ["第一章"],
            "toc_complete": False,
        })
        d = parse_chapter_detection(raw)
        assert d.toc_complete is False

    def test_no_toc(self):
        raw = json.dumps({
            "has_toc": False,
            "preface_titles": ["楔子"],
            "first_chapter_title": "第一章",
            "chapter_pattern": r"^第\d+章",
        })
        d = parse_chapter_detection(raw)
        assert d.has_toc is False
        assert d.preface_titles == ["楔子"]

    def test_titles_cleaned(self):
        raw = json.dumps({
            "has_toc": True,
            "chapter_titles": ["  第一章  ", "", 42, "第二章"],
        })
        d = parse_chapter_detection(raw)
        assert d.chapter_titles == ["第一章", "第二章"]

    def test_garbage(self):
        assert parse_chapter_detection("not json") is None


# --- format_known_characters_full ---


class TestFormatKnownCharactersFull:
    def test_empty(self):
        # Empty roster → "only narrator" message (no real characters
        # registered yet means everything in the chapter is new).
        assert "仅有旁白" in format_known_characters_full([])

    def test_only_narrator(self):
        out = format_known_characters_full([Character(id=0, name="旁白")])
        assert "仅有旁白" in out

    def test_full_profile_rendered(self):
        chars = [
            Character(id=0, name="旁白"),
            Character(
                id=1, name="萧炎", identity="少年弟子，主角",
                gender=Gender.MALE, age=Age.TEEN,
                personality=[Personality.BRAVE, Personality.DETERMINED],
            ),
        ]
        out = format_known_characters_full(chars)
        assert "萧炎" in out
        assert "gender=male" in out
        assert "age=teen" in out
        assert "brave,determined" in out
        assert "少年弟子，主角" in out
        assert "旁白" not in out  # narrator implicit


# --- format_known_characters_brief ---


class TestFormatKnownCharactersBrief:
    def test_empty(self):
        out = format_known_characters_brief([])
        assert "新角色" in out

    def test_only_narrator(self):
        out = format_known_characters_brief([Character(id=0, name="旁白")])
        assert "新角色" in out

    def test_brief_omits_attributes(self):
        chars = [
            Character(
                id=1, name="萧炎", identity="少年弟子，主角",
                gender=Gender.MALE, age=Age.TEEN,
                personality=[Personality.BRAVE],
            ),
        ]
        out = format_known_characters_brief(chars)
        assert "萧炎" in out
        assert "少年弟子，主角" in out
        # Brief form intentionally omits gender/age/personality so the
        # segmentation prompt stays focused on speaker assignment.
        assert "gender=" not in out
        assert "age=" not in out
        assert "personality=" not in out
