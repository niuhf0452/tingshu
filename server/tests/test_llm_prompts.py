"""Unit tests for the shared LLM prompt / parser module."""
from __future__ import annotations

import json

from app.core.enums import Age, Gender, Personality, Tone
from app.core.models import AnalyzedSentence, Character
from app.services.llm_prompts import (
    _parse_json_object,
    estimate_segment_output_tokens,
    format_known_characters_brief,
    format_known_characters_full,
    parse_character_updates,
    parse_chapter_detection,
    parse_classified_characters,
    parse_segmented_chapter,
    split_chapter_for_segmentation,
    split_long_segments,
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


# --- segment_chapter output budgeting + batching ---


class TestEstimateSegmentOutputTokens:
    def test_empty_is_zero(self):
        assert estimate_segment_output_tokens("") == 0

    def test_scales_with_length(self):
        short = estimate_segment_output_tokens("他笑了。")
        long = estimate_segment_output_tokens("他笑了。" * 100)
        assert long > short > 0

    def test_more_sentences_cost_more_than_one_long_run(self):
        # Same character count, but many enders => more NDJSON envelopes.
        many = estimate_segment_output_tokens("啊。" * 50)
        one = estimate_segment_output_tokens("啊" * 99 + "。")
        assert many > one


class TestSplitChapterForSegmentation:
    def test_empty_returns_empty(self):
        assert split_chapter_for_segmentation("", 16384) == []

    def test_short_chapter_single_batch(self):
        text = "第一行。\n\n第二行。\n\n第三行。"
        batches = split_chapter_for_segmentation(text, 16384)
        assert batches == [text]

    def test_batches_rejoin_to_original(self):
        text = "\n".join(f"这是第{i}段正文，内容足够长一些。" for i in range(2000))
        batches = split_chapter_for_segmentation(text, 16384)
        assert len(batches) > 1
        assert "\n".join(batches) == text

    def test_batch_boundaries_fall_on_lines(self):
        text = "\n".join(f"第{i}段。" for i in range(500))
        batches = split_chapter_for_segmentation(text, 16384)
        for batch in batches:
            # No batch starts or ends mid-line: every line is intact.
            assert all(ln for ln in batch.split("\n") if ln != "" or True)
        # Reassembly is exact, so no line was split.
        assert "\n".join(batches) == text

    def test_each_batch_within_budget(self):
        text = "\n".join("正文内容。" * 20 for _ in range(300))
        max_tokens = 16384
        batches = split_chapter_for_segmentation(text, max_tokens)
        budget = int(max_tokens * 0.8)
        for batch in batches[:-1]:  # last batch is whatever remains
            assert estimate_segment_output_tokens(batch) <= budget

    def test_single_oversized_line_becomes_its_own_batch(self):
        # One line alone exceeding the budget can't be split — it stands alone.
        huge = "超长的一行没有任何换行符。" * 4000
        batches = split_chapter_for_segmentation(huge, 16384)
        assert batches == [huge]


# --- split_long_segments ---


def _seg(text: str, speaker: str = "旁白", tone: Tone = Tone.NEUTRAL):
    return AnalyzedSentence(text=text, speaker=speaker, tone=tone)


class TestSplitLongSegments:
    def test_short_segment_unchanged(self):
        segs = [_seg("他抬起头，望着远方，长叹了一口气。")]
        out = split_long_segments(segs)
        assert [s.text for s in out] == [segs[0].text]

    def test_long_segment_split_at_commas(self):
        # 70+ visible chars with commas -> must be split into 2 pieces.
        text = (
            "他想起当年师父站在山门口送别时的神情，"
            "那双布满老茧的手颤抖着拍了拍他的肩，"
            "什么也没说，却让自己永生难忘，这件事一直记到今天。"
        )
        out = split_long_segments([_seg(text)])
        assert len(out) >= 2
        # Pieces concatenate back to the original — nothing lost.
        assert "".join(s.text for s in out) == text
        # Every piece stays reasonably near the 50-char cap.
        for s in out:
            assert sum(1 for c in s.text if not c.isspace()) <= 60

    def test_pieces_inherit_speaker_and_tone(self):
        text = "我一定要赢，" * 20  # long, comma-rich
        out = split_long_segments([_seg(text, speaker="萧炎", tone=Tone.ANGRY)])
        assert len(out) > 1
        assert all(s.speaker == "萧炎" and s.tone == Tone.ANGRY for s in out)

    def test_long_segment_without_delimiter_left_whole(self):
        # Over the cap but no comma/顿号 anywhere — nowhere safe to cut.
        text = "啊" * 80
        out = split_long_segments([_seg(text)])
        assert [s.text for s in out] == [text]

    def test_balanced_not_greedy(self):
        # 4 equal clauses, ~25 chars each (100 total) -> 2 balanced pieces
        # of ~50, not a 75/25 greedy fill.
        clause = "这是一段大约二十五个字长的测试用例文字内容啊，"
        out = split_long_segments([_seg(clause * 4)])
        assert len(out) == 2
        lens = [sum(1 for c in s.text if not c.isspace()) for s in out]
        assert abs(lens[0] - lens[1]) <= len(clause)

    def test_empty_input(self):
        assert split_long_segments([]) == []


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


# --- parse_classified_characters (two-bucket NDJSON: new/incidental) ---


class TestParseClassifiedCharacters:
    def test_new_only(self):
        raw = '{"k":"new","n":"药老"}\n{"k":"new","n":"美杜莎"}\n'
        result = parse_classified_characters(raw)
        assert result.new_names == ["药老", "美杜莎"]
        assert result.incidentals == []

    def test_incidental_parsed_with_full_profile(self):
        raw = (
            '{"k":"incidental","c":"妇人","g":"female","a":"adult",'
            '"p":["gentle"],"i":"路边妇人"}\n'
        )
        result = parse_classified_characters(raw)
        assert result.new_names == []
        assert len(result.incidentals) == 1
        inc = result.incidentals[0]
        assert inc.name == "妇人"
        assert inc.gender == Gender.FEMALE
        assert inc.age == Age.ADULT
        assert inc.personality == [Personality.GENTLE]
        assert inc.identity == "路边妇人"
        # id is a placeholder until service.py assigns the chapter-local
        # negative id.
        assert inc.id == 0

    def test_both_buckets_in_one_response(self):
        raw = (
            '{"k":"new","n":"药老"}\n'
            '{"k":"incidental","c":"仆人","g":"male","a":"adult",'
            '"p":["timid"],"i":"萧家仆人"}\n'
            '{"k":"incidental","c":"妇人","g":"female","a":"elder",'
            '"p":["kind"],"i":"卖菜老妇"}\n'
        )
        result = parse_classified_characters(raw)
        assert result.new_names == ["药老"]
        assert [c.name for c in result.incidentals] == ["仆人", "妇人"]

    def test_evolved_kind_is_ignored(self):
        """The 'evolved' kind was removed — such a line is now unknown
        and skipped, not merged anywhere."""
        raw = (
            '{"k":"new","n":"药老"}\n'
            '{"k":"evolved","c":"萧炎","g":"male","a":"youth",'
            '"p":["determined"],"i":"已突破斗师境"}\n'
        )
        result = parse_classified_characters(raw)
        assert result.new_names == ["药老"]
        assert result.incidentals == []

    def test_incidental_dedup_by_name(self):
        """LLM occasionally emits the same descriptor twice; keep only
        the first."""
        raw = (
            '{"k":"incidental","c":"妇人","g":"female","a":"adult",'
            '"p":["gentle"],"i":"first"}\n'
            '{"k":"incidental","c":"妇人","g":"female","a":"adult",'
            '"p":["timid"],"i":"second"}\n'
        )
        result = parse_classified_characters(raw)
        assert len(result.incidentals) == 1
        assert result.incidentals[0].identity == "first"

    def test_unknown_kind_skipped(self):
        raw = (
            '{"k":"new","n":"药老"}\n'
            '{"k":"weird","c":"???"}\n'
            '{"k":"incidental","c":"妇人","g":"female","a":"adult",'
            '"p":["gentle"],"i":"x"}\n'
        )
        result = parse_classified_characters(raw)
        assert result.new_names == ["药老"]
        assert [c.name for c in result.incidentals] == ["妇人"]

    def test_empty_returns_empty_buckets(self):
        result = parse_classified_characters("")
        assert result.new_names == []
        assert result.incidentals == []
