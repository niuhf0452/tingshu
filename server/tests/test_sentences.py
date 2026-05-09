"""locate_sentences — maps LLM-produced sentence text + reconciled speaker
names to ``(line, col)`` ranges.

Under Plan C ``AnalyzedSentence`` carries ``speaker: str``. The caller is
expected to run reconciliation first and pass the resulting
``speaker_to_id`` map down here.
"""
from __future__ import annotations

from app.core.enums import Tone
from app.core.models import AnalyzedSentence
from app.core.nlp.sentences import locate_sentences


def _analyzed(*pairs: tuple[str, str] | tuple[str, str, Tone]) -> list[AnalyzedSentence]:
    out = []
    for p in pairs:
        if len(p) == 2:
            text, speaker = p
            tone = Tone.NEUTRAL
        else:
            text, speaker, tone = p
        out.append(AnalyzedSentence(text=text, speaker=speaker, tone=tone))
    return out


# Standard speaker_to_id map used by the positive cases below.
_MAP = {"旁白": 0, "萧炎": 1, "药老": 2}


def test_single_line_positions():
    text = "萧炎紧握拳头。药老微笑。"
    analyzed = _analyzed(("萧炎紧握拳头。", "萧炎"), ("药老微笑。", "药老"))
    out = locate_sentences(text, analyzed, _MAP)

    assert len(out) == 2
    assert out[0].start_line == 1 and out[0].start_col == 0
    assert out[0].end_line == 1 and out[0].end_col == 7
    assert out[0].character_id == 1

    assert out[1].start_col == 7
    assert out[1].end_col == 12
    assert out[1].character_id == 2


def test_multi_line():
    text = "第一章 开端\n萧炎紧握拳头。\n药老微笑。\n"
    analyzed = _analyzed(("萧炎紧握拳头。", "萧炎"), ("药老微笑。", "药老"))
    out = locate_sentences(text, analyzed, _MAP)

    # line 1 = "第一章 开端"; sentences start on line 2 and 3.
    assert out[0].start_line == 2 and out[0].start_col == 0
    assert out[0].end_line == 2 and out[0].end_col == 7
    assert out[1].start_line == 3 and out[1].start_col == 0
    assert out[1].end_line == 3 and out[1].end_col == 5


def test_cursor_advances_on_repeated_sentence():
    text = "他笑了。他又笑了。他还笑了。"
    analyzed = _analyzed(
        ("他笑了。", "旁白"),
        ("他又笑了。", "旁白"),
        ("他还笑了。", "旁白"),
    )
    out = locate_sentences(text, analyzed, _MAP)
    cols = [(s.start_col, s.end_col) for s in out]
    assert cols == [(0, 4), (4, 9), (9, 14)]


def test_missing_sentence_is_dropped():
    text = "萧炎在场。"
    analyzed = _analyzed(("萧炎在场。", "萧炎"), ("这句话不存在。", "旁白"))
    out = locate_sentences(text, analyzed, _MAP)
    assert len(out) == 1
    assert out[0].character_id == 1


def test_whitespace_tolerant_match():
    text = "萧炎 说话。"
    analyzed = _analyzed(("萧炎  说话。", "萧炎"))
    out = locate_sentences(text, analyzed, _MAP)
    assert len(out) == 1
    assert out[0].start_col == 0
    assert out[0].end_col == len(text)


def test_unknown_speaker_defaults_to_narrator():
    text = "你好。"
    analyzed = _analyzed(("你好。", "未知陌生人", Tone.HAPPY))
    # Speaker not in the map → character_id 0 (narrator).
    out = locate_sentences(text, analyzed, _MAP)
    assert out[0].character_id == 0
    assert out[0].tone == Tone.HAPPY


def test_speaker_to_id_optional():
    """Without a mapping, everything falls back to narrator id=0."""
    text = "你好。"
    analyzed = _analyzed(("你好。", "萧炎"))
    out = locate_sentences(text, analyzed)
    assert out[0].character_id == 0


def test_attributes_preserved():
    text = "你好。"
    analyzed = _analyzed(("你好。", "药老", Tone.HAPPY))
    out = locate_sentences(text, analyzed, _MAP)
    assert out[0].character_id == 2
    assert out[0].tone == Tone.HAPPY


def test_empty_inputs():
    assert locate_sentences("", []) == []
    assert locate_sentences("some text", []) == []
    assert locate_sentences("", _analyzed(("x", "旁白"))) == []
