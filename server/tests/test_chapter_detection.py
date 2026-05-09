"""Tests for LLM-driven TXT chapter detection (§2.4.1)."""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from app.core.models import ChapterDetection
from app.core.nlp.chapters import (
    MAX_TOC_BATCHES,
    OPENING_LINE_LIMIT,
    _zh_digits_to_ascii,
    detect_and_split_chapters,
)
from app.services.llm_stub import StubLLMClient


@dataclass
class ScriptedLLM:
    """Minimal ``LLMClient`` that returns pre-scripted ``ChapterDetection`` values.

    Used to exercise the pagination loop in ``detect_and_split_chapters``.
    Only ``detect_chapters`` is implemented; other Protocol methods aren't
    called in these tests.
    """

    queue: list[ChapterDetection]
    calls: list[tuple[str, list[str] | None]] = field(default_factory=list)

    def detect_chapters(
        self,
        opening_text: str,
        *,
        known_titles: list[str] | None = None,
    ) -> ChapterDetection:
        self.calls.append((opening_text, list(known_titles) if known_titles else None))
        if self.queue:
            return self.queue.pop(0)
        return ChapterDetection()


# --- fixtures ------------------------------------------------------------


SAMPLE_TOC_BOOK = (
    "《斗破苍穹》\n"
    "作者：天蚕土豆\n\n"
    "目录\n"
    "第一章 陨落的天才\n"
    "第二章 斗之气三段\n"
    "第三章 家族测试\n\n"
    "=================\n\n"
    "第一章 陨落的天才\n"
    "萧炎站在测试石前。\n\n"
    "第二章 斗之气三段\n"
    "第二天，测试结果出来了。\n\n"
    "第三章 家族测试\n"
    "族老们围坐。\n"
)

SAMPLE_NO_TOC_BOOK = (
    "楔子\n"
    "远古时代，有一把剑。\n\n"
    "第一章 初见\n"
    "萧炎推开院门。\n\n"
    "第二章 对决\n"
    "破军现身。\n\n"
    "第三章 结局\n"
    "战斗结束。\n"
)

SAMPLE_PREFACE_NO_TOC = (
    "序章\n"
    "时间回到百年前。\n\n"
    "楔子\n"
    "另一段回忆。\n\n"
    "第一章 初见\n"
    "故事正式开始。\n\n"
    "第二章 对决\n"
    "战斗。\n\n"
    "第三章 结局\n"
    "落幕。\n"
)


def _pinned_toc_detection(titles: list[str]) -> ChapterDetection:
    return ChapterDetection(has_toc=True, chapter_titles=titles)


def _pinned_no_toc_detection(
    first_title: str,
    preface: list[str] | None = None,
    pattern: str = r"^\s*第[一二三四五六七八九十百千0-9]+章.*$",
) -> ChapterDetection:
    return ChapterDetection(
        has_toc=False,
        first_chapter_title=first_title,
        chapter_pattern=pattern,
        preface_titles=preface or [],
    )


# --- TOC path ------------------------------------------------------------


class TestTocPath:
    def test_three_chapters_from_toc(self):
        llm = StubLLMClient(chapter_detection_override=_pinned_toc_detection([
            "第一章 陨落的天才",
            "第二章 斗之气三段",
            "第三章 家族测试",
        ]))
        outcome = detect_and_split_chapters(SAMPLE_TOC_BOOK, llm, fallback_title="book")
        assert outcome.path == "toc"
        assert len(outcome.chapters) == 3
        assert outcome.chapters[0].title == "第一章 陨落的天才"
        assert "萧炎站在测试石前" in outcome.chapters[0].text
        assert "第二天" in outcome.chapters[1].text
        assert "族老们围坐" in outcome.chapters[2].text

    def test_toc_missing_title_drops_it(self):
        # LLM claims 4 titles but the 4th doesn't exist in body → 3 chapters found.
        llm = StubLLMClient(chapter_detection_override=_pinned_toc_detection([
            "第一章 陨落的天才",
            "第二章 斗之气三段",
            "第三章 家族测试",
            "第四章 不存在",
        ]))
        outcome = detect_and_split_chapters(SAMPLE_TOC_BOOK, llm, fallback_title="book")
        assert outcome.path == "toc"
        assert len(outcome.chapters) == 3


# --- pattern path --------------------------------------------------------


class TestPatternPath:
    def test_simple_no_toc(self):
        llm = StubLLMClient(
            chapter_detection_override=_pinned_no_toc_detection(
                first_title="第一章 初见",
                preface=["楔子"],
            )
        )
        outcome = detect_and_split_chapters(SAMPLE_NO_TOC_BOOK, llm, fallback_title="book")
        assert outcome.path == "pattern"
        # 楔子 + 3 chapters.
        assert len(outcome.chapters) == 4
        assert outcome.chapters[0].title == "楔子"
        assert "远古时代" in outcome.chapters[0].text
        assert outcome.chapters[1].title == "第一章 初见"
        assert outcome.chapters[2].title == "第二章 对决"

    def test_multiple_preface_titles(self):
        llm = StubLLMClient(
            chapter_detection_override=_pinned_no_toc_detection(
                first_title="第一章 初见",
                preface=["序章", "楔子"],
            )
        )
        outcome = detect_and_split_chapters(SAMPLE_PREFACE_NO_TOC, llm, fallback_title="book")
        assert outcome.path == "pattern"
        titles = [ch.title for ch in outcome.chapters]
        assert titles == ["序章", "楔子", "第一章 初见", "第二章 对决", "第三章 结局"]

    def test_no_preface(self):
        text = (
            "第一章 起点\n开头。\n\n"
            "第二章 发展\n中段。\n\n"
            "第三章 结尾\n收尾。\n"
        )
        llm = StubLLMClient(
            chapter_detection_override=_pinned_no_toc_detection(first_title="第一章 起点")
        )
        outcome = detect_and_split_chapters(text, llm, fallback_title="book")
        assert outcome.path == "pattern"
        assert [ch.title for ch in outcome.chapters] == ["第一章 起点", "第二章 发展", "第三章 结尾"]

    def test_invalid_regex_falls_back_to_single(self):
        llm = StubLLMClient(
            chapter_detection_override=ChapterDetection(
                has_toc=False,
                first_chapter_title="第一章 初见",
                chapter_pattern="[invalid(regex",
            )
        )
        outcome = detect_and_split_chapters(SAMPLE_NO_TOC_BOOK, llm, fallback_title="book")
        # No hardcoded-regex fallback path — the whole text becomes one chapter.
        assert outcome.path == "fallback_single"
        assert len(outcome.chapters) == 1
        assert outcome.chapters[0].title == "book"


# --- TOC pagination ------------------------------------------------------


class TestTocPagination:
    def test_toc_paginates_until_complete(self):
        """When ``toc_complete=False``, the detector feeds the next batch."""
        # Build a text long enough to produce 2 batches.
        padding_lines = [f"padding {i}" for i in range(OPENING_LINE_LIMIT + 10)]
        body_lines = [
            "",
            "第一章 初见", "正文一。",
            "",
            "第二章 对决", "正文二。",
            "",
            "第三章 结局", "正文三。",
        ]
        text = "\n".join(padding_lines + body_lines)

        llm = ScriptedLLM(queue=[
            ChapterDetection(
                has_toc=True,
                chapter_titles=["第一章 初见", "第二章 对决"],
                toc_complete=False,
            ),
            ChapterDetection(
                has_toc=True,
                chapter_titles=["第三章 结局"],
                toc_complete=True,
            ),
        ])

        outcome = detect_and_split_chapters(text, llm, fallback_title="book")

        assert outcome.path == "toc"
        assert len(llm.calls) == 2
        # Second call gets the accumulated titles as a hint.
        assert llm.calls[0][1] is None
        assert llm.calls[1][1] == ["第一章 初见", "第二章 对决"]
        assert outcome.llm_detection.toc_complete is True
        assert outcome.llm_detection.chapter_titles == [
            "第一章 初见", "第二章 对决", "第三章 结局",
        ]
        assert [ch.title for ch in outcome.chapters] == [
            "第一章 初见", "第二章 对决", "第三章 结局",
        ]

    def test_toc_dedup_across_batches(self):
        """Titles repeated across batches are only kept once, in first-seen order."""
        padding_lines = [f"padding {i}" for i in range(OPENING_LINE_LIMIT + 10)]
        body_lines = [
            "",
            "第一章 A", "内容 A。",
            "",
            "第二章 B", "内容 B。",
            "",
            "第三章 C", "内容 C。",
        ]
        text = "\n".join(padding_lines + body_lines)

        llm = ScriptedLLM(queue=[
            ChapterDetection(
                has_toc=True,
                chapter_titles=["第一章 A", "第二章 B"],
                toc_complete=False,
            ),
            ChapterDetection(
                has_toc=True,
                # LLM re-emits "第二章 B" by mistake — should be deduped.
                chapter_titles=["第二章 B", "第三章 C"],
                toc_complete=True,
            ),
        ])

        outcome = detect_and_split_chapters(text, llm, fallback_title="book")
        assert outcome.llm_detection.chapter_titles == [
            "第一章 A", "第二章 B", "第三章 C",
        ]

    def test_toc_hits_batch_limit_uses_accumulated(self):
        """If ``toc_complete`` never becomes True, we still try what we have."""
        # Make the text long enough for MAX_TOC_BATCHES batches.
        line_count = OPENING_LINE_LIMIT * (MAX_TOC_BATCHES + 1)
        padding = "\n".join(f"padding {i}" for i in range(line_count))
        text = padding + "\n" + "\n".join([
            "第一章 X", "正文。",
            "",
            "第二章 Y", "正文。",
        ])

        # Every batch returns the same single title with toc_complete=False.
        llm = ScriptedLLM(queue=[
            ChapterDetection(
                has_toc=True,
                chapter_titles=["第一章 X"],
                toc_complete=False,
            ) for _ in range(MAX_TOC_BATCHES)
        ])

        outcome = detect_and_split_chapters(text, llm, fallback_title="book")
        assert len(llm.calls) == MAX_TOC_BATCHES
        # Only 1 title accumulated (< MIN_CHAPTERS_FROM_LLM) → fallback.
        assert outcome.path == "fallback_single"


# --- fallback paths ------------------------------------------------------


class TestFallbacks:
    def test_empty_llm_detection_falls_back_to_single(self):
        llm = StubLLMClient(chapter_detection_override=ChapterDetection())
        outcome = detect_and_split_chapters(SAMPLE_NO_TOC_BOOK, llm, fallback_title="book")
        # No hardcoded-regex fallback — whole text becomes one chapter.
        assert outcome.path == "fallback_single"
        assert len(outcome.chapters) == 1
        assert outcome.chapters[0].title == "book"

    def test_completely_unrecognisable_text_wraps_as_single(self):
        text = "这是一本奇怪的书，没有任何章节标题。\n" * 5
        llm = StubLLMClient(chapter_detection_override=ChapterDetection())
        outcome = detect_and_split_chapters(text, llm, fallback_title="weird_book")
        assert outcome.path == "fallback_single"
        assert len(outcome.chapters) == 1
        assert outcome.chapters[0].title == "weird_book"

    def test_empty_text(self):
        outcome = detect_and_split_chapters("", StubLLMClient(), fallback_title="book")
        assert outcome.path == "fallback_single"
        assert outcome.chapters[0].title == "book"
        assert outcome.chapters[0].text == ""


# --- stub self-detect behaviour ------------------------------------------


class TestStubHeuristic:
    def test_stub_returns_detection_for_regex_pattern_input(self):
        """StubLLMClient without pinned override should detect headings via regex."""
        llm = StubLLMClient()
        detection = llm.detect_chapters(SAMPLE_NO_TOC_BOOK)
        assert detection.has_toc is False
        assert detection.first_chapter_title.startswith("第一章")
        assert detection.chapter_pattern

    def test_stub_empty_input(self):
        llm = StubLLMClient()
        detection = llm.detect_chapters("")
        assert detection.has_toc is False
        assert detection.first_chapter_title == ""


# --- number normalisation -------------------------------------------------


class TestZhDigitsToAscii:
    @pytest.mark.parametrize("zh, ascii_", [
        ("一", "1"),
        ("九", "9"),
        ("十", "10"),
        ("十一", "11"),
        ("十九", "19"),
        ("二十", "20"),
        ("三十五", "35"),
        ("九十九", "99"),
        ("第一章", "第1章"),
        ("第二十回", "第20回"),
    ])
    def test_cases(self, zh: str, ascii_: str):
        assert _zh_digits_to_ascii(zh) == ascii_
