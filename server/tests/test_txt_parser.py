from __future__ import annotations

from app.core.parsers.txt import decode_text, parse_txt, split_chapters


SAMPLE = """作者：张三

第一章 开端
这是第一章的内容。
还有第二行。

第二章 发展
第二章开始。

第三章 结尾
第三章只有一行。
"""


def test_split_chapters_basic():
    chapters = split_chapters(SAMPLE, fallback_title="Fallback")
    assert [c.title for c in chapters] == [
        "第一章 开端",
        "第二章 发展",
        "第三章 结尾",
    ]
    assert "这是第一章的内容" in chapters[0].text
    assert chapters[0].text.endswith("还有第二行。")
    assert chapters[1].text == "第二章开始。"
    assert chapters[2].text == "第三章只有一行。"


def test_split_chapters_fallback_single_chapter():
    text = "这是一本没有章节标记的书。\n只有一段文字。"
    chapters = split_chapters(text, fallback_title="无章节书")
    assert len(chapters) == 1
    assert chapters[0].title == "无章节书"
    assert chapters[0].text == text


def test_decode_text_utf8():
    raw = "第一章 测试\n内容".encode("utf-8")
    assert decode_text(raw) == "第一章 测试\n内容"


def test_decode_text_gbk():
    raw = "第一章 测试\n内容".encode("gbk")
    assert decode_text(raw) == "第一章 测试\n内容"


def test_decode_text_normalises_line_endings():
    raw = b"line1\r\nline2\rline3\n"
    assert decode_text(raw) == "line1\nline2\nline3\n"


def test_parse_txt_end_to_end():
    raw = SAMPLE.encode("utf-8")
    book = parse_txt(raw, fallback_title="测试书")
    assert book.title == "测试书"
    assert len(book.chapters) == 3
    assert book.chapters[0].title.startswith("第一章")


def test_chapter_patterns_variants():
    text = "第1回 变体\n正文1\n第 12 节 空格\n正文2\n第三篇 文言\n正文3"
    chapters = split_chapters(text, fallback_title="x")
    assert [c.title for c in chapters] == ["第1回 变体", "第 12 节 空格", "第三篇 文言"]
