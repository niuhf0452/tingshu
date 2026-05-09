"""Tests for the EPUB parser.

Synthesises minimal EPUB archives in-memory so the tests don't depend on
any fixture files. Each test builds just enough structure (container.xml,
OPF package doc, TOC, spine HTML files) to exercise one behaviour.
"""
from __future__ import annotations

import io
import zipfile

import pytest

from app.core.parsers.epub import EpubParseError, parse_epub


# --- tiny EPUB builder ---------------------------------------------------


def _build_epub(files: dict[str, str]) -> bytes:
    """Zip up ``{archive_path: text_content}`` into EPUB bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # The spec wants mimetype first and uncompressed — fine to emit
        # last for our parser, which doesn't depend on this.
        for path, content in files.items():
            zf.writestr(path, content)
    return buf.getvalue()


_CONTAINER_XML = """<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""


def _opf_epub2(
    title: str = "示例书",
    author: str = "示例作者",
    chapters: list[tuple[str, str]] | None = None,
    include_ncx: bool = True,
) -> str:
    chapters = chapters or [("ch1", "ch1.xhtml"), ("ch2", "ch2.xhtml")]
    manifest_items = "\n    ".join(
        f'<item id="{cid}" href="{href}" media-type="application/xhtml+xml"/>'
        for cid, href in chapters
    )
    spine_items = "\n    ".join(
        f'<itemref idref="{cid}"/>' for cid, _ in chapters
    )
    ncx_item = (
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
        if include_ncx else ""
    )
    spine_toc = ' toc="ncx"' if include_ncx else ""
    return f"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>{title}</dc:title>
    <dc:creator>{author}</dc:creator>
    <dc:identifier id="bookid">urn:uuid:abc</dc:identifier>
    <dc:language>zh</dc:language>
  </metadata>
  <manifest>
    {manifest_items}
    {ncx_item}
  </manifest>
  <spine{spine_toc}>
    {spine_items}
  </spine>
</package>
"""


def _ncx(chapters: list[tuple[str, str]]) -> str:
    points = "\n  ".join(
        f'<navPoint id="np{i}" playOrder="{i+1}">'
        f'<navLabel><text>{title}</text></navLabel>'
        f'<content src="{href}"/>'
        f'</navPoint>'
        for i, (title, href) in enumerate(chapters)
    )
    return f"""<?xml version="1.0"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head><meta name="dtb:uid" content="abc"/></head>
  <docTitle><text>示例</text></docTitle>
  <navMap>
  {points}
  </navMap>
</ncx>
"""


def _nav_xhtml(chapters: list[tuple[str, str]]) -> str:
    lis = "\n      ".join(
        f'<li><a href="{href}">{title}</a></li>' for title, href in chapters
    )
    return f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>TOC</title></head>
<body>
  <nav epub:type="toc">
    <ol>
      {lis}
    </ol>
  </nav>
</body>
</html>
"""


def _chapter_html(body: str) -> str:
    return f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>ch</title></head>
<body>{body}</body>
</html>
"""


# --- tests: container + metadata ----------------------------------------


class TestContainerAndMetadata:
    def test_missing_container_errors(self):
        data = _build_epub({"OEBPS/content.opf": _opf_epub2()})
        with pytest.raises(EpubParseError, match="container.xml"):
            parse_epub(data, fallback_title="fb")

    def test_not_a_zip_errors(self):
        with pytest.raises(EpubParseError, match="ZIP"):
            parse_epub(b"not a zip file", fallback_title="fb")

    def test_missing_opf_errors(self):
        data = _build_epub({"META-INF/container.xml": _CONTAINER_XML})
        with pytest.raises(EpubParseError):
            parse_epub(data, fallback_title="fb")

    def test_metadata_title_and_author(self):
        chs = [("ch1", "ch1.xhtml"), ("ch2", "ch2.xhtml")]
        data = _build_epub({
            "META-INF/container.xml": _CONTAINER_XML,
            "OEBPS/content.opf": _opf_epub2(
                title="三体", author="刘慈欣", chapters=chs,
            ),
            "OEBPS/toc.ncx": _ncx([("第一章 章名", "ch1.xhtml"), ("第二章", "ch2.xhtml")]),
            "OEBPS/ch1.xhtml": _chapter_html("<p>正文 1</p>"),
            "OEBPS/ch2.xhtml": _chapter_html("<p>正文 2</p>"),
        })
        parsed = parse_epub(data, fallback_title="fallback")
        assert parsed.title == "三体"
        assert parsed.author == "刘慈欣"


# --- tests: EPUB 2 (ncx) ------------------------------------------------


class TestEpub2Ncx:
    def test_chapters_from_ncx(self):
        data = _build_epub({
            "META-INF/container.xml": _CONTAINER_XML,
            "OEBPS/content.opf": _opf_epub2(),
            "OEBPS/toc.ncx": _ncx([
                ("楔子", "ch1.xhtml"),
                ("第一章 初见", "ch2.xhtml"),
            ]),
            "OEBPS/ch1.xhtml": _chapter_html("<p>很久很久以前……</p>"),
            "OEBPS/ch2.xhtml": _chapter_html("<p>萧炎推开了门。</p><p>院内空无一人。</p>"),
        })
        parsed = parse_epub(data, fallback_title="fb")
        assert [c.title for c in parsed.chapters] == ["楔子", "第一章 初见"]
        assert "很久很久以前" in parsed.chapters[0].text
        assert "萧炎推开了门" in parsed.chapters[1].text
        assert "院内空无一人" in parsed.chapters[1].text

    def test_spine_order_preserved(self):
        chs = [("a", "a.xhtml"), ("b", "b.xhtml"), ("c", "c.xhtml")]
        data = _build_epub({
            "META-INF/container.xml": _CONTAINER_XML,
            "OEBPS/content.opf": _opf_epub2(chapters=chs),
            "OEBPS/toc.ncx": _ncx([
                ("First", "a.xhtml"),
                ("Second", "b.xhtml"),
                ("Third", "c.xhtml"),
            ]),
            "OEBPS/a.xhtml": _chapter_html("<p>A</p>"),
            "OEBPS/b.xhtml": _chapter_html("<p>B</p>"),
            "OEBPS/c.xhtml": _chapter_html("<p>C</p>"),
        })
        parsed = parse_epub(data, fallback_title="fb")
        assert [c.text for c in parsed.chapters] == ["A", "B", "C"]

    def test_ncx_missing_uses_fallback_titles(self):
        data = _build_epub({
            "META-INF/container.xml": _CONTAINER_XML,
            "OEBPS/content.opf": _opf_epub2(include_ncx=False),
            "OEBPS/ch1.xhtml": _chapter_html("<h1>章节一</h1><p>内容一。</p>"),
            "OEBPS/ch2.xhtml": _chapter_html("<p>仅有正文的章节。</p>"),
        })
        parsed = parse_epub(data, fallback_title="fb")
        # Without TOC, we fall back to the first heading line or a default.
        assert len(parsed.chapters) == 2
        assert parsed.chapters[0].title == "章节一"
        # No heading → default title.
        assert parsed.chapters[1].title.startswith("第") or parsed.chapters[1].title == "仅有正文的章节。"


# --- tests: EPUB 3 (nav.xhtml) ------------------------------------------


class TestEpub3Nav:
    def test_nav_toc_properties(self):
        chs = [("ch1", "ch1.xhtml"), ("ch2", "ch2.xhtml"), ("navdoc", "nav.xhtml")]
        opf = _opf_epub2(chapters=chs, include_ncx=False).replace(
            '<item id="navdoc" href="nav.xhtml" media-type="application/xhtml+xml"/>',
            '<item id="navdoc" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
        )
        data = _build_epub({
            "META-INF/container.xml": _CONTAINER_XML,
            "OEBPS/content.opf": opf,
            "OEBPS/nav.xhtml": _nav_xhtml([
                ("序章", "ch1.xhtml"),
                ("第一章", "ch2.xhtml"),
            ]),
            "OEBPS/ch1.xhtml": _chapter_html("<p>序章内容</p>"),
            "OEBPS/ch2.xhtml": _chapter_html("<p>第一章内容</p>"),
        })
        parsed = parse_epub(data, fallback_title="fb")
        # nav.xhtml is part of the spine; but TOC still supplies titles for ch1/ch2.
        # Only non-empty chapters are kept — nav document's body has <nav> toc but
        # outside that we only have "序章 / 第一章" text, which the stripper keeps.
        real_chapters = [c for c in parsed.chapters if c.title in ("序章", "第一章")]
        assert [c.title for c in real_chapters] == ["序章", "第一章"]
        assert "序章内容" in real_chapters[0].text


# --- tests: HTML extraction ---------------------------------------------


class TestHtmlExtraction:
    def test_strips_tags_and_preserves_paragraphs(self):
        html = _chapter_html(
            "<p>第一段。</p>"
            "<p>第二段<em>强调</em>。</p>"
            "<script>alert('x')</script>"
            "<style>body{}</style>"
            "<p>第三段。</p>"
        )
        data = _build_epub({
            "META-INF/container.xml": _CONTAINER_XML,
            "OEBPS/content.opf": _opf_epub2(chapters=[("ch1", "ch1.xhtml")]),
            "OEBPS/toc.ncx": _ncx([("唯一章", "ch1.xhtml")]),
            "OEBPS/ch1.xhtml": html,
        })
        parsed = parse_epub(data, fallback_title="fb")
        text = parsed.chapters[0].text
        assert "第一段。" in text
        assert "第二段强调。" in text
        assert "第三段。" in text
        assert "alert" not in text  # script content stripped
        assert "body{}" not in text  # style content stripped
        # Paragraph breaks should still be present as newlines.
        assert "第一段。" in text.split("\n")[0]

    def test_entities_decoded(self):
        data = _build_epub({
            "META-INF/container.xml": _CONTAINER_XML,
            "OEBPS/content.opf": _opf_epub2(chapters=[("ch1", "ch1.xhtml")]),
            "OEBPS/toc.ncx": _ncx([("ch1", "ch1.xhtml")]),
            "OEBPS/ch1.xhtml": _chapter_html("<p>A &amp; B &#x4e2d;文</p>"),
        })
        parsed = parse_epub(data, fallback_title="fb")
        assert "A & B 中文" in parsed.chapters[0].text


# --- tests: path resolution ---------------------------------------------


class TestPathResolution:
    def test_opf_in_subdirectory_resolves_relative_hrefs(self):
        """When the OPF sits in a subdir, manifest hrefs are relative to it."""
        data = _build_epub({
            "META-INF/container.xml": _CONTAINER_XML,
            "OEBPS/content.opf": _opf_epub2(),
            "OEBPS/toc.ncx": _ncx([("First", "ch1.xhtml")]),
            "OEBPS/ch1.xhtml": _chapter_html("<p>在 OEBPS 子目录下</p>"),
            "OEBPS/ch2.xhtml": _chapter_html("<p>第二章</p>"),
        })
        parsed = parse_epub(data, fallback_title="fb")
        assert "在 OEBPS 子目录下" in parsed.chapters[0].text

    def test_parent_relative_href(self):
        """OPF references ../text/ch.xhtml style paths are resolved."""
        chs = [("ch1", "../text/ch1.xhtml"), ("ch2", "../text/ch2.xhtml")]
        data = _build_epub({
            "META-INF/container.xml": _CONTAINER_XML,
            "OEBPS/package/content.opf": _opf_epub2(chapters=chs).replace(
                "OEBPS/content.opf", "OEBPS/package/content.opf"
            ),
            "OEBPS/package/toc.ncx": _ncx([
                ("A", "../text/ch1.xhtml"),
                ("B", "../text/ch2.xhtml"),
            ]),
            "OEBPS/text/ch1.xhtml": _chapter_html("<p>Alpha</p>"),
            "OEBPS/text/ch2.xhtml": _chapter_html("<p>Beta</p>"),
        })
        # Override container.xml to point at the subdirectory opf.
        data = _build_epub({
            "META-INF/container.xml": _CONTAINER_XML.replace(
                "OEBPS/content.opf", "OEBPS/package/content.opf"
            ),
            "OEBPS/package/content.opf": _opf_epub2(chapters=chs),
            "OEBPS/package/toc.ncx": _ncx([
                ("A", "../text/ch1.xhtml"),
                ("B", "../text/ch2.xhtml"),
            ]),
            "OEBPS/text/ch1.xhtml": _chapter_html("<p>Alpha</p>"),
            "OEBPS/text/ch2.xhtml": _chapter_html("<p>Beta</p>"),
        })
        parsed = parse_epub(data, fallback_title="fb")
        assert [c.title for c in parsed.chapters] == ["A", "B"]
        assert "Alpha" in parsed.chapters[0].text
        assert "Beta" in parsed.chapters[1].text


# --- tests: edge cases --------------------------------------------------


class TestEdgeCases:
    def test_empty_chapter_skipped(self):
        data = _build_epub({
            "META-INF/container.xml": _CONTAINER_XML,
            "OEBPS/content.opf": _opf_epub2(),
            "OEBPS/toc.ncx": _ncx([
                ("A", "ch1.xhtml"),
                ("B", "ch2.xhtml"),
            ]),
            "OEBPS/ch1.xhtml": _chapter_html(""),  # empty body
            "OEBPS/ch2.xhtml": _chapter_html("<p>实际内容</p>"),
        })
        parsed = parse_epub(data, fallback_title="fb")
        assert len(parsed.chapters) == 1
        assert parsed.chapters[0].title == "B"

    def test_title_fallback_when_only_empty_chapters(self):
        """If every chapter is empty, we still get one fallback chapter."""
        data = _build_epub({
            "META-INF/container.xml": _CONTAINER_XML,
            "OEBPS/content.opf": _opf_epub2(chapters=[("ch1", "ch1.xhtml")]),
            "OEBPS/toc.ncx": _ncx([("empty", "ch1.xhtml")]),
            "OEBPS/ch1.xhtml": _chapter_html(""),
        })
        parsed = parse_epub(data, fallback_title="fallback_book")
        assert len(parsed.chapters) == 1
        assert parsed.chapters[0].title == "fallback_book"

    def test_linear_no_spine_item_skipped(self):
        """Spine items with linear='no' (e.g. footnotes) are dropped."""
        chs = [("ch1", "ch1.xhtml"), ("ch2", "ch2.xhtml")]
        opf = _opf_epub2(chapters=chs).replace(
            '<itemref idref="ch2"/>', '<itemref idref="ch2" linear="no"/>',
        )
        data = _build_epub({
            "META-INF/container.xml": _CONTAINER_XML,
            "OEBPS/content.opf": opf,
            "OEBPS/toc.ncx": _ncx([("A", "ch1.xhtml")]),
            "OEBPS/ch1.xhtml": _chapter_html("<p>主内容</p>"),
            "OEBPS/ch2.xhtml": _chapter_html("<p>脚注内容</p>"),
        })
        parsed = parse_epub(data, fallback_title="fb")
        assert len(parsed.chapters) == 1
        assert "脚注内容" not in parsed.chapters[0].text


class TestInSpineTocSkipped:
    """A book may ship a hand-rolled "目录" XHTML in the spine (separate
    from the EPUB 3 nav doc / EPUB 2 NCX). Those are clickable indices,
    not real chapters; after our HTML→text strip the links are gone, so
    keeping them produces a meaningless first "chapter" of plain title
    text. These tests pin the skip behaviour."""

    def test_skip_by_chinese_title(self):
        """Spine item titled `目录` is skipped regardless of body."""
        # NCX gives the in-spine TOC item a label of "目录". The next two
        # spine items are real chapters.
        chs = [
            ("toc", "目录.xhtml"),
            ("ch1", "ch1.xhtml"),
            ("ch2", "ch2.xhtml"),
        ]
        ncx_titles = [
            ("目录", "目录.xhtml"),
            ("第一章 序章", "ch1.xhtml"),
            ("第二章 战斗", "ch2.xhtml"),
        ]
        data = _build_epub({
            "META-INF/container.xml": _CONTAINER_XML,
            "OEBPS/content.opf": _opf_epub2(chapters=chs),
            "OEBPS/toc.ncx": _ncx(ncx_titles),
            "OEBPS/目录.xhtml": _chapter_html(
                "<h1>目录</h1>"
                "<p>第一章 序章</p><p>第二章 战斗</p>"
            ),
            "OEBPS/ch1.xhtml": _chapter_html("<p>第一章正文</p>"),
            "OEBPS/ch2.xhtml": _chapter_html("<p>第二章正文</p>"),
        })
        parsed = parse_epub(data, fallback_title="fb")
        titles = [c.title for c in parsed.chapters]
        assert "目录" not in titles
        assert titles == ["第一章 序章", "第二章 战斗"]

    def test_skip_by_english_title(self):
        chs = [
            ("toc", "toc.xhtml"),
            ("ch1", "ch1.xhtml"),
            ("ch2", "ch2.xhtml"),
        ]
        ncx_titles = [
            ("Contents", "toc.xhtml"),
            ("Chapter 1", "ch1.xhtml"),
            ("Chapter 2", "ch2.xhtml"),
        ]
        data = _build_epub({
            "META-INF/container.xml": _CONTAINER_XML,
            "OEBPS/content.opf": _opf_epub2(chapters=chs),
            "OEBPS/toc.ncx": _ncx(ncx_titles),
            "OEBPS/toc.xhtml": _chapter_html(
                "<h1>Contents</h1>"
                "<p>Chapter 1</p><p>Chapter 2</p>"
            ),
            "OEBPS/ch1.xhtml": _chapter_html("<p>Body of chapter 1</p>"),
            "OEBPS/ch2.xhtml": _chapter_html("<p>Body of chapter 2</p>"),
        })
        parsed = parse_epub(data, fallback_title="fb")
        titles = [c.title for c in parsed.chapters]
        assert titles == ["Chapter 1", "Chapter 2"]

    def test_skip_by_content_match(self):
        """Even when the in-spine TOC has a generic title, we should
        still recognise it via content overlap with the parsed TOC."""
        chs = [
            ("toc", "index.xhtml"),
            ("ch1", "ch1.xhtml"),
            ("ch2", "ch2.xhtml"),
            ("ch3", "ch3.xhtml"),
            ("ch4", "ch4.xhtml"),
        ]
        ncx_titles = [
            # Generic label for the in-spine TOC item.
            ("第一篇", "index.xhtml"),
            ("第一章 启程", "ch1.xhtml"),
            ("第二章 风波", "ch2.xhtml"),
            ("第三章 决战", "ch3.xhtml"),
            ("第四章 归途", "ch4.xhtml"),
        ]
        data = _build_epub({
            "META-INF/container.xml": _CONTAINER_XML,
            "OEBPS/content.opf": _opf_epub2(chapters=chs),
            "OEBPS/toc.ncx": _ncx(ncx_titles),
            # Body is just the chapter title list — content heuristic
            # should catch it (4/4 matches > 70%).
            "OEBPS/index.xhtml": _chapter_html(
                "<p>第一章 启程</p>"
                "<p>第二章 风波</p>"
                "<p>第三章 决战</p>"
                "<p>第四章 归途</p>"
            ),
            "OEBPS/ch1.xhtml": _chapter_html("<p>启程正文</p>"),
            "OEBPS/ch2.xhtml": _chapter_html("<p>风波正文</p>"),
            "OEBPS/ch3.xhtml": _chapter_html("<p>决战正文</p>"),
            "OEBPS/ch4.xhtml": _chapter_html("<p>归途正文</p>"),
        })
        parsed = parse_epub(data, fallback_title="fb")
        titles = [c.title for c in parsed.chapters]
        assert "第一篇" not in titles
        assert titles == [
            "第一章 启程", "第二章 风波", "第三章 决战", "第四章 归途",
        ]

    def test_real_chapter_with_few_titles_kept(self):
        """A chapter that *mentions* a couple of other chapter titles
        in passing should not be misclassified as a TOC."""
        chs = [
            ("ch1", "ch1.xhtml"),
            ("ch2", "ch2.xhtml"),
            ("ch3", "ch3.xhtml"),
        ]
        ncx_titles = [
            ("第一章 启程", "ch1.xhtml"),
            ("第二章 风波", "ch2.xhtml"),
            ("第三章 决战", "ch3.xhtml"),
        ]
        # Body has 5 substantive prose lines + 1 incidental title
        # mention → 1/6 ≈ 16% match, well below the 70% threshold.
        prose = (
            "<p>这一章是真正的故事开始</p>"
            "<p>主角在山顶醒来</p>"
            "<p>四下张望发现没有他人</p>"
            "<p>第二章 风波</p>"
            "<p>记忆里残留几个名字</p>"
            "<p>他知道前路艰险</p>"
        )
        data = _build_epub({
            "META-INF/container.xml": _CONTAINER_XML,
            "OEBPS/content.opf": _opf_epub2(chapters=chs),
            "OEBPS/toc.ncx": _ncx(ncx_titles),
            "OEBPS/ch1.xhtml": _chapter_html(prose),
            "OEBPS/ch2.xhtml": _chapter_html("<p>风波正文</p>"),
            "OEBPS/ch3.xhtml": _chapter_html("<p>决战正文</p>"),
        })
        parsed = parse_epub(data, fallback_title="fb")
        titles = [c.title for c in parsed.chapters]
        assert titles == ["第一章 启程", "第二章 风波", "第三章 决战"]
