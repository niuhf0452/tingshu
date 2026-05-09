"""EPUB book parser (§2.4.1 EPUB path).

EPUB is a ZIP of XHTML + metadata. The package specification gives us:
  - ``META-INF/container.xml`` — entry point; points at the root package file
  - ``*.opf`` — the package document, listing:
      - ``<manifest>``: every resource in the book (id → href)
      - ``<spine>``: the linear reading order (list of itemrefs → manifest ids)
  - ``toc.ncx`` (EPUB 2) or ``*.xhtml`` with ``<nav epub:type="toc">`` (EPUB 3):
      chapter titles keyed by href

Compared to TXT, we skip the LLM-driven chapter detection entirely (§2.4.1):
the TOC is authoritative. For each spine item we extract plain text by
stripping HTML tags.

This module is intentionally dependency-free beyond the stdlib
(``zipfile`` + ``xml.etree.ElementTree`` + ``html.parser``). Larger EPUB
edge cases (DRM, RTL, fixed-layout) are out of scope.
"""
from __future__ import annotations

import io
import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Optional

from .txt import ParsedBook, ParsedChapter


log = logging.getLogger(__name__)


# --- XML namespace shortcuts ---------------------------------------------

_NS = {
    "container": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
    "ncx": "http://www.daisy.org/z3986/2005/ncx/",
    "xhtml": "http://www.w3.org/1999/xhtml",
    "epub": "http://www.idpf.org/2007/ops",
}

_NS_OPF = f"{{{_NS['opf']}}}"
_NS_NCX = f"{{{_NS['ncx']}}}"
_NS_DC = f"{{{_NS['dc']}}}"
_NS_XHTML = f"{{{_NS['xhtml']}}}"
_NS_EPUB = f"{{{_NS['epub']}}}"


# --- result types --------------------------------------------------------


class EpubParseError(ValueError):
    """Raised for malformed EPUB files."""


@dataclass
class _SpineItem:
    idref: str
    href: str  # relative to opf dir


# --- public entry point --------------------------------------------------


def parse_epub(raw: bytes, fallback_title: str) -> ParsedBook:
    """Decode an EPUB from raw bytes into ``ParsedBook``.

    Raises ``EpubParseError`` on malformed archives. Empty or unrecognised
    EPUBs fall back to a single-chapter ``ParsedBook`` with the fallback
    title so upstream code always has at least one chapter.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise EpubParseError(f"not a valid ZIP archive: {exc}") from exc

    with archive:
        opf_path = _find_opf_path(archive)
        opf_root = _read_xml(archive, opf_path)

        title = _opf_title(opf_root) or fallback_title
        author = _opf_author(opf_root) or ""

        opf_dir = _parent_dir(opf_path)
        manifest = _parse_manifest(opf_root, opf_dir)
        spine = _parse_spine(opf_root, manifest)
        toc_titles = _parse_toc(archive, opf_root, manifest, opf_dir)

        chapters: list[ParsedChapter] = []
        for item in spine:
            text = _extract_chapter_text(archive, item.href)
            if not text.strip():
                continue
            title_for_item = toc_titles.get(item.href, "")
            if not title_for_item:
                title_for_item = _first_heading_in_text(text) or _default_title(len(chapters) + 1)
            if _is_toc_page(title_for_item, text, toc_titles):
                # In-spine "目录" / "Contents" page. Skipping it both
                # removes a meaningless first chapter from playback AND
                # avoids the broken UX where listed chapter titles look
                # tappable but aren't (the page is plain text after our
                # HTML→text strip; the underlying <a> hrefs are gone).
                log.info(
                    "epub: skipping in-spine TOC page %s (title=%r)",
                    item.href, title_for_item,
                )
                continue
            chapters.append(ParsedChapter(title=title_for_item.strip(), text=text.strip()))

        if not chapters:
            log.warning("epub %s produced no chapters; wrapping raw-extracted text", fallback_title)
            fallback_text = "\n\n".join(
                _extract_chapter_text(archive, item.href) for item in spine
            ).strip()
            chapters = [ParsedChapter(title=fallback_title, text=fallback_text)]

        return ParsedBook(title=title, author=author, chapters=chapters)


# --- container + opf discovery -------------------------------------------


def _find_opf_path(archive: zipfile.ZipFile) -> str:
    try:
        container_xml = archive.read("META-INF/container.xml")
    except KeyError as exc:
        raise EpubParseError("missing META-INF/container.xml") from exc
    try:
        root = ET.fromstring(container_xml)
    except ET.ParseError as exc:
        raise EpubParseError(f"invalid container.xml: {exc}") from exc
    rootfile = root.find(".//{" + _NS["container"] + "}rootfile")
    if rootfile is None or not rootfile.get("full-path"):
        raise EpubParseError("container.xml missing rootfile@full-path")
    return rootfile.get("full-path")  # type: ignore[return-value]


def _read_xml(archive: zipfile.ZipFile, path: str) -> ET.Element:
    try:
        raw = archive.read(path)
    except KeyError as exc:
        raise EpubParseError(f"missing file in EPUB: {path}") from exc
    try:
        return ET.fromstring(raw)
    except ET.ParseError as exc:
        raise EpubParseError(f"invalid XML in {path}: {exc}") from exc


# --- metadata extraction -------------------------------------------------


def _opf_title(opf: ET.Element) -> Optional[str]:
    node = opf.find(f".//{_NS_DC}title")
    return node.text.strip() if (node is not None and node.text) else None


def _opf_author(opf: ET.Element) -> Optional[str]:
    node = opf.find(f".//{_NS_DC}creator")
    return node.text.strip() if (node is not None and node.text) else None


# --- manifest + spine ----------------------------------------------------


def _parse_manifest(opf: ET.Element, opf_dir: str) -> dict[str, dict]:
    """Return ``{item_id: {href, media_type}}``. Paths are archive-relative."""
    manifest: dict[str, dict] = {}
    for item in opf.findall(f"{_NS_OPF}manifest/{_NS_OPF}item"):
        item_id = item.get("id")
        href = item.get("href")
        mt = item.get("media-type", "")
        if not item_id or not href:
            continue
        manifest[item_id] = {
            "href": _join(opf_dir, href),
            "media_type": mt,
            "properties": item.get("properties", ""),
        }
    return manifest


def _parse_spine(opf: ET.Element, manifest: dict[str, dict]) -> list[_SpineItem]:
    spine_node = opf.find(f"{_NS_OPF}spine")
    if spine_node is None:
        raise EpubParseError("opf missing <spine>")
    items: list[_SpineItem] = []
    for itemref in spine_node.findall(f"{_NS_OPF}itemref"):
        idref = itemref.get("idref")
        if not idref or idref not in manifest:
            continue
        # Skip non-linear items (footnotes, index pages) and the nav
        # document itself — it lists chapter titles, not chapter content.
        if itemref.get("linear", "yes").lower() == "no":
            continue
        props = manifest[idref].get("properties", "")
        if "nav" in props.split():
            continue
        items.append(_SpineItem(idref=idref, href=manifest[idref]["href"]))
    return items


# --- TOC parsing (ncx + nav) --------------------------------------------


def _parse_toc(
    archive: zipfile.ZipFile,
    opf: ET.Element,
    manifest: dict[str, dict],
    opf_dir: str,
) -> dict[str, str]:
    """Return ``{normalised_href: title}`` from either the NCX (EPUB 2)
    or the nav document (EPUB 3). On failure returns an empty dict so
    callers fall back to heading-based titles.
    """
    # EPUB 3: item with properties="nav".
    for meta in manifest.values():
        if "nav" in meta.get("properties", "").split():
            nav_path = meta["href"]
            try:
                return _parse_nav(archive, nav_path)
            except EpubParseError as exc:
                log.warning("failed to parse nav at %s: %s", nav_path, exc)

    # EPUB 2: spine @toc references the NCX item id.
    spine_node = opf.find(f"{_NS_OPF}spine")
    if spine_node is not None:
        ncx_id = spine_node.get("toc")
        if ncx_id and ncx_id in manifest:
            try:
                return _parse_ncx(archive, manifest[ncx_id]["href"])
            except EpubParseError as exc:
                log.warning("failed to parse ncx at %s: %s", manifest[ncx_id]["href"], exc)

    # EPUB 2 fallback: find any manifest item with media_type=application/x-dtbncx+xml.
    for meta in manifest.values():
        if meta.get("media_type") == "application/x-dtbncx+xml":
            try:
                return _parse_ncx(archive, meta["href"])
            except EpubParseError as exc:
                log.warning("failed to parse ncx at %s: %s", meta["href"], exc)

    return {}


def _parse_ncx(archive: zipfile.ZipFile, path: str) -> dict[str, str]:
    root = _read_xml(archive, path)
    ncx_dir = _parent_dir(path)
    result: dict[str, str] = {}
    for nav_point in root.findall(f".//{_NS_NCX}navPoint"):
        label = nav_point.find(f"{_NS_NCX}navLabel/{_NS_NCX}text")
        content = nav_point.find(f"{_NS_NCX}content")
        if label is None or content is None:
            continue
        text = (label.text or "").strip()
        src = content.get("src", "").split("#", 1)[0]
        if not text or not src:
            continue
        result[_join(ncx_dir, src)] = text
    return result


def _parse_nav(archive: zipfile.ZipFile, path: str) -> dict[str, str]:
    root = _read_xml(archive, path)
    nav_dir = _parent_dir(path)
    # Find the first <nav epub:type="toc">, fall back to any <nav>.
    toc_nav = None
    for nav in root.findall(f".//{_NS_XHTML}nav"):
        if nav.get(f"{_NS_EPUB}type") == "toc":
            toc_nav = nav
            break
    if toc_nav is None:
        toc_nav = root.find(f".//{_NS_XHTML}nav")
    if toc_nav is None:
        return {}

    result: dict[str, str] = {}
    for link in toc_nav.findall(f".//{_NS_XHTML}a"):
        href = (link.get("href") or "").split("#", 1)[0]
        text = "".join(link.itertext()).strip()
        if not href or not text:
            continue
        result[_join(nav_dir, href)] = text
    return result


# --- HTML → plain text ---------------------------------------------------


class _TextExtractor(HTMLParser):
    """Extract visible text from (X)HTML, inserting newlines around
    block-level elements so the resulting text looks paragraph-shaped.
    """

    _BLOCK_TAGS = frozenset([
        "p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "tr", "hr", "section", "article", "header", "footer",
    ])
    # Head + metadata tags whose *content* should never reach the text buffer
    # (e.g. <title>章节</title> inside <head> would otherwise leak into the
    # chapter body).
    _SKIP_TAGS = frozenset(["script", "style", "head", "title", "meta", "link"])

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._buf: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self._BLOCK_TAGS:
            self._buf.append("\n")

    def handle_endtag(self, tag: str):
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self._BLOCK_TAGS:
            self._buf.append("\n")

    def handle_startendtag(self, tag: str, attrs):
        if tag in self._BLOCK_TAGS:
            self._buf.append("\n")

    def handle_data(self, data: str):
        if self._skip_depth:
            return
        self._buf.append(data)

    def text(self) -> str:
        raw = "".join(self._buf)
        # Collapse whitespace runs per line but keep paragraph breaks.
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw.split("\n")]
        # Collapse runs of empty lines into a single one.
        out: list[str] = []
        blank = False
        for line in lines:
            if line:
                out.append(line)
                blank = False
            elif not blank:
                out.append("")
                blank = True
        return "\n".join(out).strip()


def _extract_chapter_text(archive: zipfile.ZipFile, href: str) -> str:
    try:
        raw = archive.read(href)
    except KeyError:
        log.warning("epub spine item missing: %s", href)
        return ""
    try:
        html = raw.decode("utf-8")
    except UnicodeDecodeError:
        html = raw.decode("utf-8", errors="replace")
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


# --- small helpers -------------------------------------------------------


# Spine items whose visible title matches one of these labels are the
# book's "Table of Contents" page rendered as plain HTML — different
# from the EPUB 3 nav doc (already filtered via properties=nav) and
# the EPUB 2 NCX (referenced via spine[@toc], never in the spine).
# Common Chinese variants + English; trailing whitespace tolerated.
_TOC_TITLE_RE = re.compile(
    r"^(?:目\s*[录錄次]|"
    r"contents?|table\s+of\s+contents|toc|"
    r"chapters?\s+list|chapter\s+index)\s*$",
    re.IGNORECASE,
)


def _is_toc_page(
    title: str,
    text: str,
    toc_titles: dict[str, str],
) -> bool:
    """Heuristic: this spine item is a clickable index of other chapters,
    not a real chapter.

    Two combined signals (any one suffices):

    1. **Title match** — the spine item's resolved title is a known
       "Table of Contents" label (目录 / Contents / TOC / …). Cheap and
       catches the typical case where the book ships an in-spine TOC.

    2. **Content match** — the body's non-trivial lines mostly match
       chapter titles from the EPUB's parsed TOC dictionary. Catches
       hand-rolled TOC pages whose own title is generic (e.g. just "第
       一篇") but whose content is dominated by other chapters' titles.

    The 70% threshold for signal 2 is high enough that a real chapter
    incidentally listing a few section headings won't trip it.
    """
    if title and _TOC_TITLE_RE.match(title.strip()):
        return True
    if not toc_titles:
        return False
    # Drop blank lines and pure-punctuation lines so an early chapter
    # whose body opens with "第一节" + a few short bullets isn't unduly
    # weighted toward the TOC verdict.
    body_lines = [
        ln.strip() for ln in text.split("\n")
        if len(ln.strip()) > 1
    ]
    if len(body_lines) < 3:
        return False
    title_set = {t.strip() for t in toc_titles.values() if t.strip()}
    if not title_set:
        return False
    matches = sum(1 for ln in body_lines if ln in title_set)
    return matches / len(body_lines) >= 0.7


_HEADING_LINE_RE = re.compile(r"^\s*.{1,60}\s*$")


def _first_heading_in_text(text: str) -> Optional[str]:
    for line in text.split("\n", 5):
        line = line.strip()
        if line and _HEADING_LINE_RE.match(line):
            return line
    return None


def _default_title(index: int) -> str:
    return f"第 {index} 章"


def _join(base: str, href: str) -> str:
    """Join an archive-relative ``base`` directory with a spec-relative
    ``href``. EPUB paths are POSIX-style regardless of host OS.
    """
    if not base:
        return href.lstrip("/")
    # Resolve ../ segments manually.
    parts: list[str] = [p for p in base.split("/") if p]
    for piece in href.split("/"):
        if piece == "" or piece == ".":
            continue
        if piece == "..":
            if parts:
                parts.pop()
            continue
        parts.append(piece)
    return "/".join(parts)


def _parent_dir(path: str) -> str:
    if "/" not in path:
        return ""
    return path.rsplit("/", 1)[0]


