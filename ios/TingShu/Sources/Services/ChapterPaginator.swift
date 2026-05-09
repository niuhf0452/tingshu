// Per-chapter text pagination for the player's horizontal page-swipe UI.
//
// Why TextKit 1 (NSLayoutManager) and not TextKit 2 (NSTextLayoutManager):
// TextKit 1's container-flow model lets us request "fill this size, return
// the char range that fit" in one call, which is exactly what pagination
// needs. TextKit 2 forces enumerating display fragments, which complicates
// the mid-paragraph break case. Performance is comparable on chapter-sized
// inputs (~3-15ms typical, dominated by glyph generation either way).
//
// Pagination is invalidated whenever font/fontSize/viewSize changes; the
// callers (PlaybackService) hold the cache and decide when to invalidate.
#if canImport(UIKit)
import UIKit
#endif
import CoreGraphics
import Foundation


struct PageRange: Equatable, Hashable {
    let chapterId: Int
    let pageIndex: Int
    /// UTF-16 offset range in the chapter text. Suitable for slicing
    /// String.utf16 and for sentence-position lookup (server `startCol`
    /// is also a UTF-16 offset).
    let utf16Start: Int
    let utf16End: Int
}


struct ChapterPagination {
    let chapterId: Int
    let pages: [PageRange]
    /// UTF-16 offset of each source line's first char. `lineStartOffsets[i]`
    /// is the offset of the (i+1)th line (1-based to match server `startLine`).
    let lineStartOffsets: [Int]
}


enum ChapterPaginator {

    /// Paginate `text` into pages that each fit in `viewSize` when
    /// rendered with `font` (and the matching SwiftUI line spacing).
    /// Empty `text` returns a single empty page so the UI always has at
    /// least one page to show.
    static func paginate(
        chapterId: Int,
        text: String,
        font: PlatformFont,
        viewSize: CGSize,
        lineSpacing: CGFloat = 4,
    ) -> ChapterPagination {
        let lineStarts = buildLineStartOffsets(text: text)
        // Defensive: a zero-area container never accepts glyphs and would
        // loop forever. Bail with a single page covering the whole text.
        guard viewSize.width > 1, viewSize.height > 1, !text.isEmpty else {
            let span = text.utf16.count
            return ChapterPagination(
                chapterId: chapterId,
                pages: [PageRange(
                    chapterId: chapterId, pageIndex: 0,
                    utf16Start: 0, utf16End: span,
                )],
                lineStartOffsets: lineStarts,
            )
        }

        #if canImport(UIKit)
        let paragraph = NSMutableParagraphStyle()
        paragraph.lineSpacing = lineSpacing
        let attributes: [NSAttributedString.Key: Any] = [
            .font: font,
            .paragraphStyle: paragraph,
        ]
        let attributed = NSAttributedString(string: text, attributes: attributes)
        let storage = NSTextStorage(attributedString: attributed)
        let layoutManager = NSLayoutManager()
        // No character substitution / hyphenation surprises.
        layoutManager.allowsNonContiguousLayout = false
        storage.addLayoutManager(layoutManager)

        var pages: [PageRange] = []
        var pageIdx = 0
        var glyphCursor = 0
        // Hard upper bound to avoid infinite loops on pathological inputs
        // (zero-glyph containers etc.). 10k pages = ~5M chars, far above
        // any realistic chapter.
        let pageLimit = 10_000

        while glyphCursor < layoutManager.numberOfGlyphs && pageIdx < pageLimit {
            let container = NSTextContainer(size: viewSize)
            container.lineFragmentPadding = 0
            container.maximumNumberOfLines = 0
            layoutManager.addTextContainer(container)
            let glyphRange = layoutManager.glyphRange(for: container)
            // Defensive: if a container couldn't accept any glyph (e.g.
            // an oversized inline attachment), force-advance one glyph
            // so we don't loop. Shouldn't happen for plain text.
            if glyphRange.length == 0 {
                glyphCursor += 1
                continue
            }
            let charRange = layoutManager.characterRange(
                forGlyphRange: glyphRange, actualGlyphRange: nil,
            )
            pages.append(PageRange(
                chapterId: chapterId,
                pageIndex: pageIdx,
                utf16Start: charRange.location,
                utf16End: charRange.location + charRange.length,
            ))
            pageIdx += 1
            glyphCursor = NSMaxRange(glyphRange)
        }
        if pages.isEmpty {
            pages = [PageRange(
                chapterId: chapterId, pageIndex: 0,
                utf16Start: 0, utf16End: text.utf16.count,
            )]
        }
        return ChapterPagination(
            chapterId: chapterId, pages: pages, lineStartOffsets: lineStarts,
        )
        #else
        // Non-UIKit platforms (tests on macOS without TextKit?): single page.
        return ChapterPagination(
            chapterId: chapterId,
            pages: [PageRange(
                chapterId: chapterId, pageIndex: 0,
                utf16Start: 0, utf16End: text.utf16.count,
            )],
            lineStartOffsets: lineStarts,
        )
        #endif
    }

    /// Convert (1-based line, 0-based UTF-16 col) to a chapter-relative
    /// UTF-16 offset. Out-of-range inputs clamp to 0 / end.
    static func utf16Offset(
        line: Int, col: Int, lineStarts: [Int], textUTF16Count: Int,
    ) -> Int {
        guard line >= 1, !lineStarts.isEmpty else { return 0 }
        let lineIdx = min(line - 1, lineStarts.count - 1)
        return min(textUTF16Count, lineStarts[lineIdx] + max(0, col))
    }

    /// First page whose range contains `offset` (i.e. first page with
    /// `utf16End > offset`). Returns last page index when offset is past
    /// the chapter's end.
    static func pageContaining(offset: Int, in pages: [PageRange]) -> Int {
        for (i, p) in pages.enumerated() where offset < p.utf16End {
            return i
        }
        return max(0, pages.count - 1)
    }

    private static func buildLineStartOffsets(text: String) -> [Int] {
        var starts: [Int] = [0]
        var offset = 0
        for unit in text.utf16 {
            if unit == 0x000A {
                starts.append(offset + 1)
            }
            offset += 1
        }
        return starts
    }
}


/// Cross-platform UIFont alias so the paginator's signature is the same
/// in iOS app code and any future macOS/test target.
#if canImport(UIKit)
typealias PlatformFont = UIFont
#elseif canImport(AppKit)
import AppKit
typealias PlatformFont = NSFont
#else
typealias PlatformFont = AnyObject
#endif
