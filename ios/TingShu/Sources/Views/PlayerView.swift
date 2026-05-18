// Main reading/playback screen.
//
// Rendering model (revised 2026-05-16):
//
// - The chapter text is rendered by a single TextKit-1 `UITextView`
//   (see `ChapterTextView`), not a SwiftUI `LazyVStack` of `Text` rows.
//   The switch was made so follow-mode can centre the *playing
//   sentence* precisely: `NSLayoutManager` exposes the exact rect of any
//   character range, which SwiftUI's `Text` does not. It also removes
//   the old TextKit-vs-SwiftUI line-metric mismatch — layout and
//   rendering are now a single engine.
// - Follow mode auto-centres the current playing sentence on every
//   position update. A user drag exits follow mode (the explicit "I'm
//   reading ahead" signal); re-enable via the bottom-bar follow button.
// - Chapter switching goes through the bottom-bar 上一章 / 下一章
//   buttons or the TOC — there is no horizontal swipe.
import SwiftUI
#if canImport(UIKit)
import UIKit
#endif


/// A scroll request handed to ``ChapterTextView``. `token` is bumped on
/// every request so the text view re-acts even when the target is
/// unchanged (e.g. re-engaging follow without the sentence moving).
private struct ChapterScrollCommand: Equatable {
    enum Target: Equatable {
        /// Centre the sentence at this index of the displayed chapter.
        case sentence(Int)
        /// Snap to the top of the chapter.
        case top
    }
    var token: Int
    var target: Target
    var animated: Bool
}


struct PlayerView: View {
    let book: LocalBook

    @EnvironmentObject var playback: PlaybackService
    @EnvironmentObject var settings: SettingsStore

    @State private var showTOC = false
    @State private var showSleepSheet = false
    @State private var transientNotice: String?
    @State private var noticeDismissTask: Task<Void, Never>?
    /// Latest scroll request for the chapter text view. Bumped by the
    /// follow-mode triggers below (position advance, chapter switch,
    /// post-load retry, follow-button re-engage).
    @State private var scrollCommand = ChapterScrollCommand(
        token: 0, target: .top, animated: false,
    )

    var body: some View {
        VStack(spacing: 0) {
            chapterScrollArea
            networkActivityArea
            prefetchProgressBar
            controlsBar
        }
        .background(Color.readerBackground)
        .overlay(alignment: .bottom) { transientNoticeBanner }
        .navigationTitle(browseChapterTitle)
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .toolbar {
            ToolbarItem(placement: .topTrailing) {
                Button { showTOC = true } label: { Image(systemName: "list.bullet") }
            }
        }
        .sheet(isPresented: $showTOC) {
            ChaptersTOCView(
                book: book,
                currentChapterId: playback.state.browseChapterId,
            ) { chapterId in
                playback.state.followMode = false
                Task { await playback.setBrowseChapter(chapterId: chapterId) }
                showTOC = false
            }
        }
        .sheet(isPresented: $showSleepSheet) {
            SleepTimerSheet(
                endDate: playback.state.sleepTimerEndDate,
                onSelect: { minutes in
                    playback.setSleepTimer(minutes: minutes)
                    showSleepSheet = false
                },
            )
            .presentationDetents([.medium])
        }
        .task { await playback.open(book: book) }
        .onAppear { setIdleTimerDisabled(playback.state.followMode) }
        .onChange(of: playback.state.followMode) { _, newValue in
            setIdleTimerDisabled(newValue)
        }
        .onDisappear {
            playback.stop()
            setIdleTimerDisabled(false)
        }
        .alert(
            "出错了",
            isPresented: Binding(
                get: { playback.state.error != nil },
                set: { if !$0 { playback.state.error = nil } }
            ),
            actions: { Button("确定") { playback.state.error = nil } },
            message: { Text(playback.state.error ?? "") }
        )
    }

    private func setIdleTimerDisabled(_ disabled: Bool) {
        #if os(iOS)
        UIApplication.shared.isIdleTimerDisabled = disabled
        #endif
    }

    private var browseChapterTitle: String {
        guard let id = playback.state.browseChapterId,
              let ch = book.meta.chapters.first(where: { $0.id == id }) else {
            return book.title
        }
        return ch.title
    }

    // MARK: - chapter text

    /// The chapter reading surface. While the browse chapter's text is
    /// still loading the area shows a spinner; once text is present the
    /// `UITextView`-backed ``ChapterTextView`` takes over (a plain
    /// SwiftUI fallback is used on non-UIKit platforms so the package
    /// still builds for macOS / tests).
    private var chapterScrollArea: some View {
        let chapterId = playback.state.browseChapterId
        let text = playback.state.browseChapterText
        let sentences = playback.state.browseChapterSentences
        return Group {
            if text.isEmpty {
                VStack(spacing: 8) {
                    ProgressView()
                    Text("加载章节…").font(.caption).foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                #if canImport(UIKit)
                ChapterTextView(
                    chapterId: chapterId ?? -1,
                    text: text,
                    fontSize: CGFloat(settings.fontSize),
                    sentences: sentences,
                    highlightIndex: highlightSentenceIndex,
                    scrollCommand: scrollCommand,
                    onSentenceDoubleTap: { sentenceIndex in
                        handleDoubleTap(sentenceIndex: sentenceIndex, chapterId: chapterId)
                    },
                    onUserScroll: {
                        if playback.state.followMode {
                            playback.state.followMode = false
                        }
                    },
                )
                #else
                ScrollView {
                    Text(text)
                        .font(.system(size: CGFloat(settings.fontSize)))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                }
                #endif
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onChange(of: playback.state.position) { _, _ in
            // Playback advanced — centre the new sentence if following.
            requestFollowScroll(animated: true)
        }
        .onChange(of: playback.state.browseChapterId) { _, _ in
            handleChapterChange()
        }
        .onChange(of: playback.state.browseChapterText) { _, _ in
            // Text landed (often after a chapter switch) — re-issue the
            // position so a resumed sentence ends up centred.
            handleChapterContentLanded()
        }
        .onChange(of: playback.state.browseChapterSentences.count) { _, _ in
            // Sentence metadata landed after the text — same retry.
            handleChapterContentLanded()
        }
    }

    /// Index of the currently-playing sentence within the *displayed*
    /// chapter, or nil when playback is in a different chapter / has no
    /// position / the meta hasn't loaded. Drives the highlight only —
    /// the highlight tracks playback even when follow mode is off.
    private var highlightSentenceIndex: Int? {
        guard let pos = playback.state.position,
              pos.chapterId == playback.state.browseChapterId else { return nil }
        let count = playback.state.browseChapterSentences.count
        guard pos.sentenceIndex >= 0, pos.sentenceIndex < count else { return nil }
        return pos.sentenceIndex
    }

    /// Request a follow-mode centre on the playing sentence. No-op
    /// unless follow is on and the play position is a valid sentence in
    /// the displayed chapter.
    private func requestFollowScroll(animated: Bool) {
        guard playback.state.followMode,
              let pos = playback.state.position,
              pos.chapterId == playback.state.browseChapterId,
              pos.sentenceIndex >= 0,
              pos.sentenceIndex < playback.state.browseChapterSentences.count else {
            return
        }
        scrollCommand = ChapterScrollCommand(
            token: scrollCommand.token + 1,
            target: .sentence(pos.sentenceIndex),
            animated: animated,
        )
    }

    /// Reposition on a chapter switch: centre on the play position when
    /// follow is on and the position is in the new chapter, otherwise
    /// snap to the top (manual TOC / prev-next navigation).
    private func handleChapterChange() {
        if playback.state.followMode,
           let pos = playback.state.position,
           pos.chapterId == playback.state.browseChapterId,
           pos.sentenceIndex >= 0,
           pos.sentenceIndex < playback.state.browseChapterSentences.count {
            scrollCommand = ChapterScrollCommand(
                token: scrollCommand.token + 1,
                target: .sentence(pos.sentenceIndex),
                animated: false,
            )
        } else {
            scrollCommand = ChapterScrollCommand(
                token: scrollCommand.token + 1, target: .top, animated: false,
            )
        }
    }

    /// Re-centre once chapter text / meta arrive after the position is
    /// already known (e.g. resuming a book: the position is restored
    /// before the text finishes loading off disk).
    private func handleChapterContentLanded() {
        guard playback.state.followMode,
              let pos = playback.state.position,
              pos.chapterId == playback.state.browseChapterId,
              pos.sentenceIndex >= 0,
              pos.sentenceIndex < playback.state.browseChapterSentences.count else {
            return
        }
        scrollCommand = ChapterScrollCommand(
            token: scrollCommand.token + 1,
            target: .sentence(pos.sentenceIndex),
            animated: false,
        )
    }

    /// Double-tap → jump playback to the double-tapped sentence.
    /// `sentenceIndex` is resolved precisely by character range inside
    /// ``ChapterTextView`` — so the second sentence of a paragraph maps
    /// to itself, not to the paragraph's first sentence. A negative
    /// index means the tap fell outside any sentence (header /
    /// inter-sentence gap), which is intentionally silent.
    private func handleDoubleTap(sentenceIndex: Int, chapterId: Int?) {
        guard let cid = chapterId else { return }
        if playback.chapterSentences(for: cid).isEmpty {
            showNotice("章节分析中，稍后再试")
            return
        }
        guard sentenceIndex >= 0 else { return }
        Task { await playback.jumpPlay(chapterId: cid, sentenceIndex: sentenceIndex) }
    }

    // MARK: - controls bar

    /// Bottom bar order toggles by left-hand mode. Both layouts expose
    /// the same actions; only the side they sit on flips. Chapter
    /// progress always sits on the *opposite* side from the primary
    /// thumb area to avoid overlapping with the play button.
    private var controlsBar: some View {
        HStack(spacing: 18) {
            if settings.leftHandMode {
                playButton
                followButton
                prevChapterButton
                nextChapterButton
                sleepTimerButton
                settingsButton
                Spacer()
                chapterProgressLabel
            } else {
                chapterProgressLabel
                Spacer()
                settingsButton
                sleepTimerButton
                nextChapterButton
                prevChapterButton
                followButton
                playButton
            }
        }
        .padding(.horizontal)
        .padding(.top, 8)
        .padding(.bottom, 4)
        .background(.bar)
    }

    private var playButton: some View {
        let isPlaying = playback.state.isPlaying
        return Button(action: {
            if isPlaying { playback.pause() } else { playback.play() }
        }) {
            Image(systemName: isPlaying ? "pause.circle.fill" : "play.circle.fill")
                .font(.system(size: 44))
        }
        .accessibilityLabel(isPlaying ? "暂停" : "播放")
    }

    /// Follow toggle. Tapping while off re-engages follow + snaps to the
    /// current sentence; tapping while on disables follow (so the user
    /// can scroll ahead without the view yanking back). Filled vs
    /// outlined icon signals state — no accent tint, per the bar's
    /// "only the play button gets the accent colour" rule.
    private var followButton: some View {
        Button(action: tapFollowButton) {
            Image(systemName: playback.state.followMode ? "location.fill" : "location")
                .font(.title3)
        }
        .foregroundStyle(.primary)
        .accessibilityLabel(playback.state.followMode ? "关闭跟读" : "跟读")
    }

    private var prevChapterButton: some View {
        Button(action: { tapPrevChapter() }) {
            Image(systemName: "backward.end")
                .font(.title3)
        }
        .foregroundStyle(.primary)
        .disabled(prevChapterId == nil)
        .accessibilityLabel("上一章")
    }

    private var nextChapterButton: some View {
        Button(action: { tapNextChapter() }) {
            Image(systemName: "forward.end")
                .font(.title3)
        }
        .foregroundStyle(.primary)
        .disabled(nextChapterId == nil)
        .accessibilityLabel("下一章")
    }

    private var sleepTimerButton: some View {
        Button(action: { showSleepSheet = true }) {
            // Filled vs outline icon already conveys active/idle —
            // no accent tint needed (per design: only the play button
            // earns the accent colour on the bar).
            Image(systemName: sleepTimerActive ? "moon.fill" : "moon")
                .font(.title3)
        }
        .foregroundStyle(.primary)
        .accessibilityLabel("定时停止")
    }

    /// Push (not sheet) so the system back button on the navigation bar
    /// returns to the player view rather than slamming the modal away.
    /// Routes to ``PlayerSettingsView`` (subset of app settings + the
    /// per-book character roster), not the full app settings — those
    /// still live behind the bookshelf gear button.
    private var settingsButton: some View {
        NavigationLink {
            PlayerSettingsView(book: book)
        } label: {
            Image(systemName: "gearshape")
                .font(.title3)
        }
        .foregroundStyle(.primary)
        .accessibilityLabel("设置")
    }

    private var chapterProgressLabel: some View {
        HStack(spacing: 6) {
            if let countdown = sleepCountdownText {
                Text(countdown)
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(Color.accentColor)
            }
            Text(chapterProgress)
                .font(.caption)
                .foregroundStyle(.secondary)
                .monospacedDigit()
        }
    }

    // MARK: - chapter navigation

    private var sortedChapterIds: [Int] {
        book.meta.chapters.map(\.id).sorted()
    }

    private var prevChapterId: Int? {
        guard let cid = playback.state.browseChapterId else { return nil }
        let ids = sortedChapterIds
        guard let idx = ids.firstIndex(of: cid), idx > 0 else { return nil }
        return ids[idx - 1]
    }

    private var nextChapterId: Int? {
        guard let cid = playback.state.browseChapterId else { return nil }
        let ids = sortedChapterIds
        guard let idx = ids.firstIndex(of: cid), idx < ids.count - 1 else { return nil }
        return ids[idx + 1]
    }

    private func tapPrevChapter() {
        guard let target = prevChapterId else { return }
        playback.state.followMode = false
        Task { await playback.setBrowseChapter(chapterId: target) }
    }

    private func tapNextChapter() {
        guard let target = nextChapterId else { return }
        playback.state.followMode = false
        Task { await playback.setBrowseChapter(chapterId: target) }
    }

    // MARK: - chapter / sleep helpers

    private var chapterProgress: String {
        guard let current = playback.state.browseChapterId else { return "" }
        return "\(current) / \(book.meta.chapters.count)"
    }

    /// Prefetch progress strip — same semantics as before: cached
    /// sentences in browse chapter + cursor at current play sentence.
    @ViewBuilder
    private var prefetchProgressBar: some View {
        let total = playback.state.browseChapterSentences.count
        if total > 0 {
            let pos = playback.state.position
            let curIdx: Int? = (pos?.chapterId == playback.state.browseChapterId)
                ? pos?.sentenceIndex : nil
            PrefetchProgressBar(
                totalSentences: total,
                cachedIndices: playback.state.cachedSentencesInBrowseChapter,
                currentIndex: curIdx,
            )
        }
    }

    @ViewBuilder
    private var networkActivityArea: some View {
        if let item = networkActivityItem {
            HStack(spacing: 6) {
                ProgressView().controlSize(.mini)
                Text(item)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Spacer()
            }
            .padding(.horizontal)
            .padding(.vertical, 4)
            .background(Color.gray.opacity(0.06))
            .transition(.opacity)
        }
    }

    private var networkActivityItem: String? {
        if playback.state.fetchingBrowseSentences {
            return "正在分析章节…"
        }
        return nil
    }

    private func showNotice(_ message: String) {
        transientNotice = message
        noticeDismissTask?.cancel()
        noticeDismissTask = Task { [message] in
            try? await Task.sleep(nanoseconds: 2_500_000_000)
            if !Task.isCancelled {
                await MainActor.run {
                    if transientNotice == message { transientNotice = nil }
                }
            }
        }
    }

    @ViewBuilder
    private var transientNoticeBanner: some View {
        if let notice = transientNotice {
            Text(notice)
                .font(.caption)
                .foregroundStyle(.white)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(
                    Capsule().fill(Color.black.opacity(0.78)),
                )
                .padding(.bottom, 80)
                .transition(.opacity.combined(with: .move(edge: .bottom)))
                .animation(.easeInOut(duration: 0.18), value: transientNotice)
                .allowsHitTesting(false)
        }
    }

    /// Toggle follow mode. When turning on, also snap to the current
    /// play position so the user immediately sees what's playing —
    /// otherwise re-engaging follow after a manual scroll appears to
    /// "do nothing" until the next position update fires.
    private func tapFollowButton() {
        if playback.state.followMode {
            playback.state.followMode = false
            return
        }
        playback.state.followMode = true
        if let p = playback.state.position,
           p.chapterId != playback.state.browseChapterId {
            // Cross-chapter: switching the browse chapter triggers
            // handleChapterChange, which centres on the play position.
            Task { await playback.setBrowseChapter(chapterId: p.chapterId) }
        } else {
            // Same chapter: snap to the playing sentence immediately.
            requestFollowScroll(animated: true)
        }
    }

    private var sleepTimerActive: Bool {
        playback.state.sleepTimerEndDate != nil
    }

    private var sleepCountdownText: String? {
        guard let end = playback.state.sleepTimerEndDate else { return nil }
        let remain = max(0, Int(end.timeIntervalSinceNow))
        let m = remain / 60
        let s = remain % 60
        return String(format: "%02d:%02d", m, s)
    }
}


#if canImport(UIKit)

/// `UITextView` subclass that reports layout passes. A scroll requested
/// before the view had a non-zero size (first appearance) is parked and
/// flushed here once the size is real.
private final class ChapterUITextView: UITextView {
    var onLayout: (() -> Void)?

    override func layoutSubviews() {
        super.layoutSubviews()
        onLayout?()
    }
}


/// The chapter reading surface, backed by a TextKit-1 `UITextView`.
///
/// Why a `UITextView` and not a SwiftUI `LazyVStack` of `Text` rows (the
/// previous design): playback advances sentence-by-sentence, and one
/// source paragraph holds several sentences. SwiftUI's `Text` exposes no
/// sub-paragraph geometry, so follow-mode could only centre the whole
/// paragraph. A `UITextView`'s `NSLayoutManager` yields the exact rect of
/// any character range, so we centre the *sentence* precisely.
///
/// Every behaviour of the old surface is preserved:
/// - the body / playing-sentence two-tier palette (foreground-colour
///   only — no metric-shifting weight or fill);
/// - double-tap a line to jump playback there;
/// - a user drag exits follow mode;
/// - follow-mode auto-centre, chapter-switch repositioning, and the
///   post-load retry when text / meta arrive after the position.
private struct ChapterTextView: UIViewRepresentable {
    let chapterId: Int
    let text: String
    let fontSize: CGFloat
    let sentences: [Sentence]
    let highlightIndex: Int?
    let scrollCommand: ChapterScrollCommand
    /// Called with the index of the double-tapped sentence, or -1 when
    /// the tap fell outside every sentence.
    let onSentenceDoubleTap: (Int) -> Void
    let onUserScroll: () -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> ChapterUITextView {
        // Explicit TextKit-1 stack: accessing `.layoutManager` on a
        // default (TextKit-2) UITextView would force a downgrade anyway,
        // and we need `NSLayoutManager.boundingRect(forGlyphRange:)` for
        // precise sentence centring.
        let storage = NSTextStorage()
        let layoutManager = NSLayoutManager()
        layoutManager.allowsNonContiguousLayout = false
        storage.addLayoutManager(layoutManager)
        let container = NSTextContainer(
            size: CGSize(width: 0, height: CGFloat.greatestFiniteMagnitude),
        )
        container.widthTracksTextView = true
        container.lineFragmentPadding = 0
        layoutManager.addTextContainer(container)

        let tv = ChapterUITextView(frame: .zero, textContainer: container)
        tv.isEditable = false
        // Not selectable: matches the old SwiftUI `Text` (no selection)
        // and frees the tap gestures for our double-tap-to-jump.
        tv.isSelectable = false
        tv.isScrollEnabled = true
        tv.alwaysBounceVertical = true
        tv.backgroundColor = UIColor.readerBackground
        // Replaces the old `.padding(.horizontal, 16).padding(.vertical, 8)`.
        tv.textContainerInset = UIEdgeInsets(top: 8, left: 16, bottom: 8, right: 16)
        // Deterministic offset maths — no auto safe-area insets.
        tv.contentInsetAdjustmentBehavior = UIScrollView.ContentInsetAdjustmentBehavior.never
        tv.delegate = context.coordinator

        let doubleTap = UITapGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handleDoubleTap(_:)),
        )
        doubleTap.numberOfTapsRequired = 2
        tv.addGestureRecognizer(doubleTap)

        context.coordinator.textView = tv
        tv.onLayout = { [weak coordinator = context.coordinator] in
            coordinator?.flushPendingScrollIfPossible()
        }
        return tv
    }

    func updateUIView(_ uiView: ChapterUITextView, context: Context) {
        context.coordinator.parent = self
        context.coordinator.sync()
    }

    // MARK: - Coordinator

    @MainActor
    final class Coordinator: NSObject, UITextViewDelegate {
        var parent: ChapterTextView
        weak var textView: ChapterUITextView?

        private var renderedText: String?
        private var renderedFontSize: CGFloat = 0
        private var renderedChapterId: Int?
        private var appliedHighlight: NSRange?
        private var lastScrollToken: Int = .min
        /// UTF-16 offset of each source line's first character.
        private var lineStarts: [Int] = []
        /// A scroll that arrived before the view had a usable size.
        private var pendingScroll: PendingScroll?

        private enum PendingScroll {
            case range(NSRange, animated: Bool)
            case top(animated: Bool)
        }

        init(_ parent: ChapterTextView) {
            self.parent = parent
        }

        /// Reconcile the text view with the latest `parent` values.
        func sync() {
            guard let tv = textView else { return }
            let chapterChanged = renderedChapterId != parent.chapterId
            let textChanged = renderedText != parent.text
                || renderedFontSize != parent.fontSize

            if textChanged {
                tv.textStorage.setAttributedString(
                    Self.makeAttributedText(parent.text, fontSize: parent.fontSize),
                )
                renderedText = parent.text
                renderedFontSize = parent.fontSize
                renderedChapterId = parent.chapterId
                appliedHighlight = nil
                lineStarts = Self.buildLineStarts(parent.text)
                applyHighlight()
                // A font-size change reflows the same chapter — keep the
                // playing sentence in view across it. A chapter change
                // gets an explicit scroll command instead (below).
                if !chapterChanged, let range = highlightRange() {
                    scroll(.range(range, animated: false))
                }
            } else {
                applyHighlight()
            }

            if lastScrollToken != parent.scrollCommand.token {
                lastScrollToken = parent.scrollCommand.token
                performScroll(parent.scrollCommand)
            }
        }

        // MARK: text

        private static func makeAttributedText(
            _ text: String, fontSize: CGFloat,
        ) -> NSAttributedString {
            let paragraph = NSMutableParagraphStyle()
            // Matches the old SwiftUI `.lineSpacing(4)`.
            paragraph.lineSpacing = 4
            let attributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: fontSize),
                .paragraphStyle: paragraph,
                .foregroundColor: UIColor.readerText,
            ]
            return NSAttributedString(string: text, attributes: attributes)
        }

        private static func buildLineStarts(_ text: String) -> [Int] {
            var starts = [0]
            var offset = 0
            for unit in text.utf16 {
                if unit == 0x000A { starts.append(offset + 1) }
                offset += 1
            }
            return starts
        }

        // MARK: highlight

        /// Re-colour the playing sentence. Foreground-colour only, so no
        /// glyph metrics shift — the two-tier palette as before.
        private func applyHighlight() {
            guard let tv = textView else { return }
            let newRange = highlightRange()
            if newRange == appliedHighlight { return }
            let storage = tv.textStorage
            let length = storage.length
            storage.beginEditing()
            if let old = appliedHighlight, NSMaxRange(old) <= length {
                storage.addAttribute(
                    .foregroundColor, value: UIColor.readerText, range: old,
                )
            }
            if let new = newRange, NSMaxRange(new) <= length {
                storage.addAttribute(
                    .foregroundColor, value: UIColor.readerHighlight, range: new,
                )
            }
            storage.endEditing()
            appliedHighlight = newRange
        }

        private func highlightRange() -> NSRange? {
            guard let idx = parent.highlightIndex,
                  idx >= 0, idx < parent.sentences.count else { return nil }
            return sentenceRange(parent.sentences[idx])
        }

        /// Char range of a sentence in the chapter text, clamped to the
        /// stored text. `start_col` / `end_col` are UTF-16 offsets, which
        /// is exactly what `NSRange` indexes.
        private func sentenceRange(_ s: Sentence) -> NSRange? {
            guard let tv = textView else { return nil }
            let length = tv.textStorage.length
            let lo = charOffset(line: s.startLine, col: s.startCol)
            let hi = charOffset(line: s.endLine, col: s.endCol)
            let start = max(0, min(lo, length))
            let end = max(start, min(hi, length))
            return NSRange(location: start, length: end - start)
        }

        private func charOffset(line: Int, col: Int) -> Int {
            guard line >= 1, !lineStarts.isEmpty else { return 0 }
            let idx = min(line - 1, lineStarts.count - 1)
            return lineStarts[idx] + max(0, col)
        }

        /// Index of the sentence whose character range covers `offset`,
        /// or -1 when the tap fell outside every sentence (a header or
        /// inter-sentence gap). Resolution is by character range, not by
        /// source line — several sentences share one paragraph line, so
        /// a line-level match would always collapse to the paragraph's
        /// first sentence.
        private func sentenceIndex(coveringCharOffset offset: Int) -> Int {
            for (i, s) in parent.sentences.enumerated() {
                guard let range = sentenceRange(s) else { continue }
                if range.location <= offset, offset < NSMaxRange(range) {
                    return i
                }
            }
            return -1
        }

        // MARK: scrolling

        private func performScroll(_ command: ChapterScrollCommand) {
            switch command.target {
            case .top:
                scroll(.top(animated: command.animated))
            case .sentence(let idx):
                guard idx >= 0, idx < parent.sentences.count,
                      let range = sentenceRange(parent.sentences[idx]) else { return }
                scroll(.range(range, animated: command.animated))
            }
        }

        private func scroll(_ target: PendingScroll) {
            guard let tv = textView else { return }
            if tv.bounds.height > 0 {
                execute(target, in: tv)
            } else {
                // No real size yet (first appearance) — park it.
                pendingScroll = target
            }
        }

        func flushPendingScrollIfPossible() {
            guard let tv = textView, let pending = pendingScroll,
                  tv.bounds.height > 0 else { return }
            pendingScroll = nil
            execute(pending, in: tv)
        }

        private func execute(_ target: PendingScroll, in tv: ChapterUITextView) {
            let y: CGFloat
            let animated: Bool
            switch target {
            case .top(let a):
                y = 0
                animated = a
            case .range(let range, let a):
                guard let centred = centeredOffsetY(for: range, in: tv) else { return }
                y = centred
                animated = a
            }
            if animated {
                // 0.7 s gentle ease — matches the prose-reading pace the
                // old SwiftUI `.smooth` scroll was tuned to. `.allowUser-
                // Interaction` keeps a finger-drag able to interrupt it
                // (and so exit follow mode via `scrollViewWillBeginDragging`).
                UIView.animate(
                    withDuration: 0.7, delay: 0,
                    options: [.curveEaseInOut, .beginFromCurrentState, .allowUserInteraction],
                ) {
                    tv.contentOffset = CGPoint(x: 0, y: y)
                }
            } else {
                tv.setContentOffset(CGPoint(x: 0, y: y), animated: false)
            }
        }

        /// Content-offset Y that places `range`'s vertical centre at the
        /// viewport centre, clamped to the scrollable range.
        private func centeredOffsetY(
            for range: NSRange, in tv: ChapterUITextView,
        ) -> CGFloat? {
            guard range.length > 0, range.location != NSNotFound else { return nil }
            let layoutManager = tv.layoutManager
            let container = tv.textContainer
            layoutManager.ensureLayout(for: container)
            let glyphRange = layoutManager.glyphRange(
                forCharacterRange: range, actualCharacterRange: nil,
            )
            var rect = layoutManager.boundingRect(
                forGlyphRange: glyphRange, in: container,
            )
            rect.origin.y += tv.textContainerInset.top
            let contentHeight = layoutManager.usedRect(for: container).height
                + tv.textContainerInset.top + tv.textContainerInset.bottom
            let maxY = max(0, contentHeight - tv.bounds.height)
            let target = rect.midY - tv.bounds.height / 2
            return min(max(0, target), maxY)
        }

        // MARK: gestures / delegate

        @objc func handleDoubleTap(_ gesture: UITapGestureRecognizer) {
            guard let tv = textView, gesture.state == .ended else { return }
            let point = gesture.location(in: tv)
            let adjusted = CGPoint(
                x: point.x - tv.textContainerInset.left,
                y: point.y - tv.textContainerInset.top,
            )
            var fraction: CGFloat = 0
            let charIndex = tv.layoutManager.characterIndex(
                for: adjusted, in: tv.textContainer,
                fractionOfDistanceBetweenInsertionPoints: &fraction,
            )
            parent.onSentenceDoubleTap(sentenceIndex(coveringCharOffset: charIndex))
        }

        /// Fires only for a genuine finger drag — programmatic
        /// `setContentOffset` / `UIView.animate` scrolls do not trigger
        /// it — so it cleanly means "user is reading ahead".
        func scrollViewWillBeginDragging(_ scrollView: UIScrollView) {
            parent.onUserScroll()
        }
    }
}

#endif


/// Slim progress strip drawn between the chapter text and the controls
/// bar. Identical implementation to the previous version — kept here so
/// the file is self-contained.
private struct PrefetchProgressBar: View {
    let totalSentences: Int
    let cachedIndices: Set<Int>
    let currentIndex: Int?

    private static let backgroundColor = Color.gray.opacity(0.18)
    private static let cachedOpacity: CGFloat = 0.40

    var body: some View {
        Canvas { ctx, size in
            ctx.fill(
                Path(CGRect(origin: .zero, size: size)),
                with: .color(Self.backgroundColor),
            )
            guard totalSentences > 0 else { return }
            let unit = size.width / CGFloat(totalSentences)
            let segmentWidth = max(unit, 1)
            let cachedColor = Color.accentColor.opacity(Self.cachedOpacity)
            for idx in cachedIndices {
                let x = CGFloat(idx) * unit
                let rect = CGRect(x: x, y: 0, width: segmentWidth, height: size.height)
                ctx.fill(Path(rect), with: .color(cachedColor))
            }
            if let cur = currentIndex, cur >= 0, cur < totalSentences {
                let x = CGFloat(cur) * unit
                let cursorWidth = max(unit, 2.5)
                let rect = CGRect(x: x, y: 0, width: cursorWidth, height: size.height)
                ctx.fill(Path(rect), with: .color(Color.accentColor))
            }
        }
        .frame(height: 4)
        .accessibilityHidden(true)
    }
}


private struct SleepTimerSheet: View {
    let endDate: Date?
    let onSelect: (Int?) -> Void
    @Environment(\.dismiss) private var dismiss

    private static let presets: [Int] = [15, 30, 45, 60]

    var body: some View {
        NavigationStack {
            List {
                if let endDate {
                    Section("当前定时") {
                        HStack {
                            Image(systemName: "moon.fill")
                                .foregroundStyle(Color.accentColor)
                            Text("将在 \(Self.timeFormatter.string(from: endDate)) 暂停")
                            Spacer()
                        }
                        Button(role: .destructive) { onSelect(nil) } label: {
                            Label("取消定时", systemImage: "xmark.circle")
                        }
                    }
                }
                Section("分钟后停止") {
                    ForEach(Self.presets, id: \.self) { m in
                        Button { onSelect(m) } label: {
                            HStack {
                                Text("\(m) 分钟")
                                Spacer()
                                Image(systemName: "chevron.right")
                                    .font(.caption)
                                    .foregroundStyle(.tertiary)
                            }
                            .contentShape(Rectangle())
                        }
                        .foregroundStyle(.primary)
                    }
                }
            }
            .navigationTitle("定时停止")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .topTrailing) {
                    Button("关闭") { dismiss() }
                }
            }
        }
    }

    private static let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .none
        f.timeStyle = .short
        return f
    }()
}


#if canImport(UIKit)
/// Reader-surface palette as `UIColor` (dynamic light/dark) — needed for
/// the `UITextView` attributed text. ``Color`` mirrors these below.
extension UIColor {
    static var readerBackground: UIColor {
        UIColor { traits in
            traits.userInterfaceStyle == .dark
                ? UIColor.systemBackground
                : UIColor(red: 0.969, green: 0.937, blue: 0.851, alpha: 1.0)  // ~#F7EFD9
        }
    }

    static var readerText: UIColor {
        UIColor { traits in
            traits.userInterfaceStyle == .dark
                ? UIColor(red: 0.659, green: 0.635, blue: 0.604, alpha: 1.0)  // ~#A8A29A
                : UIColor(red: 0.420, green: 0.357, blue: 0.278, alpha: 1.0)  // ~#6B5B47
        }
    }

    static var readerHighlight: UIColor {
        UIColor { traits in
            traits.userInterfaceStyle == .dark
                ? UIColor.white
                : UIColor(red: 0.059, green: 0.039, blue: 0.020, alpha: 1.0)  // ~#0F0A05
        }
    }
}
#endif


/// Reader-surface palette for the player page (background + body +
/// highlight). The triple is designed together so the visual hierarchy
/// always points at the currently-playing sentence:
///
///                  highlight  vs  body  contrast ratio
///   light mode:        17:1   vs   6.5:1     ≈ 2.6×
///   dark  mode:        21:1   vs    10:1     ≈ 2.1×
///
/// Body is *deliberately* lower-contrast than iOS's default
/// `Color.primary`. The eye should land on what's playing, not on the
/// wall of prose around it. Dark mode body colour is a warm gray so it
/// shares a "paper" identity with the cream light background.
///
/// Background is reused by the bookshelf too so the App's light theme
/// looks coherent across pages; body / highlight are player-only.
extension Color {
    static var readerBackground: Color {
        #if canImport(UIKit)
        return Color(uiColor: UIColor.readerBackground)
        #else
        return Color(red: 0.969, green: 0.937, blue: 0.851)
        #endif
    }

    /// Body text colour. Light: warm sepia gray (~6.5:1 on cream).
    /// Dark: warm mid-gray (~10:1 on near-black).
    static var readerText: Color {
        #if canImport(UIKit)
        return Color(uiColor: UIColor.readerText)
        #else
        return Color(red: 0.420, green: 0.357, blue: 0.278)
        #endif
    }

    /// Currently-playing-sentence colour. Light: near-black with a
    /// warm undertone (~17:1 on cream). Dark: pure white (~21:1).
    /// Always the single highest-contrast element on the page.
    static var readerHighlight: Color {
        #if canImport(UIKit)
        return Color(uiColor: UIColor.readerHighlight)
        #else
        return Color(red: 0.059, green: 0.039, blue: 0.020)
        #endif
    }
}


struct ChaptersTOCView: View {
    let book: LocalBook
    /// The chapter currently shown in the player. Its row is highlighted
    /// and scrolled into view when the sheet opens.
    let currentChapterId: Int?
    let onSelect: (Int) -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollViewReader { proxy in
                List(book.meta.chapters) { chapter in
                    let isCurrent = chapter.id == currentChapterId
                    Button {
                        onSelect(chapter.id)
                    } label: {
                        HStack {
                            Text("\(chapter.id)")
                                .foregroundStyle(isCurrent ? Color.accentColor : Color.secondary)
                                .frame(width: 40, alignment: .leading)
                            Text(chapter.title)
                                .lineLimit(2)
                                .fontWeight(isCurrent ? .semibold : .regular)
                                .foregroundStyle(isCurrent ? Color.accentColor : Color.primary)
                            Spacer(minLength: 8)
                            if isCurrent {
                                Image(systemName: "headphones")
                                    .font(.caption)
                                    .foregroundStyle(Color.accentColor)
                            }
                        }
                    }
                    .listRowBackground(
                        isCurrent ? Color.accentColor.opacity(0.12) : nil,
                    )
                }
                .navigationTitle("目录")
                .toolbar {
                    ToolbarItem(placement: .topTrailing) {
                        Button("关闭") { dismiss() }
                    }
                }
                .onAppear {
                    guard let current = currentChapterId else { return }
                    // Defer one runloop so the List has laid its rows
                    // out before being asked to scroll.
                    DispatchQueue.main.async {
                        proxy.scrollTo(current, anchor: .center)
                    }
                }
            }
        }
    }
}
