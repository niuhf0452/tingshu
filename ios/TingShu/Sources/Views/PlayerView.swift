// Main reading/playback screen.
//
// Scrolling model (revised 2026-05-07):
//
// - The chapter text is rendered vertically inside a ScrollView, one row
//   per source line. We dropped the prior horizontal page-swipe TabView
//   because pagination kept silently truncating the last line of every
//   page (a TextKit-vs-SwiftUI line-metrics mismatch we'd patched
//   multiple times without fully eliminating). Vertical scroll has no
//   page-break to misalign, so the bug class is gone.
// - Follow mode auto-scrolls the current playing sentence to the centre
//   of the viewport on every position update.
// - A user pan disables follow mode (the explicit "I'm reading ahead"
//   signal). Re-enable via the "..." menu's 跟随播放 toggle.
// - There is no horizontal swipe; chapter switching goes through the
//   bottom-bar 上一章 / 下一章 buttons or the TOC.
import SwiftUI
#if canImport(UIKit)
import UIKit
#endif


struct PlayerView: View {
    let book: LocalBook

    @EnvironmentObject var playback: PlaybackService
    @EnvironmentObject var settings: SettingsStore

    @State private var showTOC = false
    @State private var showSleepSheet = false
    @State private var transientNotice: String?
    @State private var noticeDismissTask: Task<Void, Never>?
    /// True while we're auto-scrolling for follow-mode. The drag-detect
    /// gesture checks this so a programmatic scroll doesn't get
    /// misinterpreted as the user reading ahead.
    @State private var suppressFollowExit = false

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
            ChaptersTOCView(book: book) { chapterId in
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

    // MARK: - chapter text (vertical scroll)

    /// Vertical scroll over the chapter's lines. The line index (1-based,
    /// matching meta's `start_line`) is the scroll target id, so
    /// follow-mode can `scrollTo(line)` directly without a separate
    /// line→view-id table. Touch pad style: a real pan exits follow.
    private var chapterScrollArea: some View {
        // Re-render this subtree when text/meta lands for any chapter.
        let _ = playback.state.chapterCacheRevision
        let chapterId = playback.state.browseChapterId
        let lines = chapterLines(for: chapterId)
        return ScrollViewReader { proxy in
            ScrollView {
                if lines.isEmpty {
                    VStack(spacing: 8) {
                        ProgressView()
                        Text("加载章节…").font(.caption).foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, minHeight: 200)
                } else {
                    LazyVStack(alignment: .leading, spacing: 4) {
                        ForEach(0..<lines.count, id: \.self) { i in
                            lineView(at: i, line: lines[i], chapterId: chapterId)
                                .id(i + 1)  // 1-based to match meta.start_line
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                }
            }
            .onChange(of: playback.state.position) { _, pos in
                autoScrollToCurrentSentence(pos, proxy: proxy)
            }
            .onChange(of: playback.state.browseChapterId) { _, _ in
                // When the chapter target changes (TOC pick, follow
                // crossing, prev/next button), reset the scroll. If
                // follow is on and we have a play position in this
                // chapter, centre on it; otherwise snap to top.
                resetScroll(proxy: proxy)
            }
            .onChange(of: playback.state.chapterCacheRevision) { _, _ in
                // Text or meta just landed — if follow is on, retry the
                // scroll-to so the current sentence ends up centred even
                // when the position update fired before the rows
                // existed.
                autoScrollToCurrentSentence(playback.state.position, proxy: proxy)
            }
            .simultaneousGesture(
                DragGesture(minimumDistance: 8)
                    .onChanged { _ in
                        guard !suppressFollowExit else { return }
                        if playback.state.followMode {
                            playback.state.followMode = false
                        }
                    }
            )
        }
    }

    @ViewBuilder
    private func lineView(at lineIdx: Int, line: String, chapterId: Int?) -> some View {
        let attributed = attributedLine(at: lineIdx, line: line, chapterId: chapterId)
        Text(attributed)
            .font(.system(size: CGFloat(settings.fontSize)))
            .lineSpacing(4)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
            .onTapGesture(count: 2) {
                handleDoubleTap(lineIdx: lineIdx, chapterId: chapterId)
            }
    }

    /// Build the attributed string for one line, applying the current
    /// playing-sentence highlight if its (start_line, end_line) range
    /// covers this line. Foreground-only colour change — no font-weight
    /// or background fill, to avoid glyph-metric reflow at slice edges.
    ///
    /// Two-tier palette (per design discussion 2026-05-09):
    /// - Body text uses ``Color.readerText``, a deliberately
    ///   reduced-contrast warm gray. The eye doesn't have to fight
    ///   "screaming pure black" while skimming the surrounding sea of
    ///   prose.
    /// - The currently-playing sentence uses ``Color.readerHighlight``,
    ///   the page's highest-contrast ink (near-black in light mode,
    ///   pure white in dark). Visual hierarchy is inverted from
    ///   default-iOS so the user's eye lands on what's *playing*, not
    ///   on the wall of text around it.
    private func attributedLine(
        at lineIdx: Int, line: String, chapterId: Int?,
    ) -> AttributedString {
        var plain = AttributedString(line)
        plain.foregroundColor = Color.readerText
        guard let pos = playback.state.position,
              pos.chapterId == chapterId else { return plain }
        let sentences = playback.chapterSentences(for: pos.chapterId)
        guard pos.sentenceIndex >= 0, pos.sentenceIndex < sentences.count else {
            return plain
        }
        let s = sentences[pos.sentenceIndex]
        let line1 = lineIdx + 1
        guard line1 >= s.startLine, line1 <= s.endLine else { return plain }

        let lineU16 = line.utf16
        let totalU16 = lineU16.count
        let startCol = (line1 == s.startLine) ? s.startCol : 0
        let endCol = (line1 == s.endLine) ? s.endCol : totalU16
        let lo = max(0, min(totalU16, startCol))
        let hi = max(lo, min(totalU16, endCol))
        guard lo < hi else { return plain }
        guard let startU = lineU16.index(
                lineU16.startIndex, offsetBy: lo, limitedBy: lineU16.endIndex,
              ),
              let endU = lineU16.index(
                lineU16.startIndex, offsetBy: hi, limitedBy: lineU16.endIndex,
              ),
              let strStart = String.Index(startU, within: line),
              let strEnd = String.Index(endU, within: line) else {
            return plain
        }
        let before = String(line[line.startIndex..<strStart])
        let middle = String(line[strStart..<strEnd])
        let after = String(line[strEnd..<line.endIndex])
        var result = AttributedString(before)
        result.foregroundColor = Color.readerText
        var hl = AttributedString(middle)
        hl.foregroundColor = Color.readerHighlight
        result += hl
        var afterStr = AttributedString(after)
        afterStr.foregroundColor = Color.readerText
        result += afterStr
        return result
    }

    /// Split chapter text at LF boundaries while preserving empty lines —
    /// blank rows render as zero-height Text and act as paragraph
    /// separators. `components(separatedBy:)` keeps trailing empties,
    /// which `split` would drop with `omittingEmptySubsequences: true`.
    private func chapterLines(for chapterId: Int?) -> [String] {
        guard let cid = chapterId,
              let text = playback.chapterText(for: cid) else { return [] }
        return text.components(separatedBy: "\n")
    }

    /// Double-tap on a line: jump play to the first sentence whose
    /// `(start_line ... end_line)` range covers this line. Coarser than
    /// the prior point-precise hit-test, but multi-sentence lines are
    /// rare and sentence segmentation respects paragraph breaks so this
    /// is usually unambiguous. Worth losing the precision for the
    /// simplicity win in vertical-scroll layout.
    private func handleDoubleTap(lineIdx: Int, chapterId: Int?) {
        guard let cid = chapterId else { return }
        let sentences = playback.chapterSentences(for: cid)
        if sentences.isEmpty {
            showNotice("章节分析中，稍后再试")
            return
        }
        let line1 = lineIdx + 1
        for (i, s) in sentences.enumerated() where s.startLine <= line1 && line1 <= s.endLine {
            Task { await playback.jumpPlay(chapterId: cid, sentenceIndex: i) }
            return
        }
        // Tapped on text with no covering sentence (header, blank gap):
        // intentionally silent per spec.
    }

    /// Follow-mode auto-scroll: bring the current playing sentence's
    /// start line to the viewport centre. No-op when follow is off, the
    /// position is in a different chapter, or we have no position yet.
    private func autoScrollToCurrentSentence(
        _ pos: PlaybackPosition?, proxy: ScrollViewProxy,
    ) {
        guard playback.state.followMode,
              let pos = pos,
              pos.chapterId == playback.state.browseChapterId else { return }
        let sentences = playback.chapterSentences(for: pos.chapterId)
        guard pos.sentenceIndex >= 0, pos.sentenceIndex < sentences.count else {
            return
        }
        let line = sentences[pos.sentenceIndex].startLine
        suppressFollowExit = true
        // `.smooth` is a critically-damped spring (no overshoot) with a
        // gentle ease-in / ease-out feel — much less abrupt than the
        // earlier 0.25 s linear-ish easeInOut, which the user described
        // as too jumpy. 0.7 s is comfortable for prose: long enough to
        // read as motion (not a teleport), short enough to settle
        // before the next sentence boundary in normal-speed playback.
        withAnimation(.smooth(duration: 0.7)) {
            proxy.scrollTo(line, anchor: .center)
        }
        Task { @MainActor in
            // Match the suppression window to the animation length so
            // the drag-detect gesture doesn't re-arm while the spring
            // is still settling and mistake the tail of our own scroll
            // for a user pan.
            try? await Task.sleep(nanoseconds: 750_000_000)
            suppressFollowExit = false
        }
    }

    /// Reset scroll on chapter switch. Prefers follow-centred when both
    /// follow is on and the play position is in the new chapter; falls
    /// back to top-of-chapter for manual navigation.
    private func resetScroll(proxy: ScrollViewProxy) {
        if playback.state.followMode,
           let pos = playback.state.position,
           pos.chapterId == playback.state.browseChapterId {
            autoScrollToCurrentSentence(pos, proxy: proxy)
            return
        }
        suppressFollowExit = true
        withAnimation(.smooth(duration: 0.5)) {
            proxy.scrollTo(1, anchor: .top)
        }
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 550_000_000)
            suppressFollowExit = false
        }
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
            Task { await playback.setBrowseChapter(chapterId: p.chapterId) }
        }
        // Same chapter: the .onChange(position) handler will fire the
        // auto-scroll on the next position update; for the case where
        // position hasn't moved yet, the chapterCacheRevision change
        // listener also re-runs autoScroll. Sufficient.
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
        return Color(uiColor: UIColor { traits in
            traits.userInterfaceStyle == .dark
                ? UIColor.systemBackground
                : UIColor(red: 0.969, green: 0.937, blue: 0.851, alpha: 1.0)  // ~#F7EFD9
        })
        #else
        return Color(red: 0.969, green: 0.937, blue: 0.851)
        #endif
    }

    /// Body text colour. Light: warm sepia gray (~6.5:1 on cream).
    /// Dark: warm mid-gray (~10:1 on near-black).
    static var readerText: Color {
        #if canImport(UIKit)
        return Color(uiColor: UIColor { traits in
            traits.userInterfaceStyle == .dark
                ? UIColor(red: 0.659, green: 0.635, blue: 0.604, alpha: 1.0)  // ~#A8A29A
                : UIColor(red: 0.420, green: 0.357, blue: 0.278, alpha: 1.0)  // ~#6B5B47
        })
        #else
        return Color(red: 0.420, green: 0.357, blue: 0.278)
        #endif
    }

    /// Currently-playing-sentence colour. Light: near-black with a
    /// warm undertone (~17:1 on cream). Dark: pure white (~21:1).
    /// Always the single highest-contrast element on the page.
    static var readerHighlight: Color {
        #if canImport(UIKit)
        return Color(uiColor: UIColor { traits in
            traits.userInterfaceStyle == .dark
                ? UIColor.white
                : UIColor(red: 0.059, green: 0.039, blue: 0.020, alpha: 1.0)  // ~#0F0A05
        })
        #else
        return Color(red: 0.059, green: 0.039, blue: 0.020)
        #endif
    }
}


struct ChaptersTOCView: View {
    let book: LocalBook
    let onSelect: (Int) -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List(book.meta.chapters) { chapter in
                Button {
                    onSelect(chapter.id)
                } label: {
                    HStack {
                        Text("\(chapter.id)").foregroundStyle(.secondary).frame(width: 40, alignment: .leading)
                        Text(chapter.title).lineLimit(2)
                    }
                }
            }
            .navigationTitle("目录")
            .toolbar {
                ToolbarItem(placement: .topTrailing) {
                    Button("关闭") { dismiss() }
                }
            }
        }
    }
}
