// Audiobook playback engine (see docs/technical-plan.md §3.9, §3.6.2).
//
// Two-thread model:
//
//   Prefetch thread (single serial worker, `prefetchTask`):
//     - Loops over the priority-ordered window (anchor → anchor+1 → ...
//       spilling across chapter boundaries).
//     - One TTS HTTP request in flight at a time. Server-side TTS is
//       serial (gpu_guard), so client-side parallelism gains nothing
//       and worse, the anchor's audio (the one the player needs RIGHT
//       NOW) might not finish first.
//     - `reconcileWindow()` is called on every `position` change + meta
//       arrival; it (re)starts the worker if there's pending work.
//
//   Playback thread (single Task, `playerTask`):
//     - Loop: ensure meta for anchor.chapterId → reconcileWindow →
//       await awaitPrefetched(anchor) → play via AVAudioEngine →
//       advance `state.position` → next iteration.
//
//   Cache:
//     - TTS audio on disk via `TTSCache`, keyed by text+speaker+tone+speed
//       (SHA-1). Survives beyond the window so small jumps hit cache
//       rather than re-synth (§3.6.1).
//     - Chapter meta in-memory `metaByChapter`. Small (KB per chapter).
//
// Audio graph (§3.8): playerNode → eq(globalGain) → mainMixer → out.
import AVFoundation
import Combine
import Foundation
#if canImport(MediaPlayer)
import MediaPlayer
#endif


/// Player position. Carries both the runtime address (`sentenceIndex` —
/// the natural key for the prefetch window and the highlight) and the
/// canonical durable address (`startLine` / `startCol`) per
/// docs/technical-plan.md §3.4. Persistence layer reads the latter; the
/// player loop reads the former.
///
/// Line / col are 1-based-ish (server convention: lines start at 1) when
/// meta is available; both are 0 as a sentinel before meta has loaded for
/// the chapter. Restorers should treat (0, 0) as "start of chapter".
struct PlaybackPosition: Equatable, Sendable {
    let chapterId: Int
    let sentenceIndex: Int
    let startLine: Int
    let startCol: Int

    init(chapterId: Int, sentenceIndex: Int, startLine: Int = 0, startCol: Int = 0) {
        self.chapterId = chapterId
        self.sentenceIndex = sentenceIndex
        self.startLine = startLine
        self.startCol = startCol
    }
}

/// Durable position handed back/forth across persistence boundaries
/// (SwiftData, restore on book open). No `sentenceIndex` — that's a
/// runtime artefact of the current meta and may change if the chapter
/// is re-analysed; `(chapterId, startLine, startCol)` is stable.
struct DurablePosition: Equatable, Sendable {
    let chapterId: Int
    let startLine: Int
    let startCol: Int
}

/// Addressable sentence slot: (chapter, index). The network thread
/// uses this as the key into the prefetch window.
private struct SentenceAddress: Hashable, Sendable {
    let chapterId: Int
    let sentenceIndex: Int
}


/// View-visible playback state. Lives on the main actor so SwiftUI
/// updates are free of thread hops.
///
/// Split per §3.4 — **browse position** (what the user is looking at)
/// and **play position** (where TTS is reading) are independent. The
/// player loop drives `position`; the View drives `browseChapterId` via
/// swipe / TOC. They sync only when ``followMode`` is on.
///
/// Browse-side fields (`browseChapterId` + `browseChapter*`) are
/// populated synchronously from local disk on every browse switch, never
/// wait on network — see §3.9.
@MainActor
final class PlaybackState: ObservableObject {
    @Published var isPlaying = false
    /// Where TTS playback currently is. May refer to a different chapter
    /// than ``browseChapterId`` (user reading ahead while play continues).
    @Published var position: PlaybackPosition?
    /// Whether the browse view auto-follows the play position. Off when
    /// the user manually scrolls / swipes / picks from TOC; restored by
    /// the bottom-bar follow button.
    @Published var followMode = true
    /// Chapter shown in the View. Independent of `position.chapterId`
    /// once the user starts browsing ahead.
    @Published var browseChapterId: Int?
    @Published var browseChapterText: String = ""
    @Published var browseChapterSentences: [Sentence] = []
    /// Lit while sentence metadata is being fetched for the **browse**
    /// chapter (so the chapter text view can render sentence-list mode
    /// once ready). Never gates any visible element on its own.
    @Published var fetchingBrowseSentences = false
    @Published var error: String?
    /// Wall-clock time at which the sleep timer will pause playback,
    /// nil when no timer is set. Surfaced for the player UI's countdown.
    @Published var sleepTimerEndDate: Date?
    /// Bumped whenever the per-chapter caches (text or meta) for ANY
    /// chapter mutate. Lets the View layer's paged TabView re-evaluate
    /// adjacent pages when their text / sentences land — without making
    /// the whole cache @Published (which would mean every page re-renders
    /// for every page's load, n²).
    @Published var chapterCacheRevision: Int = 0
    /// Sentence indices in the **browse** chapter whose TTS audio is
    /// cached on disk. Drives the prefetch-progress indicator. Repopulated
    /// on every browse chapter switch + incrementally updated as new
    /// TTS fetches complete. May be slightly stale wrt cache eviction
    /// (eviction happens transparently in `TTSCache`); the progress bar
    /// is allowed to be optimistic.
    @Published var cachedSentencesInBrowseChapter: Set<Int> = []
    /// Which page within the browse chapter the user is on. Driven by the
    /// horizontal page-swipe TabView in PlayerView. Resets to 0 on
    /// chapter switch (`loadBrowseChapter`); follow mode updates it to
    /// the page containing the current play sentence.
    @Published var browsePageIndex: Int = 0
}


@MainActor
final class PlaybackService: ObservableObject {
    let state = PlaybackState()

    /// Read-only access for the Settings pages (e.g. the player-
    /// settings character list calls ``api.bookCharacters``). Internal
    /// playback path still uses this same instance — no separate
    /// client.
    let api: APIClient
    private let store: BookStore
    /// Read-only access for the Settings page (display + clear). Internal
    /// playback path still uses this same instance — no separate cache.
    let cache: TTSCache
    private let settings: SettingsStore
    private let progressStore: ProgressStore?

    private var book: LocalBook?

    // Audio graph — reused across sentences.
    // playerNode → timePitch → eq (gain) → limiter (peak shave) → mainMixer → out.
    private let engine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    /// Time-pitch unit applies playback rate without changing pitch
    /// (phase-vocoder DSP). Since the server now always synthesizes at
    /// 1.0x and a single audio file serves all speeds, all rate control
    /// happens here. ``AVAudioUnitVarispeed`` would be cheaper but
    /// shifts pitch — sounds like a chipmunk at high rates and is
    /// inappropriate for spoken-word audiobooks.
    private let timePitch = AVAudioUnitTimePitch()
    private let eq = AVAudioUnitEQ(numberOfBands: 0)
    /// AUDynamicsProcessor configured as a brick-wall limiter so a +20 dB
    /// user gain can't push peaks past the spec ceiling (-0.3 dBFS, §3.8).
    /// AUPeakLimiter would also work but doesn't expose a threshold knob —
    /// AUDynamicsProcessor lets us hit the exact dBFS the plan calls for.
    private let limiter: AVAudioUnitEffect = {
        let desc = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: kAudioUnitSubType_DynamicsProcessor,
            componentManufacturer: kAudioUnitManufacturer_Apple,
            componentFlags: 0,
            componentFlagsMask: 0,
        )
        return AVAudioUnitEffect(audioComponentDescription: desc)
    }()
    /// Subscription that re-applies ``settings.playbackSpeed`` to the
    /// time-pitch unit whenever the user changes it.
    private var speedChangeSink: AnyCancellable?

    // Sliding-window parameters. Window covers
    // `[position, position + windowAhead]` — so windowAhead+1 active slots.
    // 20 ahead = 21-sentence buffer; deep enough to span chapter
    // boundaries comfortably (most chapters are >20 sentences, so the
    // window typically stays within one chapter and a bit of the next).
    private static let windowAhead = 20

    // --- playback thread ---
    private var playerTask: Task<Void, Never>?

    // --- sleep timer ---
    private var sleepTimerTask: Task<Void, Never>?

    // --- network thread state ---
    // Single-worker serial prefetch design (rationale in technical-plan
    // §3.6.2): instead of spawning N parallel tasks (one per window
    // slot) and racing for server-side TTS bandwidth, exactly one
    // ``runPrefetchWorker`` Task fetches sentences in priority order
    // (anchor → anchor+1 → ...). Server TTS is serial anyway
    // (`gpu_guard` in Qwen3-TTS), so parallelism on the client gains
    // nothing — and worse, the order in which the server picks up the
    // 21 pending HTTP requests is unpredictable, so the anchor's audio
    // (the one the player needs RIGHT NOW) might not finish first.
    /// Already-fetched URLs for window slots. Resolved Result so failures
    /// don't get re-attempted on every reconcile.
    private var prefetchedURLs: [SentenceAddress: Result<URL, Error>] = [:]
    /// Single serial prefetch worker. Started lazily by
    /// ``startPrefetchWorkerIfIdle()``; loops while there are missing
    /// slots in the current window.
    private var prefetchTask: Task<Void, Never>?
    /// Player loop awaits one of these per slot it needs. The worker
    /// resumes them as fetches complete.
    private var prefetchSubscribers: [SentenceAddress: [CheckedContinuation<URL, Error>]] = [:]
    /// Chapter-meta cache by chapter id. Each entry is the full
    /// ``ChapterMeta`` (sentences + per-chapter character snapshot).
    /// Grows monotonically during a session so backward jumps hit memory.
    private var metaByChapter: [Int: ChapterMeta] = [:]
    /// In-flight chapter-meta fetches.
    private var metaInFlight: [Int: Task<Void, Never>] = [:]
    /// Chapters whose last meta fetch returned an error. Used to
    /// distinguish "meta not loaded yet" from "meta load failed" in the
    /// player loop — without this, a failed fetch returns [] from
    /// ``ensureMeta`` and the loop silently skips to the next chapter.
    private var metaFetchFailed: Set<Int> = []
    /// Per-chapter raw text, cached for the lifetime of this book open.
    /// Holds text for both the browse chapter (UI display) and the play
    /// chapter (`extractText` consults it during TTS prefetch). Without
    /// this cache, prefetch breaks the moment the user browses to a
    /// different chapter than the one playing.
    private var chapterTextByChapter: [Int: String] = [:]

    /// Re-emits `state`'s `objectWillChange` on our own publisher.
    /// SwiftUI's `@EnvironmentObject` subscribes to THIS service's
    /// `objectWillChange`; without this forwarding, mutations to nested
    /// `state.@Published` fields would never trigger view updates.
    private var stateForwarding: AnyCancellable?
    /// Subscription that debounce-writes ``state.position`` to SwiftData.
    private var positionPersistSink: AnyCancellable?
    /// AVAudioSession route-change observer — pauses playback when the
    /// active output device disappears (headphone unplug, BT disconnect)
    /// so the lock-screen play/pause icon flips to "play" instead of
    /// staying stuck on "pause" while audio is silent.
    private var routeChangeObserver: NSObjectProtocol?
    /// AVAudioSession interruption observer — pauses playback when the
    /// system interrupts us (phone call, Siri, the rare alert that
    /// slips past ``setPrefersNoInterruptionsFromSystemAlerts``). iOS
    /// stops audio output behind our back; without this, ``state.isPlaying``
    /// stays true and the UI / lock-screen toggle keeps showing pause
    /// over silent output.
    private var interruptionObserver: NSObjectProtocol?

    init(
        api: APIClient,
        store: BookStore,
        cache: TTSCache,
        settings: SettingsStore,
        progressStore: ProgressStore? = nil,
    ) {
        self.api = api
        self.store = store
        self.cache = cache
        self.settings = settings
        self.progressStore = progressStore
        setupAudioGraph()
        // Forward state changes. PlaybackState is @MainActor, so its
        // objectWillChange fires on MainActor; the sink closure runs
        // on the same actor and can call our objectWillChange.send()
        // directly. SwiftUI's view body is re-evaluated on the next
        // frame, which is when it reads the updated state fields.
        stateForwarding = state.objectWillChange.sink { [weak self] _ in
            self?.objectWillChange.send()
        }
        // Persist play position whenever it changes, debounced so a fast
        // sentence cadence doesn't hammer SwiftData. 2 s is short enough
        // that ^C / app-kill loses at most one sentence; long enough that
        // a normal play loop writes once per ~5 sentences.
        positionPersistSink = state.$position
            .removeDuplicates()
            .debounce(for: .seconds(2), scheduler: DispatchQueue.main)
            .sink { [weak self] _ in self?.persistProgress() }
        // React to user-driven speed changes — apply to the running
        // audio graph immediately so currently-scheduled audio shifts
        // rate within ~50ms (AVAudioUnitTimePitch parameter changes
        // take effect on the next render cycle).
        speedChangeSink = settings.$playbackSpeed
            .removeDuplicates()
            .sink { [weak self] _ in self?.applyPlaybackSpeed() }
        setupRemoteCommands()
        setupRouteChangeObserver()
        setupInterruptionObserver()
    }

    deinit {
        if let token = routeChangeObserver {
            NotificationCenter.default.removeObserver(token)
        }
        if let token = interruptionObserver {
            NotificationCenter.default.removeObserver(token)
        }
    }

    // MARK: - public API

    /// Open a book for playback. Loads the target chapter (local read only)
    /// and sets position but does NOT start network or audio work — those
    /// kick in when the user presses play.
    ///
    /// When ``saved`` is provided, awaits chapter meta and resolves
    /// `(startLine, startCol)` back to a sentence index per §3.4
    /// ("找不到则取最接近的句子"). When nil, falls back to whatever the
    /// progress store has for this book; if it has nothing, starts at
    /// chapter 1. Browse view starts on the play chapter with follow
    /// mode on.
    func open(book: LocalBook, saved: DurablePosition? = nil) async {
        stop()
        self.book = book
        state.followMode = true
        // Pre-allocate audio HAL so the first play-button tap doesn't
        // hitch on engine start.
        warmAudioEngineIfNeeded()
        let resolvedSaved = saved ?? progressStore?.load(bookId: book.bookId)
        let chapterId = resolvedSaved?.chapterId ?? 1
        await loadBrowseChapter(chapterId: chapterId)
        guard state.browseChapterId == chapterId else { return }

        guard let saved = resolvedSaved else {
            updatePosition(makePosition(chapterId: chapterId, sentenceIndex: 0))
            return
        }

        // Resolve the durable (line, col) back to a sentence index. Block
        // on meta — restore-time accuracy matters more than first-paint
        // latency, and the chapter text is already on screen.
        let sentences = await ensureMeta(chapterId: chapterId)
        guard state.browseChapterId == chapterId else { return }
        let idx = Self.resolveSentenceIndex(
            startLine: saved.startLine, startCol: saved.startCol,
            in: sentences,
        )
        updatePosition(makePosition(chapterId: chapterId, sentenceIndex: idx))
    }

    /// Write the current position to the progress store, if both are
    /// known. Called on the 2 s debounce + on stop().
    private func persistProgress() {
        guard let book = book,
              let durable = currentDurablePosition(),
              let progressStore = progressStore else { return }
        progressStore.save(bookId: book.bookId, position: durable)
    }

    /// Start playback at the current position.
    func play() {
        guard book != nil else { return }
        guard !state.isPlaying else { return }
        state.isPlaying = true
        playerTask = Task { [weak self] in await self?.playerLoop() }
        updateNowPlaying()
    }

    /// Pause immediately — do not wait for the current sentence to
    /// finish. Matches §3.5 "点击暂停按钮后立即停止播放".
    func pause() {
        state.isPlaying = false
        playerNode.stop()
        playerTask?.cancel()
        playerTask = nil
        // Pause cancels any pending sleep timer — once playback's off,
        // the timer no longer has anything to do.
        cancelSleepTimer()
        updateNowPlaying()
    }

    // MARK: - sleep timer (§3.2 settings page item, §3.5 pause semantics)

    /// Schedule playback to pause after `minutes` from now. Replaces any
    /// existing timer. Pass nil to cancel without scheduling a new one.
    /// No-op when `minutes` is non-positive (treated as cancel).
    func setSleepTimer(minutes: Int?) {
        cancelSleepTimer()
        guard let minutes = minutes, minutes > 0 else { return }
        let seconds = TimeInterval(minutes * 60)
        let endDate = Date().addingTimeInterval(seconds)
        state.sleepTimerEndDate = endDate
        sleepTimerTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self = self else { return }
                // Only act if we're still the live timer — pause()/stop()
                // would have cleared sleepTimerTask + endDate already.
                guard self.state.sleepTimerEndDate == endDate else { return }
                self.pause()
            }
        }
    }

    private func cancelSleepTimer() {
        sleepTimerTask?.cancel()
        sleepTimerTask = nil
        state.sleepTimerEndDate = nil
    }

    /// Hard stop on view exit: pause + drop position + cancel every
    /// network task + tear down audio. Nothing must keep running after
    /// the user leaves the player (see §3.9).
    func stop() {
        // Flush the latest position before clearing it — the debounced
        // sink may not have fired yet, and the user expects "I closed
        // the player at sentence X" to be where they resume.
        persistProgress()
        pause()
        state.position = nil
        cancelAllPrefetch()
        cancelAllMetaFetches()
        if engine.isRunning { engine.stop() }
        // Clear cache anchor so eviction doesn't bias toward this
        // session's last position once the player is gone.
        Task { await cache.setAnchor(nil) }
        clearNowPlaying()
    }

    /// Move the **play** position to `(chapterId, sentenceIndex)`. Used by
    /// double-tap on a sentence and by lock-screen next/prev. The browse
    /// view follows (this is an explicit "play here" gesture — assumes
    /// follow mode), so the user lands looking at what's playing.
    /// Resumes play if it was already playing.
    func jumpPlay(chapterId: Int, sentenceIndex: Int) async {
        let wasPlaying = state.isPlaying
        pause()
        state.followMode = true
        if chapterId != state.browseChapterId {
            await loadBrowseChapter(chapterId: chapterId)
        }
        updatePosition(makePosition(chapterId: chapterId, sentenceIndex: sentenceIndex))
        reconcileWindow()
        if wasPlaying { play() }
    }

    /// Switch the **browse** view to a different chapter. Does NOT touch
    /// the play position or follow mode — the caller decides whether
    /// this counts as "user looking away" (swipe / TOC → followMode=false)
    /// or "snap back to play" (follow button → followMode=true).
    ///
    /// `pageIndex` lets the caller open the chapter on a specific page
    /// (used by continuous page-swipe: swiping past the last page of
    /// chapter N lands on page 0 of chapter N+1; swiping back from page 0
    /// of chapter N+1 lands on the last page of chapter N — the View
    /// passes the desired index via this parameter).
    func setBrowseChapter(chapterId: Int, pageIndex: Int = 0) async {
        if state.browseChapterId == chapterId {
            if state.browsePageIndex != pageIndex {
                state.browsePageIndex = pageIndex
            }
            return
        }
        await loadBrowseChapter(chapterId: chapterId, pageIndex: pageIndex)
    }

    /// Set just the page index within the current browse chapter. No-op
    /// if the chapter doesn't match — caller should use ``setBrowseChapter``
    /// in that case.
    func setBrowsePage(_ pageIndex: Int) {
        if state.browsePageIndex != pageIndex {
            state.browsePageIndex = pageIndex
        }
    }

    // MARK: - per-chapter cache accessors (View layer)

    /// Cached chapter text, or nil if not yet loaded. Synchronous; the
    /// View calls this during its body to render adjacent pages without
    /// triggering a fetch. ``ensureChapterText`` warms the cache.
    func chapterText(for chapterId: Int) -> String? {
        chapterTextByChapter[chapterId]
    }

    /// Cached sentence list for ``chapterId``. Empty when meta hasn't
    /// loaded yet — the View should fall back to plain text rendering.
    func chapterSentences(for chapterId: Int) -> [Sentence] {
        metaByChapter[chapterId]?.sentences ?? []
    }

    /// Per-chapter character snapshot from the ChapterMeta. Empty when
    /// meta hasn't loaded — the View just won't be able to highlight
    /// per-character voices, which is fine for render-only.
    func chapterCharacters(for chapterId: Int) -> [Character] {
        metaByChapter[chapterId]?.characters ?? []
    }

    /// Warm the text + meta caches for ``chapterId`` so a paged View
    /// can render the chapter without flickering from blank to text.
    /// Non-blocking, idempotent. Bumps `chapterCacheRevision` when the
    /// text lands so SwiftUI re-evaluates pages observing it.
    func ensureChapterText(for chapterId: Int) {
        guard let book = book else { return }
        if chapterTextByChapter[chapterId] != nil { return }
        guard let chapter = book.meta.chapters.first(where: { $0.id == chapterId }) else {
            return
        }
        Task { [weak self] in
            guard let self = self else { return }
            let text = await self.store.chapterText(
                bookId: book.bookId, chapter: chapter,
            ) ?? ""
            // Re-check under the actor — a concurrent load may have
            // landed first.
            if self.chapterTextByChapter[chapterId] == nil {
                self.chapterTextByChapter[chapterId] = text
                self.state.chapterCacheRevision &+= 1
            }
        }
        // Meta usually arrives via the player loop / browse load. Kick
        // it off here too so paged adjacent pages get sentence rendering
        // before the user actually swipes onto them.
        if metaByChapter[chapterId] == nil {
            kickOffMetaFetch(chapterId: chapterId)
        }
    }

    /// Warm `(chapterId − 1)` and `(chapterId + 1)` so the paging
    /// container has both neighbours visible on the first swipe.
    func prefetchAdjacentChapters(around chapterId: Int) {
        guard let book = book else { return }
        for candidate in [chapterId - 1, chapterId + 1] {
            guard book.meta.chapters.contains(where: { $0.id == candidate }) else { continue }
            ensureChapterText(for: candidate)
        }
    }

    /// Update the play position and run the auto-follow side effects.
    /// All ``state.position`` writes go through this helper so we never
    /// forget to sync the browse view when followMode is on, or to push
    /// the new chapter to the lock screen.
    private func updatePosition(_ pos: PlaybackPosition?) {
        let oldChapterId = state.position?.chapterId
        state.position = pos
        syncBrowseToPlay()
        if oldChapterId != pos?.chapterId {
            updateNowPlaying()
        }
    }

    /// When follow mode is on and the play position has crossed into a
    /// different chapter than the browse view, switch browse to follow.
    /// Background — uses `Task` because `loadBrowseChapter` is async; the
    /// current chapter's text loads quickly off disk. While it loads,
    /// the View briefly shows the previous chapter, then snaps over.
    private func syncBrowseToPlay() {
        guard state.followMode,
              let pos = state.position,
              state.browseChapterId != pos.chapterId else { return }
        Task { await loadBrowseChapter(chapterId: pos.chapterId) }
    }

    /// Build a ``PlaybackPosition`` filling in `(startLine, startCol)`
    /// from the cached chapter meta when available. Falls back to the
    /// (0, 0) sentinel if meta isn't loaded or the index is out of range.
    private func makePosition(chapterId: Int, sentenceIndex: Int) -> PlaybackPosition {
        guard let sentences = metaByChapter[chapterId]?.sentences,
              sentenceIndex >= 0, sentenceIndex < sentences.count else {
            return PlaybackPosition(chapterId: chapterId, sentenceIndex: sentenceIndex)
        }
        let s = sentences[sentenceIndex]
        return PlaybackPosition(
            chapterId: chapterId, sentenceIndex: sentenceIndex,
            startLine: s.startLine, startCol: s.startCol,
        )
    }

    /// Resolve `(startLine, startCol)` back to the closest sentence
    /// index in `sentences`. Returns 0 for empty input or `(0, 0)`
    /// sentinels (treat as start of chapter).
    static func resolveSentenceIndex(
        startLine: Int, startCol: Int, in sentences: [Sentence],
    ) -> Int {
        if sentences.isEmpty { return 0 }
        if startLine <= 0 { return 0 }
        // Compose a comparable position; line dominates so a sentence on
        // the same line as the saved one is always closer than one on a
        // different line, regardless of column drift.
        let target = startLine * 1_000_000 + startCol
        var bestIdx = 0
        var bestDist = Int.max
        for (i, s) in sentences.enumerated() {
            let p = s.startLine * 1_000_000 + s.startCol
            let d = abs(p - target)
            if d < bestDist {
                bestDist = d
                bestIdx = i
                if d == 0 { break }
            }
        }
        return bestIdx
    }

    /// Snapshot the current position in its persistence-stable form.
    /// Returns nil when there's no current position or when meta hasn't
    /// loaded yet (line/col still at the (0, 0) sentinel — writing that
    /// would just blat over a previously good record). The chapter id is
    /// always trustworthy, so an alternative is to persist `(chapter, 0, 0)`
    /// — but the loop fills line/col within milliseconds of pressing play,
    /// so we'd rather wait for a real position.
    func currentDurablePosition() -> DurablePosition? {
        guard let pos = state.position else { return nil }
        if pos.startLine == 0 && pos.startCol == 0 { return nil }
        return DurablePosition(
            chapterId: pos.chapterId,
            startLine: pos.startLine,
            startCol: pos.startCol,
        )
    }

    // MARK: - browse-side: load chapter for view (no network gating)

    /// Load chapter text + meta for the **browse** view. UI-side only —
    /// does NOT cancel prefetch (playback may be in a different chapter)
    /// and does NOT touch the play position. Reads text off disk into the
    /// per-chapter cache so subsequent visits don't re-hit the disk;
    /// kicks off meta fetch so the sentence-list view can render.
    private func loadBrowseChapter(chapterId: Int, pageIndex: Int = 0) async {
        guard let book = book else { return }
        let chapter = book.meta.chapters.first(where: { $0.id == chapterId })
        state.browseChapterId = chapterId
        state.browsePageIndex = pageIndex
        state.browseChapterText = chapterTextByChapter[chapterId] ?? ""
        state.browseChapterSentences = metaByChapter[chapterId]?.sentences ?? []
        state.fetchingBrowseSentences = metaInFlight[chapterId] != nil

        if state.browseChapterText.isEmpty, let chapter {
            let text = await store.chapterText(
                bookId: book.bookId, chapter: chapter,
            ) ?? ""
            // Rapid chapter switching may race: the late read must not
            // clobber a newer browse target.
            guard state.browseChapterId == chapterId else { return }
            chapterTextByChapter[chapterId] = text
            state.chapterCacheRevision &+= 1
            state.browseChapterText = text
        }
        // Make sure meta is in flight or cached so sentence-list mode can
        // render. Player loop also kicks the same fetch when needed; the
        // dedup in `kickOffMetaFetch` makes the double call cheap.
        if metaByChapter[chapterId] == nil {
            kickOffMetaFetch(chapterId: chapterId)
        }
        // Warm adjacent chapter texts so the View's paged TabView can
        // render the next/prev page during a swipe instead of flashing
        // empty until the user releases.
        prefetchAdjacentChapters(around: chapterId)
        // Reset the cached-sentence set for the prefetch progress bar
        // and recompute it from on-disk cache (if meta + text are
        // ready — otherwise the meta-arrival path will re-trigger).
        state.cachedSentencesInBrowseChapter = []
        refreshCachedSentencesForBrowseChapter()
        updateNowPlaying()
    }

    /// Drop every cached audio file for ``bookId`` and reset the
    /// in-memory state that referenced those files. Used after a user
    /// edits a book character — the matched speaker may have changed
    /// server-side, but the local cache key is
    /// ``(bookId, characterId, text)`` and would otherwise replay the
    /// old voice from disk.
    ///
    /// Does NOT interrupt the currently playing sentence: the audio
    /// is already loaded into the AVAudioEngine's playerNode, so it
    /// finishes naturally. The next sentence's prefetch starts fresh
    /// against the now-empty cache, hits the server, and writes new
    /// audio under the same key.
    ///
    /// Cancels any in-flight prefetch worker first — its pending HTTP
    /// requests may have started before the server-side patch landed
    /// and could write old-voice audio into the slot we're about to
    /// evict. ``runPrefetchWorker`` handles ``CancellationError`` by
    /// exiting cleanly without failing already-waiting continuations,
    /// so the player loop's pending awaits stay alive and resume from
    /// the new worker we kick off after the eviction completes.
    func invalidateBookAudio(bookId: String) async {
        prefetchTask?.cancel()
        prefetchTask = nil
        if book?.bookId == bookId {
            prefetchedURLs.removeAll()
        }
        await cache.evict(bookId: bookId)
        // Browse-chapter cached-set ("which sentences are pre-rendered")
        // is now stale; recompute against the freshly-evicted cache so
        // the prefetch progress bar shows reality. Safe to call when
        // the user is on a different screen — the @Published just
        // updates state for whoever reads it next.
        state.cachedSentencesInBrowseChapter = []
        refreshCachedSentencesForBrowseChapter()
        // Restart the worker if there's pending playback (subscribers
        // waiting on slots, or the player loop will fetch shortly).
        startPrefetchWorkerIfNeeded()
    }

    /// Recompute which sentences in the **browse** chapter have their
    /// TTS audio already cached on disk. Cheap per-sentence file-existence
    /// check via TTSCache's nonisolated `contains`. No-op until both
    /// meta + text for the browse chapter are available.
    private func refreshCachedSentencesForBrowseChapter() {
        guard let book = book,
              let chapterId = state.browseChapterId,
              let meta = metaByChapter[chapterId],
              let text = chapterTextByChapter[chapterId], !text.isEmpty
        else { return }
        let lines = text.components(separatedBy: "\n")
        var found: Set<Int> = []
        for (idx, sentence) in meta.sentences.enumerated() {
            let extracted = Self.extractSentenceText(sentence, lines: lines)
            if extracted.isEmpty { continue }
            let key = TTSCache.key(
                bookId: book.bookId,
                characterId: effectiveCharacterId(sentence.characterId),
                text: extracted,
            )
            if cache.contains(key: key) {
                found.insert(idx)
            }
        }
        state.cachedSentencesInBrowseChapter = found
    }

    /// Map a sentence's raw character_id to the id we actually send to
    /// the server. The narrator slot (id 0) is substituted with the
    /// user's selected narrator (``settings.narratorCharacterId``,
    /// 0 = male / 1 = female). All other ids pass through unchanged.
    private func effectiveCharacterId(_ rawId: Int) -> Int {
        rawId == 0 ? settings.narratorCharacterId : rawId
    }

    /// Add `sentenceIndex` to the browse-chapter cached set if the
    /// chapter matches. Called from prefetch-task completion paths.
    /// Uses `_ = state.cachedSentencesInBrowseChapter.insert(...)` so
    /// the @Published only fires when the set actually changes.
    private func markSentenceCachedIfBrowse(chapterId: Int, sentenceIndex: Int) {
        guard chapterId == state.browseChapterId else { return }
        if !state.cachedSentencesInBrowseChapter.contains(sentenceIndex) {
            state.cachedSentencesInBrowseChapter.insert(sentenceIndex)
        }
    }

    /// Pure helper for extracting sentence text from a chapter's lines.
    /// Mirrors the body of ``extractText`` but takes pre-split lines to
    /// amortise the O(N) split across many sentences.
    private static func extractSentenceText(_ s: Sentence, lines: [String]) -> String {
        guard s.startLine >= 1, s.endLine <= lines.count,
              s.startLine <= s.endLine else { return "" }
        if s.startLine == s.endLine {
            let line = lines[s.startLine - 1]
            let u = line.utf16
            guard s.startCol <= u.count, s.endCol <= u.count,
                  s.startCol <= s.endCol,
                  let start = u.index(u.startIndex, offsetBy: s.startCol, limitedBy: u.endIndex)?.samePosition(in: line),
                  let end = u.index(u.startIndex, offsetBy: s.endCol, limitedBy: u.endIndex)?.samePosition(in: line)
            else { return "" }
            return String(line[start..<end])
        }
        var parts: [String] = []
        for li in s.startLine...s.endLine {
            let line = lines[li - 1]
            if li == s.startLine, let s_idx = line.utf16.index(line.utf16.startIndex, offsetBy: s.startCol, limitedBy: line.utf16.endIndex)?.samePosition(in: line) {
                parts.append(String(line[s_idx...]))
            } else if li == s.endLine, let e_idx = line.utf16.index(line.utf16.startIndex, offsetBy: s.endCol, limitedBy: line.utf16.endIndex)?.samePosition(in: line) {
                parts.append(String(line[..<e_idx]))
            } else {
                parts.append(line)
            }
        }
        return parts.joined(separator: "\n")
    }

    /// Ensure the chapter text is in the cache (off-disk load when not
    /// already there). Used by the player loop when transitioning into a
    /// chapter the user isn't browsing — without this, `extractText`
    /// returns "" and prefetch silently fails.
    private func ensureChapterTextCached(chapterId: Int) async {
        if chapterTextByChapter[chapterId] != nil { return }
        guard let book = book,
              let chapter = book.meta.chapters.first(where: { $0.id == chapterId }) else {
            return
        }
        let text = await store.chapterText(
            bookId: book.bookId, chapter: chapter,
        ) ?? ""
        chapterTextByChapter[chapterId] = text
        state.chapterCacheRevision &+= 1
    }

    // MARK: - network thread: serial sliding window

    /// Recompute the prefetch window from the current position. Trims
    /// already-fetched URLs that fell out of the window, cancels any
    /// subscribers waiting for out-of-window slots, and (re)starts the
    /// serial worker if it has work to do.
    /// Idempotent — safe to call from every event that may change the
    /// anchor (position change, chapter switch, meta arrival).
    private func reconcileWindow() {
        guard let book = book, let position = state.position else {
            cancelAllPrefetch()
            return
        }
        let coord = SentenceCoord(
            bookId: book.bookId,
            chapterId: position.chapterId,
            sentenceIndex: position.sentenceIndex,
        )
        // Publish the anchor to the TTS cache so distance-based eviction
        // keeps entries near the user's current position (see §3.6.1).
        Task { await cache.setAnchor(coord) }

        let anchor = SentenceAddress(
            chapterId: position.chapterId, sentenceIndex: position.sentenceIndex,
        )
        let desiredOrdered = orderedWindowSlots(from: anchor, book: book)
        let desired = Set(desiredOrdered)

        // Drop URLs that fell outside the window — saves memory; the
        // server-side TTSCache still has the audio if we ever revisit.
        prefetchedURLs = prefetchedURLs.filter { desired.contains($0.key) }

        // Resume any subscribers waiting on slots that are no longer
        // in the window with a cancellation error so the player loop
        // can decide what to do.
        for slot in Array(prefetchSubscribers.keys) where !desired.contains(slot) {
            if let conts = prefetchSubscribers.removeValue(forKey: slot) {
                for cont in conts { cont.resume(throwing: CancellationError()) }
            }
        }

        startPrefetchWorkerIfNeeded()
    }

    /// The window's slots in **priority order**: anchor first, then
    /// anchor+1, etc., spilling into the next chapter when the current
    /// runs out. The serial worker walks this list to pick what to
    /// fetch next; the player loop's anchor naturally sits at the head.
    ///
    /// Side effect: warms text + meta caches for the next chapter as
    /// soon as the window spills into it (so when the worker reaches
    /// those slots, ``extractText`` has what it needs).
    private func orderedWindowSlots(
        from anchor: SentenceAddress, book: LocalBook,
    ) -> [SentenceAddress] {
        var slots: [SentenceAddress] = []
        var remaining = Self.windowAhead + 1
        var chapterId = anchor.chapterId
        var startIdx = anchor.sentenceIndex

        while remaining > 0 {
            guard let chapterMeta = metaByChapter[chapterId] else {
                // Need this chapter's meta to enumerate sentences. Kick
                // it; reconcile reruns when it lands.
                kickOffMetaFetch(chapterId: chapterId)
                break
            }
            let sentences = chapterMeta.sentences
            let endIdx = min(sentences.count - 1, startIdx + remaining - 1)
            if startIdx <= endIdx {
                for idx in startIdx...endIdx {
                    slots.append(SentenceAddress(chapterId: chapterId, sentenceIndex: idx))
                }
                remaining -= (endIdx - startIdx + 1)
            }
            if remaining <= 0 { break }

            let nextId = chapterId + 1
            guard book.meta.chapters.contains(where: { $0.id == nextId }) else {
                break  // end of book
            }
            ensureChapterText(for: nextId)
            chapterId = nextId
            startIdx = 0
        }
        return slots
    }

    /// Player loop awaits this for each sentence it needs to play. If
    /// the slot has already been prefetched, returns/throws immediately;
    /// otherwise waits for the serial worker to fetch it (this is the
    /// only point where the player loop can be slowed down by TTS
    /// latency). Multiple awaits on the same slot are allowed —
    /// they're all resumed when the fetch completes.
    private func awaitPrefetched(slot: SentenceAddress) async throws -> URL {
        if let result = prefetchedURLs[slot] {
            return try result.get()
        }
        return try await withCheckedThrowingContinuation { cont in
            prefetchSubscribers[slot, default: []].append(cont)
            startPrefetchWorkerIfNeeded()
        }
    }

    private func startPrefetchWorkerIfNeeded() {
        guard prefetchTask == nil, book != nil, state.position != nil else { return }
        prefetchTask = Task.detached(priority: .utility) { [weak self] in
            guard let strong = self else { return }
            await strong.runPrefetchWorker()
            await MainActor.run { strong.prefetchTask = nil }
        }
    }

    /// Serial prefetch loop. Each iteration: pick the highest-priority
    /// slot that hasn't been fetched yet (anchor → anchor+1 → ...);
    /// fetch it; record the result; resume any subscribers; repeat.
    /// Exits when the window has no more pending slots.
    private func runPrefetchWorker() async {
        while !Task.isCancelled {
            guard let book = book,
                  let position = state.position else { return }
            let anchor = SentenceAddress(
                chapterId: position.chapterId, sentenceIndex: position.sentenceIndex,
            )
            let ordered = orderedWindowSlots(from: anchor, book: book)
            // First slot in priority order that hasn't been attempted yet.
            guard let next = ordered.first(where: { prefetchedURLs[$0] == nil }) else {
                return
            }
            do {
                let url = try await fetchAudioForSlot(next, book: book)
                prefetchedURLs[next] = .success(url)
                if let waiters = prefetchSubscribers.removeValue(forKey: next) {
                    for cont in waiters { cont.resume(returning: url) }
                }
            } catch is CancellationError {
                // Worker itself got cancelled (cancelAllPrefetch / stop).
                return
            } catch {
                prefetchedURLs[next] = .failure(error)
                if let waiters = prefetchSubscribers.removeValue(forKey: next) {
                    for cont in waiters { cont.resume(throwing: error) }
                }
            }
        }
    }

    /// Resolve a slot to its sentence + character + text and call
    /// ``fetchAudio``. Returns CancellationError if the slot's chapter
    /// meta or text aren't ready yet — caller should treat as transient
    /// and let reconcile re-pick when the underlying caches land.
    private func fetchAudioForSlot(
        _ slot: SentenceAddress, book: LocalBook,
    ) async throws -> URL {
        guard let chapterMeta = metaByChapter[slot.chapterId] else {
            throw CancellationError()
        }
        let sentences = chapterMeta.sentences
        guard slot.sentenceIndex >= 0, slot.sentenceIndex < sentences.count else {
            throw CancellationError()
        }
        let sentence = sentences[slot.sentenceIndex]
        let text = extractText(for: sentence, chapterId: slot.chapterId)
        if text.isEmpty {
            throw CancellationError()  // text not loaded yet
        }
        let character = chapterMeta.characters.first(where: { $0.id == sentence.characterId })
        return try await fetchAudio(
            book: book,
            chapterId: slot.chapterId,
            sentenceIndex: slot.sentenceIndex,
            sentence: sentence,
            character: character,
            text: text,
        )
    }

    private func cancelAllPrefetch() {
        prefetchTask?.cancel()
        prefetchTask = nil
        prefetchedURLs.removeAll()
        for (_, conts) in prefetchSubscribers {
            for cont in conts { cont.resume(throwing: CancellationError()) }
        }
        prefetchSubscribers.removeAll()
    }

    private func cancelAllMetaFetches() {
        for (_, task) in metaInFlight { task.cancel() }
        metaInFlight.removeAll()
        metaFetchFailed.removeAll()
        state.fetchingBrowseSentences = false
    }

    // MARK: - network thread: meta fetch

    /// Kick off a chapter-meta fetch if not already running / cached.
    /// On success populates `metaByChapter[chapterId]` and triggers
    /// `reconcileWindow()` so the window re-expands.
    private func kickOffMetaFetch(chapterId: Int) {
        if metaByChapter[chapterId] != nil { return }
        if metaInFlight[chapterId] != nil { return }
        guard let book = book else { return }

        // Clear prior failure flag so a retry (user re-play attempt) can
        // succeed instead of getting insta-stopped by the playerLoop guard.
        metaFetchFailed.remove(chapterId)
        if chapterId == state.browseChapterId {
            state.fetchingBrowseSentences = true
        }
        let task = Task<Void, Never> { [weak self] in
            guard let self = self else { return }
            defer {
                if self.state.browseChapterId == chapterId {
                    self.state.fetchingBrowseSentences = false
                }
                self.metaInFlight[chapterId] = nil
            }
            do {
                let response = try await self.api.chapterMeta(
                    bookId: book.bookId, chapterId: chapterId,
                )
                if Task.isCancelled { return }
                self.metaByChapter[chapterId] = response.meta
                self.state.chapterCacheRevision &+= 1
                if chapterId == self.state.browseChapterId {
                    self.state.browseChapterSentences = response.meta.sentences
                    // Sentence list just became available — recompute
                    // which ones are already in the TTS cache so the
                    // progress bar reflects state.
                    self.refreshCachedSentencesForBrowseChapter()
                }
                // If position was set before meta arrived (open(), jumpPlay(),
                // chapter transition), it has the (0, 0) line/col sentinel.
                // Now that we have meta, refresh it so persistence / lock-
                // screen consumers see the real durable address.
                if let pos = self.state.position,
                   pos.chapterId == chapterId,
                   pos.startLine == 0, pos.startCol == 0 {
                    self.state.position = self.makePosition(
                        chapterId: chapterId, sentenceIndex: pos.sentenceIndex,
                    )
                }
                self.metaFetchFailed.remove(chapterId)
                // Meta arrived — unlock window slots that were waiting on it.
                self.reconcileWindow()
            } catch is CancellationError {
                return
            } catch let urlError as URLError where urlError.code == .cancelled {
                // URLSession's bytes-stream surfaces cancellation as
                // `URLError(.cancelled)` (NSURLErrorCancelled, code -999),
                // *not* `CancellationError`. We don't want to alert on
                // it — superseded fetches (chapter switch, jump-play,
                // browse-ahead) cancel the prior in-flight request by
                // design, so this catch keeps the cancellation silent
                // alongside the structured-cancellation path above.
                return
            } catch {
                self.metaFetchFailed.insert(chapterId)
                // Surface the error only when the failed chapter is the
                // one the user is actually looking at — a failed prefetch
                // for a chapter ahead of the play position shouldn't pop
                // an alert. Show the raw error type + full message
                // (no truncation, no UX softening — debugging takes
                // priority over presentation).
                if chapterId == self.state.browseChapterId {
                    let nsErr = error as NSError
                    self.state.error = "ch=\(chapterId) "
                        + "type=\(type(of: error)) "
                        + "domain=\(nsErr.domain) code=\(nsErr.code)\n"
                        + "desc=\(error.localizedDescription)\n"
                        + "raw=\(String(describing: error))"
                }
            }
        }
        metaInFlight[chapterId] = task
    }

    // MARK: - playback thread

    private func playerLoop() async {
        while !Task.isCancelled && state.isPlaying {
            guard let anchor = currentAddress() else {
                state.isPlaying = false
                return
            }

            // Wait for this chapter's meta (hits cache if already fetched).
            let sentences = await ensureMeta(chapterId: anchor.chapterId)
            if Task.isCancelled || !state.isPlaying { return }

            // Meta fetch failed — stop playback with error instead of
            // silently "transitioning" to the next chapter (which would
            // make it look like the chapter was empty).
            if metaFetchFailed.contains(anchor.chapterId) {
                state.isPlaying = false
                return
            }

            // Past chapter end? Transition.
            if anchor.sentenceIndex >= sentences.count {
                if await transitionToNextChapter() { continue } else { return }
            }

            // Publish anchor + let the prefetch worker pick it up.
            updatePosition(makePosition(
                chapterId: anchor.chapterId,
                sentenceIndex: anchor.sentenceIndex,
            ))
            reconcileWindow()

            do {
                // Worker fetches in priority order, so the anchor is the
                // first slot it works on — typically already done or
                // imminent by the time we get here.
                let url = try await awaitPrefetched(slot: anchor)
                if Task.isCancelled || !state.isPlaying { return }
                try await playFile(url)
            } catch is CancellationError {
                return
            } catch {
                if Task.isCancelled || !state.isPlaying { return }
                let nsErr = error as NSError
                state.error = "TTS ch=\(anchor.chapterId) idx=\(anchor.sentenceIndex) "
                    + "type=\(type(of: error)) "
                    + "domain=\(nsErr.domain) code=\(nsErr.code)\n"
                    + "desc=\(error.localizedDescription)\n"
                    + "raw=\(String(describing: error))"
                try? await Task.sleep(nanoseconds: 500_000_000)
            }

            if Task.isCancelled || !state.isPlaying { return }

            // Advance. Triggers next iteration's reconcile via new anchor.
            updatePosition(makePosition(
                chapterId: anchor.chapterId,
                sentenceIndex: anchor.sentenceIndex + 1,
            ))
        }
    }

    private func currentAddress() -> SentenceAddress? {
        if let pos = state.position {
            return SentenceAddress(
                chapterId: pos.chapterId, sentenceIndex: pos.sentenceIndex,
            )
        }
        // No play position yet — fall back to whatever the user is
        // browsing so a press of "play" before the loop ever ran starts
        // somewhere sensible.
        guard let cid = state.browseChapterId else { return nil }
        return SentenceAddress(chapterId: cid, sentenceIndex: 0)
    }

    /// Block until we have meta for the given chapter. Returns the
    /// sentences (possibly empty if the fetch failed).
    private func ensureMeta(chapterId: Int) async -> [Sentence] {
        if let cached = metaByChapter[chapterId] { return cached.sentences }
        kickOffMetaFetch(chapterId: chapterId)
        if let task = metaInFlight[chapterId] {
            await task.value
        }
        return metaByChapter[chapterId]?.sentences ?? []
    }

    /// Called by player loop when the current play chapter is finished.
    /// Advances the **play** position to the next chapter and warms the
    /// per-chapter text cache so prefetch's `extractText` can succeed.
    /// Browse follows automatically when followMode is on (via
    /// `updatePosition`); otherwise the user keeps reading wherever they
    /// were and only the lock-screen / `position` changes.
    /// Returns true on transition, false on book end.
    private func transitionToNextChapter() async -> Bool {
        guard let book = book,
              let pos = state.position else { return false }
        let next = pos.chapterId + 1
        guard book.meta.chapters.contains(where: { $0.id == next }) else {
            state.isPlaying = false
            return false
        }
        await ensureChapterTextCached(chapterId: next)
        updatePosition(makePosition(chapterId: next, sentenceIndex: 0))
        reconcileWindow()
        return true
    }

    // MARK: - TTS fetch (network thread inner detail)

    private func fetchAudio(
        book: LocalBook, chapterId: Int, sentenceIndex: Int,
        sentence: Sentence, character: Character?, text: String,
    ) async throws -> URL {
        // Voice resolution lives entirely on the server now — the App
        // sends only the (post-substitution) character_id and lets the
        // server map it to a Speaker. The ``character`` parameter is
        // kept as input for diagnostic logging only.
        _ = character
        let characterId = effectiveCharacterId(sentence.characterId)
        let request = TTSRequest(
            bookId: book.bookId,
            chapterId: chapterId,
            characterId: characterId,
            text: text,
            tone: sentence.tone,
        )
        let key = TTSCache.key(
            bookId: book.bookId,
            characterId: characterId,
            text: text,
        )
        if let cached = await cache.get(key: key) {
            // Already on disk — make sure the progress bar reflects it
            // (initial scan may have run before this sentence was in
            // the meta yet, so the set might be missing it).
            markSentenceCachedIfBrowse(chapterId: chapterId, sentenceIndex: sentenceIndex)
            return cached
        }
        // Retry with cancellation short-circuit. Jump cancels the task;
        // URLSession raises URLError.cancelled; Task.sleep raises
        // CancellationError. Either aborts the loop immediately.
        let coord = SentenceCoord(
            bookId: book.bookId,
            chapterId: chapterId,
            sentenceIndex: sentenceIndex,
        )
        var lastError: Error?
        for attempt in 1...3 {
            try Task.checkCancellation()
            do {
                let response = try await api.synthesize(request: request)
                try Task.checkCancellation()
                let url = try await cache.store(
                    key: key, coord: coord, data: response.audioData,
                )
                markSentenceCachedIfBrowse(chapterId: chapterId, sentenceIndex: sentenceIndex)
                return url
            } catch is CancellationError {
                throw CancellationError()
            } catch let urlError as URLError where urlError.code == .cancelled {
                throw CancellationError()
            } catch {
                lastError = error
                try await Task.sleep(nanoseconds: UInt64(200_000_000 * attempt))
            }
        }
        throw lastError ?? APIError.badResponse
    }

    // MARK: - text extraction

    /// Extract the verbatim text for `sentence` from a chapter's text.
    /// Reads from the per-chapter text cache so prefetch works even when
    /// the user is browsing a different chapter than what's playing.
    /// Browse loads + chapter transitions warm the cache before fetching
    /// any TTS, so a miss here means we genuinely don't have the chapter
    /// (early in `open()` / failed disk read).
    private func extractText(for sentence: Sentence, chapterId: Int) -> String {
        guard let text = chapterTextByChapter[chapterId], !text.isEmpty else {
            return ""
        }
        let lines = text.components(separatedBy: "\n")
        guard sentence.startLine >= 1,
              sentence.endLine <= lines.count,
              sentence.startLine <= sentence.endLine else { return "" }
        if sentence.startLine == sentence.endLine {
            let line = lines[sentence.startLine - 1]
            let start = utf16Index(line, offset: sentence.startCol)
            let end = utf16Index(line, offset: sentence.endCol)
            guard let s = start, let e = end, s <= e else { return "" }
            return String(line[s..<e])
        }
        var parts: [String] = []
        for lineIdx in sentence.startLine...sentence.endLine {
            let line = lines[lineIdx - 1]
            if lineIdx == sentence.startLine {
                if let s = utf16Index(line, offset: sentence.startCol) {
                    parts.append(String(line[s...]))
                }
            } else if lineIdx == sentence.endLine {
                if let e = utf16Index(line, offset: sentence.endCol) {
                    parts.append(String(line[..<e]))
                }
            } else {
                parts.append(line)
            }
        }
        return parts.joined(separator: "\n")
    }

    private func utf16Index(_ s: String, offset: Int) -> String.Index? {
        // Server positions are UTF-16 offsets (Python `str[...]` semantics
        // on BMP code points — Chinese novels are all BMP).
        return s.utf16.index(
            s.utf16.startIndex, offsetBy: offset, limitedBy: s.utf16.endIndex,
        )?.samePosition(in: s)
    }

    // MARK: - audio graph + playback

    private func setupAudioGraph() {
        engine.attach(playerNode)
        engine.attach(timePitch)
        engine.attach(eq)
        engine.attach(limiter)
        // playerNode → timePitch → eq → limiter → mainMixer
        engine.connect(playerNode, to: timePitch, format: nil)
        engine.connect(timePitch, to: eq, format: nil)
        engine.connect(eq, to: limiter, format: nil)
        engine.connect(limiter, to: engine.mainMixerNode, format: nil)
        configureLimiter()
        applyPlaybackSpeed()
    }

    /// Push the current settings.playbackSpeed into the time-pitch unit.
    /// Called at startup, on user changes, and right before playFile so
    /// every sentence honours the latest speed.
    private func applyPlaybackSpeed() {
        // AVAudioUnitTimePitch.rate is clamped to [1/32, 32] internally;
        // settings.playbackSpeed is already validated to [0.5, 2.0].
        timePitch.rate = Float(settings.playbackSpeed)
    }

    /// AUDynamicsProcessor as a brick-wall limiter (§3.8): -0.3 dBFS
    /// threshold, 3 ms attack, 60 ms release. Param ids and ranges are
    /// from `AudioToolbox/AudioUnitParameters.h`
    /// (`kDynamicsProcessorParam_*`).
    private func configureLimiter() {
        let unit = limiter.audioUnit
        // Threshold (dB, -40..20)
        AudioUnitSetParameter(unit, 0, kAudioUnitScope_Global, 0, -0.3, 0)
        // HeadRoom (dB, 0.1..40) — soft-knee width above threshold
        AudioUnitSetParameter(unit, 1, kAudioUnitScope_Global, 0, 0.5, 0)
        // ExpansionRatio (1..50) — keep at 1 (no expansion)
        AudioUnitSetParameter(unit, 2, kAudioUnitScope_Global, 0, 1.0, 0)
        // ExpansionThreshold (dB, -120..0) — far below to disable
        AudioUnitSetParameter(unit, 3, kAudioUnitScope_Global, 0, -120.0, 0)
        // AttackTime (s, 0.0001..0.2)
        AudioUnitSetParameter(unit, 4, kAudioUnitScope_Global, 0, 0.003, 0)
        // ReleaseTime (s, 0.01..3)
        AudioUnitSetParameter(unit, 5, kAudioUnitScope_Global, 0, 0.060, 0)
        // MasterGain (dB, -40..40) — keep neutral; user gain lives on EQ
        AudioUnitSetParameter(unit, 6, kAudioUnitScope_Global, 0, 0.0, 0)
    }

    // MARK: - lock-screen / control center (§3.5)

    /// Register MPRemoteCommandCenter targets once. Idempotent — re-
    /// registration would just stack handlers, so we remove any prior
    /// targets first.
    private func setupRemoteCommands() {
        #if canImport(MediaPlayer)
        let cc = MPRemoteCommandCenter.shared()
        cc.playCommand.removeTarget(nil)
        cc.pauseCommand.removeTarget(nil)
        cc.togglePlayPauseCommand.removeTarget(nil)
        cc.nextTrackCommand.removeTarget(nil)
        cc.previousTrackCommand.removeTarget(nil)

        cc.playCommand.isEnabled = true
        cc.pauseCommand.isEnabled = true
        cc.togglePlayPauseCommand.isEnabled = true
        cc.nextTrackCommand.isEnabled = true
        cc.previousTrackCommand.isEnabled = true

        cc.playCommand.addTarget { [weak self] _ in
            Task { @MainActor in self?.play() }
            return .success
        }
        cc.pauseCommand.addTarget { [weak self] _ in
            Task { @MainActor in self?.pause() }
            return .success
        }
        cc.togglePlayPauseCommand.addTarget { [weak self] _ in
            Task { @MainActor in
                guard let self = self else { return }
                if self.state.isPlaying { self.pause() } else { self.play() }
            }
            return .success
        }
        cc.nextTrackCommand.addTarget { [weak self] _ in
            Task { @MainActor in await self?.gotoChapter(offset: 1) }
            return .success
        }
        cc.previousTrackCommand.addTarget { [weak self] _ in
            Task { @MainActor in await self?.gotoChapter(offset: -1) }
            return .success
        }
        #endif
    }

    /// Pause when the system interrupts our audio session — phone
    /// call, Siri, FaceTime, and the rare alert that slips past
    /// ``setPrefersNoInterruptionsFromSystemAlerts`` (the API call in
    /// ``TingShuApp.init`` that asks iOS to keep notification sounds
    /// from cutting in). iOS stops AVAudioEngine output on its own
    /// during the interruption; we mirror that into ``state.isPlaying``
    /// so the play button + lock-screen toggle update.
    ///
    /// We do NOT auto-resume on ``.ended`` even when the system sets
    /// ``shouldResume`` — for an audiobook the user typically wants
    /// to decide whether to keep going, and silently restarting after
    /// a phone call surprises people more than it helps.
    private func setupInterruptionObserver() {
        #if canImport(UIKit)
        interruptionObserver = NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification,
            object: nil,
            queue: .main,
        ) { [weak self] note in
            guard let raw = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
                  let type = AVAudioSession.InterruptionType(rawValue: raw),
                  type == .began else { return }
            Task { @MainActor [weak self] in
                guard let self = self, self.state.isPlaying else { return }
                self.pause()
            }
        }
        #endif
    }

    /// Pause when the active output route disappears (headphone unplug,
    /// Bluetooth disconnect). iOS automatically silences the engine in
    /// that case, but `state.isPlaying` and `MPNowPlayingInfoCenter`
    /// stay in their "playing" state — so the lock-screen toggle would
    /// otherwise keep showing pause-icon over silent audio. Mirrors the
    /// system Music app: pause on disconnect, do NOT auto-resume on
    /// reconnect.
    private func setupRouteChangeObserver() {
        #if canImport(UIKit)
        routeChangeObserver = NotificationCenter.default.addObserver(
            forName: AVAudioSession.routeChangeNotification,
            object: nil,
            queue: .main,
        ) { [weak self] note in
            guard let raw = note.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt,
                  let reason = AVAudioSession.RouteChangeReason(rawValue: raw),
                  reason == .oldDeviceUnavailable else { return }
            Task { @MainActor [weak self] in
                guard let self = self, self.state.isPlaying else { return }
                self.pause()
            }
        }
        #endif
    }

    /// Hop the **play** position N chapters forward/back, clamped. Used
    /// by lock-screen next/previous — these are "play here" gestures so
    /// they go through `jumpPlay` (which also pulls browse + follow
    /// mode along).
    private func gotoChapter(offset: Int) async {
        guard let book = book, let pos = state.position else { return }
        let target = pos.chapterId + offset
        guard book.meta.chapters.contains(where: { $0.id == target }) else { return }
        await jumpPlay(chapterId: target, sentenceIndex: 0)
    }

    /// Push the current book/chapter + play state to the system Now-
    /// Playing center so the lock screen and control center reflect
    /// reality. Reports the **play** chapter (not browse) — that's what's
    /// actually being heard.
    private func updateNowPlaying() {
        #if canImport(MediaPlayer)
        guard let book = book,
              let cid = state.position?.chapterId ?? state.browseChapterId else {
            clearNowPlaying()
            return
        }
        let chapterTitle = book.meta.chapters.first(where: { $0.id == cid })?.title
            ?? "第 \(cid) 章"
        var info: [String: Any] = [
            MPMediaItemPropertyTitle: chapterTitle,
            MPMediaItemPropertyArtist: book.title,
            MPMediaItemPropertyAlbumTrackNumber: cid,
            MPMediaItemPropertyAlbumTrackCount: book.meta.chapters.count,
            MPNowPlayingInfoPropertyPlaybackRate: state.isPlaying ? 1.0 : 0.0,
            MPNowPlayingInfoPropertyDefaultPlaybackRate: 1.0,
            MPNowPlayingInfoPropertyMediaType: MPNowPlayingInfoMediaType.audio.rawValue,
        ]
        if !book.author.isEmpty {
            info[MPMediaItemPropertyAlbumTitle] = book.author
        }
        MPNowPlayingInfoCenter.default().nowPlayingInfo = info
        #endif
    }

    private func clearNowPlaying() {
        #if canImport(MediaPlayer)
        MPNowPlayingInfoCenter.default().nowPlayingInfo = nil
        #endif
    }

    // MARK: - audio playback

    /// Eagerly bring the audio graph online so the user's first tap on
    /// the play button isn't delayed by HAL allocation. AVAudioEngine's
    /// first ``start()`` does substantial setup (typically 50-200ms)
    /// that can hitch the play-button animation if it runs in
    /// ``playFile``. Called from ``open(book:saved:)``.
    private func warmAudioEngineIfNeeded() {
        if engine.isRunning { return }
        // Non-fatal — `playFile` will retry the start inline. Most
        // start failures are transient (route change, no output device).
        try? engine.start()
    }

    private func playFile(_ url: URL) async throws {
        // Disk read + AVAudioFile decoder init can take a few ms;
        // push it off the main actor so the play-button animation
        // doesn't hitch.
        let file = try await Task.detached(priority: .userInitiated) {
            try AVAudioFile(forReading: url)
        }.value
        if !engine.isRunning {
            try engine.start()
        }
        eq.globalGain = Float(settings.gainDB)
        applyPlaybackSpeed()

        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            var resumed = false
            let resume: (Result<Void, Error>) -> Void = { result in
                guard !resumed else { return }
                resumed = true
                cont.resume(with: result)
            }
            playerNode.scheduleFile(file, at: nil, completionCallbackType: .dataPlayedBack) { _ in
                resume(.success(()))
            }
            playerNode.play()
        }
    }
}
