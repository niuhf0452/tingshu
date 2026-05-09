// Top-level model for the bookshelf: owns the canonical list of books
// from the server + the subset downloaded locally. SwiftUI observes this
// via @Published so the bookshelf view updates on sync / import / delete.
//
// Split of concerns:
//   - APIClient     : raw HTTP
//   - BookLibrary   : local file system (zip install, meta.json, chapter text)
//   - BookStore     : orchestrates both, exposes state for the UI
import Foundation
import SwiftUI

/// User-visible state per book. Merges the server's view (from
/// `GET /api/books`) with the local download state.
struct ShelfBook: Identifiable, Hashable {
    let bookId: String
    var title: String
    var author: String
    var serverStatus: BookStatus
    var isLocal: Bool
    var chapterCount: Int

    var id: String { bookId }

    /// What the UI should render as the overall status.
    var displayStatus: BookStatus {
        if !isLocal {
            switch serverStatus {
            case .ready: return .downloading
            default: return serverStatus
            }
        }
        return .ready
    }
}

@MainActor
final class BookStore: ObservableObject {
    @Published private(set) var books: [ShelfBook] = []
    /// Last refresh/sync error. Shown inline in the bookshelf empty state —
    /// not as a modal alert, so a flaky connection doesn't block the UI.
    @Published var connectionError: String?
    /// Separate channel for transient user-facing errors (upload failed,
    /// download failed, etc.). Surfaced as an alert so the user notices.
    @Published var lastError: String?

    private let api: APIClient
    private let library: BookLibrary
    private let settings: SettingsStore
    private let progressStore: ProgressStore?
    /// Set while a ``refresh()`` is in flight so overlapping calls (pull-to-
    /// refresh, `.task` onAppear, post-upload auto-refresh) share one pass
    /// instead of stacking up and thrashing the disk.
    private var currentRefresh: Task<Void, Never>?
    /// Book ids currently being downloaded. Used to dedup concurrent
    /// `downloadBook` calls — without this, refresh() + refreshMetaIfStale()
    /// can each spawn a download for the same id, racing on the local
    /// library directory.
    private var downloadsInFlight: Set<String> = []
    /// Background poll for books still in server-side processing —
    /// see ``schedulePollIfNeeded``. Only active while at least one book
    /// is `uploading` / `processing`.
    private var pollTask: Task<Void, Never>?

    init(
        api: APIClient,
        settings: SettingsStore,
        library: BookLibrary = BookLibrary(rootURL: BookLibrary.defaultRoot),
        progressStore: ProgressStore? = nil,
    ) {
        self.api = api
        self.settings = settings
        self.library = library
        self.progressStore = progressStore
    }

    // MARK: - read-side

    /// Expose the raw local book for a given id (nil if not yet downloaded).
    /// Disk I/O happens off the main thread.
    func localBook(bookId: String) async -> LocalBook? {
        let library = self.library
        return await Task.detached(priority: .userInitiated) {
            try? library.load(bookId: bookId)
        }.value
    }

    /// Async variant used by playback. The chapter text is small so the
    /// cost is dominated by the syscall — still off-main to keep the
    /// player UI responsive when swiping chapters.
    func chapterText(bookId: String, chapter: ChapterEntry) async -> String? {
        let library = self.library
        return await Task.detached(priority: .userInitiated) {
            try? library.chapterText(bookId: bookId, chapter: chapter)
        }.value
    }

    // MARK: - sync

    /// Pull server book list, reconcile with local library, and trigger
    /// downloads for books the server has marked `ready` that we don't
    /// yet have on disk. Re-entrant calls are coalesced: a second call
    /// while a refresh is in flight just awaits the existing one.
    func refresh() async {
        if let ongoing = currentRefresh {
            await ongoing.value
            return
        }
        let task = Task { await self.doRefresh() }
        currentRefresh = task
        await task.value
        currentRefresh = nil
    }

    private func doRefresh() async {
        // Phase 1: publish the local-disk snapshot immediately. This
        // enforces §3.9 "UI 永不等待网络" — if the server is down, the
        // user still sees every book they've already downloaded within
        // milliseconds, instead of waiting out the HTTP timeout.
        //
        // Skip the publish when the shelf already has entries (pull-to-
        // refresh: don't flicker the list back to local-only while the
        // network call is in flight).
        if books.isEmpty {
            let localSnapshot = await loadLocalOnlyShelf()
            if !localSnapshot.isEmpty {
                books = localSnapshot
            }
        }

        // Phase 2: hit the server.
        let serverBooks: [BookListItem]
        do {
            serverBooks = try await api.listBooks()
        } catch {
            connectionError = error.localizedDescription
            // If Phase 1 published nothing (still empty on a first load
            // that timed out) retry the local read now so the shelf
            // isn't empty.
            if books.isEmpty {
                books = await loadLocalOnlyShelf()
            }
            return
        }
        connectionError = nil

        // Phase 3: merge server truth over local snapshot.
        //
        // The server's `listBooks()` response is authoritative when it
        // succeeds — books absent from it are treated as deleted, even
        // if the local cache still has them. Two reasons:
        //
        // 1. A locally-cached but server-deleted book is a zombie —
        //    no chapter-meta SSE, no TTS synthesis, no re-download. It
        //    half-works at best and pops cancellation/404 errors at
        //    worst.
        // 2. The server-side delete path (DELETE /api/books/{id}) and
        //    the out-of-band path (user rm-ing data/books/<id>/) should
        //    propagate the same way — neither survives a refresh.
        //
        // Offline survival is preserved because listBooks() throwing
        // returns early at Phase 2 above, leaving `books` populated from
        // the local-only fallback. We only reach Phase 3 with a
        // confirmed-good server response.
        let localIds = await listLocalIds()
        var merged: [ShelfBook] = []
        for s in serverBooks {
            merged.append(ShelfBook(
                bookId: s.bookId,
                title: s.title,
                author: s.author,
                serverStatus: s.status,
                isLocal: localIds.contains(s.bookId),
                chapterCount: s.chapterCount
            ))
        }
        // Garbage-collect zombies: local books the server has forgotten.
        // Delete the on-disk copy + persisted progress so they don't
        // resurface on next launch (the LocalLibrary scan picks up
        // anything in `Documents/library/`).
        let serverIdSet = Set(serverBooks.map(\.bookId))
        let zombieIds = localIds.subtracting(serverIdSet)
        if !zombieIds.isEmpty {
            let library = self.library
            for zid in zombieIds {
                progressStore?.clear(bookId: zid)
                Task.detached(priority: .utility) {
                    try? library.delete(bookId: zid)
                }
            }
        }
        // Sort by title only (locale-aware, so Chinese sorts by pinyin
        // not Unicode code-point — `大道之上` (d…) < `剑来` (j…), which
        // matches what users expect from a Chinese-language file
        // browser). State (isLocal / processing / downloading) MUST NOT
        // affect order — otherwise a book moves position as it
        // transitions through import → download → ready, which the
        // user perceives as the shelf "shuffling" itself.
        books = merged.sorted {
            $0.title.localizedStandardCompare($1.title) == .orderedAscending
        }

        // Kick off downloads for ready-but-not-local books. Detached so
        // the unzip work in each download doesn't stack up on MainActor.
        for book in merged where book.serverStatus == .ready && !book.isLocal {
            Task.detached(priority: .utility) { [weak self] in
                await self?.downloadBook(bookId: book.bookId)
            }
        }

        // Schedule a follow-up poll if anything is still server-side
        // processing (§3.3). Re-evaluated on every refresh — once the
        // shelf has no in-progress books, the chain stops.
        schedulePollIfNeeded()
    }

    // MARK: - import-status polling

    /// Wall-clock gap between two consecutive `/api/books` calls while
    /// any book is still server-side processing. Measured **between**
    /// API responses (not start-to-start), since the sleep happens at
    /// the tail of `doRefresh()` after the response has landed —
    /// network latency is not counted toward this budget. Imports are
    /// usually fast, so 1 s feels live without flooding the server.
    private static let importPollIntervalSec: UInt64 = 1

    /// Schedule the next refresh if any book is still being processed
    /// server-side. Cancels any prior pending poll. Stops the chain
    /// once everything reaches a terminal state (ready / failed).
    private func schedulePollIfNeeded() {
        pollTask?.cancel()
        let stillProcessing = books.contains { needsServerPoll($0) }
        guard stillProcessing else {
            pollTask = nil
            return
        }
        let nanos = Self.importPollIntervalSec * 1_000_000_000
        pollTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: nanos)
            guard !Task.isCancelled, let self = self else { return }
            await self.refresh()
            // refresh() calls schedulePollIfNeeded() at the end, so we
            // get re-armed automatically when more polling is required.
        }
    }

    /// Server is still working on this book — poll it. Local-only states
    /// (`downloading`, `ready`, `failed`) don't move on their own and
    /// shouldn't keep the poll alive.
    private func needsServerPoll(_ b: ShelfBook) -> Bool {
        switch b.serverStatus {
        case .uploading, .processing: return true
        case .ready, .downloading, .failed: return false
        }
    }

    // MARK: - detached library helpers

    /// Returns the ids of books that are **completely** installed locally
    /// (meta.json + every declared chapter text file). Half-installed
    /// books — leftover from pre-atomic-install crashes — are excluded,
    /// so ``downloadBook`` can re-fetch them on the next refresh.
    private func listLocalIds() async -> Set<String> {
        let library = self.library
        return await Task.detached(priority: .userInitiated) {
            Set(library.listBookIds().filter { library.isComplete(bookId: $0) })
        }.value
    }

    private func loadLocalOnlyShelf() async -> [ShelfBook] {
        let library = self.library
        return await Task.detached(priority: .userInitiated) { () -> [ShelfBook] in
            let ids = library.listBookIds()
            return ids.compactMap { id -> ShelfBook? in
                // Only surface complete books. Half-installs would show
                // up as "已导入" but crash into "章节无内容" when opened.
                guard library.isComplete(bookId: id),
                      let local = try? library.load(bookId: id) else { return nil }
                return ShelfBook(
                    bookId: local.bookId, title: local.title, author: local.author,
                    serverStatus: .ready, isLocal: true,
                    chapterCount: local.chapterCount,
                )
            }.sorted {
                $0.title.localizedStandardCompare($1.title) == .orderedAscending
            }
        }.value
    }

    // MARK: - upload + download

    /// Upload a TXT/EPUB file. Shows an immediate placeholder in the book
    /// list so the user sees activity while the server runs chapter
    /// detection (can take minutes on MLX for long books). Returns the
    /// real book_id on success.
    @discardableResult
    func uploadBook(data: Data, filename: String) async -> String? {
        let placeholderId = "uploading:\(UUID().uuidString)"
        let displayTitle = (filename as NSString).deletingPathExtension
        let placeholder = ShelfBook(
            bookId: placeholderId,
            title: displayTitle.isEmpty ? filename : displayTitle,
            author: "",
            serverStatus: .uploading,
            isLocal: false,
            chapterCount: 0,
        )
        // Slot the placeholder into its sorted-by-title position so the
        // book appears where it'll eventually live — instead of jumping
        // to the top during upload and shuffling down to its real spot
        // after refresh().
        books.append(placeholder)
        books.sort {
            $0.title.localizedStandardCompare($1.title) == .orderedAscending
        }
        defer { books.removeAll { $0.bookId == placeholderId } }

        do {
            let response = try await api.uploadBook(data: data, filename: filename)
            await refresh()
            return response.bookId
        } catch {
            lastError = "导入失败：\(error.localizedDescription)"
            return nil
        }
    }

    func downloadBook(bookId: String) async {
        // Dedup: if another caller (refresh's auto-downloader or
        // refreshMetaIfStale) is already pulling this book, bail.
        if downloadsInFlight.contains(bookId) { return }
        downloadsInFlight.insert(bookId)
        defer { downloadsInFlight.remove(bookId) }

        let library = self.library
        do {
            let zipData = try await api.downloadBookArchive(bookId: bookId)
            // Unzip is multi-MB on big books — never run it on MainActor.
            try await Task.detached(priority: .utility) {
                try library.installFromZip(bookId: bookId, zipData: zipData)
            }.value
            if let idx = books.firstIndex(where: { $0.bookId == bookId }) {
                books[idx].isLocal = true
            }
        } catch {
            lastError = "下载 \(bookId) 失败：\(error.localizedDescription)"
        }
    }

    /// Delete a book everywhere: server first, then local copy. If the
    /// server call fails, the local copy is left intact so the user can
    /// retry once the server is reachable again. See technical-plan §2.2.1.
    func deleteBook(bookId: String) async {
        do {
            try await api.deleteBook(bookId: bookId)
        } catch {
            lastError = "删除失败：\(error.localizedDescription)"
            return
        }
        let library = self.library
        await Task.detached(priority: .utility) {
            try? library.delete(bookId: bookId)
        }.value
        // Drop the persisted play progress too — the row would otherwise
        // dangle and resurface if the book is re-imported with the same id.
        progressStore?.clear(bookId: bookId)
        books.removeAll { $0.bookId == bookId }
    }

}
