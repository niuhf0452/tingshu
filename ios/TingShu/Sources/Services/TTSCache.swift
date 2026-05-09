// On-disk cache for TTS audio (tech-plan §3.6.1).
//
// Storage layout:
//
//   <Caches>/tts/<hash>.m4a      — audio files (AAC in MP4 container,
//                                  48 kbps mono 24 kHz), keyed by
//                                  SHA256(TTS params)
//   <Caches>/tts/index.json      — hash → SentenceCoord, used for eviction
//
// Eviction policy: **distance-from-anchor**, NOT FIFO. When the cache
// exceeds `maxBytes`, the entries farthest from the current playback
// anchor are evicted first — so a backward jump inside the current
// chapter reliably hits cache even if the cached window has rolled
// forward. See §3.6.1 for the scenario that motivates this.
//
// Compute-reuse only: a full cache wipe is safe (re-synth re-fills on
// demand). The entry index is rebuilt lazily and tolerates missing /
// orphan files.
import Foundation
import CryptoKit


/// Where a cached TTS audio chunk belongs in the book's sentence address
/// space. Used for distance-based eviction.
struct SentenceCoord: Codable, Hashable, Sendable {
    let bookId: String
    let chapterId: Int
    let sentenceIndex: Int
}


actor TTSCache {
    private let rootURL: URL
    private let maxBytes: Int

    /// hash → coord, persisted at `<rootURL>/index.json`.
    private var index: [String: SentenceCoord] = [:]
    /// Current playback position. When set, eviction sorts by distance
    /// from this anchor instead of falling back to FIFO.
    private var anchor: SentenceCoord?
    /// True between a mutation and the debounced write-back.
    private var indexDirty = false

    init(rootURL: URL = TTSCache.defaultRoot, maxBytes: Int) {
        self.rootURL = rootURL
        self.maxBytes = maxBytes
        try? FileManager.default.createDirectory(
            at: rootURL, withIntermediateDirectories: true,
        )
        // Load the index eagerly at init. This is safe non-async because
        // `index` is this actor's isolated state and nothing else has a
        // reference to this instance yet.
        let indexPath = rootURL.appendingPathComponent("index.json")
        if let data = try? Data(contentsOf: indexPath),
           let parsed = try? JSONDecoder().decode([String: SentenceCoord].self, from: data) {
            self.index = parsed
        }
    }

    static var defaultRoot: URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        return caches.appendingPathComponent("tts", isDirectory: true)
    }

    /// Cache key = `sha256(bookId || characterId || text)`. Tone is
    /// excluded (server-side cache also drops it; see
    /// ``server/app/core/tts_cache.py``). Speed is excluded because
    /// playback rate is applied client-side via ``AVAudioUnitTimePitch``,
    /// so a single cached file serves all speeds. ``characterId`` here
    /// is the **post-substitution** id — for narrator sentences, the
    /// PlaybackService replaces id 0 with the user's chosen narrator id
    /// (``settings.narratorCharacterId``) before computing the key.
    static func key(
        bookId: String, characterId: Int, text: String,
    ) -> String {
        var hasher = SHA256()
        hasher.update(data: Data(bookId.utf8))
        hasher.update(data: Data(":\(characterId):".utf8))
        hasher.update(data: Data(text.utf8))
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    func get(key: String) -> URL? {
        let url = fileURL(for: key)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// Cheap, non-isolated existence check — for batch scans (e.g. the
    /// player's prefetch progress bar checking dozens of sentences per
    /// chapter switch) that would otherwise pay the actor-hop cost
    /// per check. Reads only `rootURL` (immutable) + the file system,
    /// no shared mutable state.
    nonisolated func contains(key: String) -> Bool {
        let url = rootURL.appendingPathComponent("\(key).m4a")
        return FileManager.default.fileExists(atPath: url.path)
    }

    /// Total bytes occupied by cached audio files. Walks `rootURL` and
    /// sums every regular file's size, skipping `index.json` and any
    /// dotfiles. Non-isolated so the Settings view can poll it without
    /// hopping the actor; reads only the filesystem.
    nonisolated func currentSizeBytes() -> Int {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(
            at: rootURL,
            includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return 0 }
        var total = 0
        for url in entries {
            if url.lastPathComponent == "index.json" { continue }
            let values = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            if values?.isRegularFile != true { continue }
            total += values?.fileSize ?? 0
        }
        return total
    }

    func store(key: String, coord: SentenceCoord, data: Data) throws -> URL {
        let url = fileURL(for: key)
        try data.write(to: url, options: .atomic)
        index[key] = coord
        indexDirty = true
        evictIfNeeded()
        persistIndex()
        return url
    }

    /// Tell the cache the current playback position so distance-based
    /// eviction can prefer nearby entries. Passing `nil` reverts to FIFO.
    func setAnchor(_ coord: SentenceCoord?) {
        self.anchor = coord
    }

    func clear() throws {
        try FileManager.default.removeItem(at: rootURL)
        try FileManager.default.createDirectory(
            at: rootURL, withIntermediateDirectories: true,
        )
        index.removeAll()
        indexDirty = true
        persistIndex()
    }

    // MARK: - helpers

    private func fileURL(for key: String) -> URL {
        rootURL.appendingPathComponent("\(key).m4a")
    }

    private var indexURL: URL {
        rootURL.appendingPathComponent("index.json")
    }

    private func persistIndex() {
        guard indexDirty else { return }
        indexDirty = false
        if let data = try? JSONEncoder().encode(index) {
            try? data.write(to: indexURL, options: .atomic)
        }
    }

    /// Evict until disk usage ≤ `maxBytes`.
    ///
    /// Entries are scored by distance from `anchor`:
    /// - Different book:                  `Int.max` (always evict first)
    /// - Same book, cross-chapter:        `|Δchapter| * 1000 + |Δsentence|`
    /// - Same book, same chapter:         `|Δsentence|`
    /// - No coord in index (orphan):      `Int.max - 1`
    /// - No anchor set:                   fall back to FIFO (oldest first)
    ///
    /// Sort descending by score; drop from the top until under budget.
    private func evictIfNeeded() {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(
            at: rootURL,
            includingPropertiesForKeys: [.fileSizeKey, .creationDateKey, .isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return }

        struct FileEntry {
            let url: URL
            let key: String
            let size: Int
            let created: Date
        }

        // Treat every regular file in `rootURL` as a cache entry except
        // `index.json` (the eviction-coord index itself). Earlier this
        // filtered on `.wav` extension, but the cache actually writes
        // `.m4a` (see `fileURL`) — so the predicate matched nothing and
        // eviction never freed space. Same predicate as
        // `currentSizeBytes` so the Settings view's reported size and
        // the eviction set stay in sync.
        let files: [FileEntry] = entries.compactMap { url in
            if url.lastPathComponent == "index.json" { return nil }
            let values = try? url.resourceValues(
                forKeys: [.fileSizeKey, .creationDateKey, .isRegularFileKey],
            )
            guard values?.isRegularFile == true,
                  let size = values?.fileSize else { return nil }
            let created = values?.creationDate ?? .distantPast
            return FileEntry(
                url: url,
                key: url.deletingPathExtension().lastPathComponent,
                size: size,
                created: created,
            )
        }
        let total = files.reduce(0) { $0 + $1.size }
        guard total > maxBytes else { return }

        let ordered: [FileEntry]
        if let anchor = anchor {
            // Sort so evict-first (highest score) comes first.
            ordered = files.sorted { a, b in
                score(for: a.key, against: anchor) > score(for: b.key, against: anchor)
            }
        } else {
            // FIFO fallback: oldest first.
            ordered = files.sorted { $0.created < $1.created }
        }

        var remaining = total
        for file in ordered {
            if remaining <= maxBytes { break }
            try? fm.removeItem(at: file.url)
            index.removeValue(forKey: file.key)
            indexDirty = true
            remaining -= file.size
        }
    }

    /// Distance-based eviction score. Higher = evict sooner.
    private func score(for key: String, against anchor: SentenceCoord) -> Int {
        guard let coord = index[key] else {
            // Orphan file (no index entry — maybe index was lost, or the
            // file was written before index tracking existed). Prefer to
            // evict these over tracked entries.
            return Int.max - 1
        }
        if coord.bookId != anchor.bookId {
            return Int.max
        }
        let chapterDiff = abs(coord.chapterId - anchor.chapterId)
        let sentenceDiff = abs(coord.sentenceIndex - anchor.sentenceIndex)
        return chapterDiff * 1000 + sentenceDiff
    }
}
