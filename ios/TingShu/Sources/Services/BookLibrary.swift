// Local book library layout (matches technical-plan.md §3.3.1):
//
//   <Documents>/library/<book_id>/
//     ├── meta.json
//     └── chapters/
//         ├── 0001.txt
//         ├── 0002.txt
//         └── ...
//
// Chapter metadata (chapters/*.json) is **not** persisted client-side —
// the App requests it lazily via APIClient.chapterMeta and holds the
// result in memory. Playback progress + user preferences live in
// SwiftData (not in this file).
//
// This layer is pure file IO. Syncing with the server (list/upload/
// download/refresh) is the job of BookStore above it.
import Foundation
import ZIPFoundation

struct LocalBook: Identifiable, Hashable, Sendable {
    let bookId: String
    var meta: BookMeta
    let dir: URL

    var id: String { bookId }
    var title: String { meta.title }
    var author: String { meta.author }
    var chapterCount: Int { meta.chapters.count }
}

enum BookLibraryError: Error {
    case missingMeta
    case invalidArchive(reason: String)
    case io(underlying: Error)
}

/// Reads/writes books under `<Documents>/library/`. All methods are
/// synchronous — callers should hop to a background queue for heavy work
/// (unzip / file scanning). We keep this type small and stateless so
/// SwiftUI / SwiftData can hold whatever cached views they need above it.
///
/// ``Sendable`` so @MainActor callers can hand it off to ``Task.detached``
/// for off-thread file work (unzip, bulk directory scans).
struct BookLibrary: Sendable {
    let rootURL: URL

    init(rootURL: URL) {
        self.rootURL = rootURL
        try? FileManager.default.createDirectory(
            at: rootURL, withIntermediateDirectories: true,
        )
        // A previous run may have been killed mid-install, leaving
        // `.<bookId>.installing.*` staging dirs on disk. They don't
        // affect correctness (hidden, so `listBookIds` skips them) but
        // eat space; clean them up here.
        cleanupStagingLeftovers()
    }

    static var defaultRoot: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("library", isDirectory: true)
    }

    // MARK: - discovery

    func listBookIds() -> [String] {
        let fm = FileManager.default
        let contents = (try? fm.contentsOfDirectory(
            at: rootURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        return contents
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            // Require `meta.json` — a directory without it is leftover
            // from a crashed install (pre-atomic-swap versions) and
            // would only confuse callers.
            .filter { fm.fileExists(atPath: $0.appendingPathComponent("meta.json").path) }
            .map { $0.lastPathComponent }
            .sorted()
    }

    func load(bookId: String) throws -> LocalBook {
        let dir = bookDir(bookId: bookId)
        let metaURL = dir.appendingPathComponent("meta.json")
        guard FileManager.default.fileExists(atPath: metaURL.path) else {
            throw BookLibraryError.missingMeta
        }
        let data = try Data(contentsOf: metaURL)
        let meta = try JSONDecoder().decode(BookMeta.self, from: data)
        return LocalBook(bookId: bookId, meta: meta, dir: dir)
    }

    /// Returns true if the book directory contains every chapter text
    /// file declared by `meta.json`. Prior non-atomic versions of
    /// ``installFromZip`` could leave a half-extracted book on disk
    /// (meta present but some chapters missing) — callers should treat
    /// incomplete books as "not local" so downloads re-fetch the full
    /// archive when the server is reachable.
    func isComplete(bookId: String) -> Bool {
        let dir = bookDir(bookId: bookId)
        let metaURL = dir.appendingPathComponent("meta.json")
        let fm = FileManager.default
        guard fm.fileExists(atPath: metaURL.path) else { return false }
        guard let data = try? Data(contentsOf: metaURL),
              let meta = try? JSONDecoder().decode(BookMeta.self, from: data) else {
            return false
        }
        for chapter in meta.chapters {
            let path = dir.appendingPathComponent(chapter.textFile).path
            if !fm.fileExists(atPath: path) {
                return false
            }
        }
        return true
    }

    func saveMeta(_ meta: BookMeta) throws {
        let dir = bookDir(bookId: meta.bookId)
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true,
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(meta)
        try data.write(to: dir.appendingPathComponent("meta.json"), options: .atomic)
    }

    // MARK: - chapter text

    func chapterText(bookId: String, chapter: ChapterEntry) throws -> String {
        let url = bookDir(bookId: bookId).appendingPathComponent(chapter.textFile)
        let data = try Data(contentsOf: url)
        return String(data: data, encoding: .utf8) ?? ""
    }

    // MARK: - install from server zip

    /// Expand the zip returned by `/api/books/{id}/download` into
    /// `library/<bookId>/`, **atomically**. The unzip goes into a
    /// per-install staging directory; once unpack succeeds and
    /// `meta.json` is confirmed present, we swap the staged directory
    /// into the final location with a single `moveItem`.
    ///
    /// If the process is killed mid-unzip (Xcode Stop, OS kill), the
    /// staging directory is left behind but the final `<bookId>/`
    /// directory is either the **previous** complete version or absent.
    /// Stale staging dirs are cleaned up at next init / first list call
    /// via ``cleanupStagingLeftovers``.
    func installFromZip(bookId: String, zipData: Data) throws {
        let dir = bookDir(bookId: bookId)
        let stagingDir = rootURL.appendingPathComponent(
            ".\(bookId).installing.\(UUID().uuidString)", isDirectory: true,
        )
        // On any exit path (including thrown errors from below), remove
        // the staging dir. The happy path's `moveItem` moves it out
        // before the defer fires, so the cleanup is a no-op.
        defer { try? FileManager.default.removeItem(at: stagingDir) }

        try FileManager.default.createDirectory(
            at: stagingDir, withIntermediateDirectories: true,
        )

        // Write the zip to a temp file (ZIPFoundation unzips from disk).
        let zipPath = stagingDir.appendingPathComponent("archive.zip")
        try zipData.write(to: zipPath, options: .atomic)

        let unpackedDir = stagingDir.appendingPathComponent("content", isDirectory: true)
        try FileManager.default.createDirectory(
            at: unpackedDir, withIntermediateDirectories: true,
        )
        do {
            try FileManager.default.unzipItem(at: zipPath, to: unpackedDir)
        } catch {
            throw BookLibraryError.invalidArchive(reason: error.localizedDescription)
        }

        // Verify the archive looked like a TingShu book. If the server
        // ever returns a truncated/malformed zip we fail loudly instead
        // of moving half a book into place.
        let metaAtStaging = unpackedDir.appendingPathComponent("meta.json")
        guard FileManager.default.fileExists(atPath: metaAtStaging.path) else {
            throw BookLibraryError.invalidArchive(reason: "meta.json missing in archive")
        }

        // Atomic swap. `moveItem` is not strictly rename(2) when crossing
        // volumes, but both paths live in the app's Documents dir so in
        // practice this is a single syscall.
        if FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.removeItem(at: dir)
        }
        try FileManager.default.moveItem(at: unpackedDir, to: dir)
    }

    /// Remove any `.<bookId>.installing.*` directories left behind by a
    /// previous install that crashed / was killed. Safe to call at any
    /// time — only matches the distinct staging prefix.
    func cleanupStagingLeftovers() {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(
            at: rootURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return }
        for entry in entries where entry.lastPathComponent.contains(".installing.") {
            try? fm.removeItem(at: entry)
        }
    }

    // MARK: - delete

    func delete(bookId: String) throws {
        let dir = bookDir(bookId: bookId)
        if FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.removeItem(at: dir)
        }
    }

    // MARK: - paths

    func bookDir(bookId: String) -> URL {
        rootURL.appendingPathComponent(bookId, isDirectory: true)
    }
}
