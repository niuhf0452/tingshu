// Per-book playback progress, persisted via SwiftData (§3.4).
//
// One row per book keyed by `book_id`. The plan deliberately picks the
// position-as-`(line, col)` form over a sentence id so the row survives
// chapter re-analysis (which can shift sentence count + indices).
import Foundation
import SwiftData

@Model
final class BookProgress {
    @Attribute(.unique) var bookId: String
    var chapterId: Int
    var startLine: Int
    var startCol: Int
    var updatedAt: Date

    init(bookId: String, chapterId: Int, startLine: Int, startCol: Int) {
        self.bookId = bookId
        self.chapterId = chapterId
        self.startLine = startLine
        self.startCol = startCol
        self.updatedAt = Date()
    }
}

/// Thin facade around SwiftData's `ModelContainer` for the playback
/// progress table. Main-actor confined because `ModelContext` must be
/// touched from the actor that owns it (we use the container's
/// `mainContext`).
@MainActor
final class ProgressStore {
    private let container: ModelContainer

    init(container: ModelContainer) {
        self.container = container
    }

    /// Convenience: build an on-disk container for production use.
    /// Falls back to an in-memory store if disk init fails so the app
    /// still runs (we'd rather lose progress than crash on launch).
    static func makeDefault() -> ProgressStore {
        let schema = Schema([BookProgress.self])
        let config = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)
        let container: ModelContainer
        do {
            container = try ModelContainer(for: schema, configurations: [config])
        } catch {
            let fallback = ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)
            // swiftlint:disable:next force_try
            container = try! ModelContainer(for: schema, configurations: [fallback])
        }
        return ProgressStore(container: container)
    }

    func load(bookId: String) -> DurablePosition? {
        let context = container.mainContext
        let target = bookId
        let descriptor = FetchDescriptor<BookProgress>(
            predicate: #Predicate { $0.bookId == target },
        )
        guard let row = try? context.fetch(descriptor).first else { return nil }
        return DurablePosition(
            chapterId: row.chapterId,
            startLine: row.startLine,
            startCol: row.startCol,
        )
    }

    func save(bookId: String, position: DurablePosition) {
        let context = container.mainContext
        let target = bookId
        let descriptor = FetchDescriptor<BookProgress>(
            predicate: #Predicate { $0.bookId == target },
        )
        if let row = try? context.fetch(descriptor).first {
            row.chapterId = position.chapterId
            row.startLine = position.startLine
            row.startCol = position.startCol
            row.updatedAt = Date()
        } else {
            context.insert(BookProgress(
                bookId: target,
                chapterId: position.chapterId,
                startLine: position.startLine,
                startCol: position.startCol,
            ))
        }
        try? context.save()
    }

    /// Remove the row for a book (called when the book is deleted, so the
    /// row doesn't outlive the data it points to).
    func clear(bookId: String) {
        let context = container.mainContext
        let target = bookId
        let descriptor = FetchDescriptor<BookProgress>(
            predicate: #Predicate { $0.bookId == target },
        )
        guard let row = try? context.fetch(descriptor).first else { return }
        context.delete(row)
        try? context.save()
    }
}
