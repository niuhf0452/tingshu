// Wire models that match `server/app/core/models.py`.
//
// Decoding is lenient — fields added server-side won't break older clients,
// and unknown enum values fall back to sane defaults (e.g. an unrecognised
// tone decodes to `.neutral`). The server owns the canonical shape; we
// accept its JSON verbatim.
import Foundation

struct ChapterEntry: Codable, Identifiable, Hashable, Sendable {
    let id: Int
    let title: String
    let textFile: String
    let metaFile: String

    enum CodingKeys: String, CodingKey {
        case id, title
        case textFile = "text_file"
        case metaFile = "meta_file"
    }
}

struct Character: Codable, Identifiable, Hashable, Sendable {
    let id: Int
    let name: String
    var identity: String
    var gender: Gender
    var age: Age
    var personality: [Personality]

    init(
        id: Int, name: String, identity: String = "",
        gender: Gender = .neutral, age: Age = .adult,
        personality: [Personality] = []
    ) {
        self.id = id
        self.name = name
        self.identity = identity
        self.gender = gender
        self.age = age
        self.personality = personality
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(Int.self, forKey: .id)
        name = try c.decode(String.self, forKey: .name)
        identity = (try? c.decode(String.self, forKey: .identity)) ?? ""
        gender = (try? c.decode(Gender.self, forKey: .gender)) ?? .neutral
        age = (try? c.decode(Age.self, forKey: .age)) ?? .adult
        personality = (try? c.decode([Personality].self, forKey: .personality)) ?? []
    }
}

struct BookMeta: Codable, Hashable, Sendable {
    let version: Int
    let bookId: String
    let title: String
    let author: String
    let cover: String?
    let summary: String
    let chapters: [ChapterEntry]
    let status: BookStatus
    let sourceFilename: String?

    enum CodingKeys: String, CodingKey {
        case version, title, author, cover, summary, chapters, status
        case bookId = "book_id"
        case sourceFilename = "source_filename"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        version = (try? c.decode(Int.self, forKey: .version)) ?? 1
        bookId = try c.decode(String.self, forKey: .bookId)
        title = try c.decode(String.self, forKey: .title)
        author = (try? c.decode(String.self, forKey: .author)) ?? ""
        cover = try? c.decode(String.self, forKey: .cover)
        summary = (try? c.decode(String.self, forKey: .summary)) ?? ""
        chapters = (try? c.decode([ChapterEntry].self, forKey: .chapters)) ?? []
        status = (try? c.decode(BookStatus.self, forKey: .status)) ?? .processing
        sourceFilename = try? c.decode(String.self, forKey: .sourceFilename)
    }
}

struct Sentence: Codable, Hashable, Sendable {
    let startLine: Int
    let startCol: Int
    let endLine: Int
    let endCol: Int
    let characterId: Int
    let tone: Tone

    enum CodingKeys: String, CodingKey {
        case startLine = "start_line"
        case startCol = "start_col"
        case endLine = "end_line"
        case endCol = "end_col"
        case characterId = "character_id"
        case tone
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        startLine = try c.decode(Int.self, forKey: .startLine)
        startCol = try c.decode(Int.self, forKey: .startCol)
        endLine = try c.decode(Int.self, forKey: .endLine)
        endCol = try c.decode(Int.self, forKey: .endCol)
        characterId = (try? c.decode(Int.self, forKey: .characterId)) ?? 0
        tone = (try? c.decode(Tone.self, forKey: .tone)) ?? .neutral
    }
}

struct ChapterMeta: Codable, Sendable {
    let sentences: [Sentence]
    /// Snapshot of every character that speaks in this chapter, with
    /// the full profile as of when the chapter was analysed (server-side
    /// §2.3). The App reads voices from here — no global character table
    /// to maintain.
    let characters: [Character]

    init(sentences: [Sentence] = [], characters: [Character] = []) {
        self.sentences = sentences
        self.characters = characters
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sentences = (try? c.decode([Sentence].self, forKey: .sentences)) ?? []
        characters = (try? c.decode([Character].self, forKey: .characters)) ?? []
    }

    enum CodingKeys: String, CodingKey {
        case sentences, characters
    }
}

// MARK: - API DTOs

struct BookListItem: Codable, Identifiable, Hashable, Sendable {
    let bookId: String
    let title: String
    let author: String
    let status: BookStatus
    let chapterCount: Int

    var id: String { bookId }

    enum CodingKeys: String, CodingKey {
        case bookId = "book_id"
        case title, author, status
        case chapterCount = "chapter_count"
    }
}

struct BookListResponse: Codable, Sendable {
    let books: [BookListItem]
}

struct UploadResponse: Codable, Sendable {
    let bookId: String
    let status: BookStatus
    let title: String
    let chapterCount: Int

    enum CodingKeys: String, CodingKey {
        case bookId = "book_id"
        case status, title
        case chapterCount = "chapter_count"
    }
}

/// Partial update for ``PATCH /api/books/{book_id}/characters/{id}``.
/// Only fields the user actually changed are sent — the server preserves
/// any field absent from the body. Mirrors ``CharacterUpdate`` on the
/// server (``server/app/core/models.py``).
struct CharacterUpdate: Codable, Sendable {
    var gender: Gender?
    var age: Age?
    var personality: [Personality]?

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encodeIfPresent(gender, forKey: .gender)
        try c.encodeIfPresent(age, forKey: .age)
        try c.encodeIfPresent(personality, forKey: .personality)
    }

    enum CodingKeys: String, CodingKey {
        case gender, age, personality
    }
}

struct TTSRequest: Codable, Sendable {
    let bookId: String
    let chapterId: Int
    /// Reserved range 0..15 for narrators; ≥16 for book characters.
    /// Server resolves this to a Speaker — the App never sends voice
    /// attributes (gender/age/personality) directly anymore.
    let characterId: Int
    let text: String
    let tone: Tone

    enum CodingKeys: String, CodingKey {
        case bookId = "book_id"
        case chapterId = "chapter_id"
        case characterId = "character_id"
        case text
        case tone
    }

    init(
        bookId: String,
        chapterId: Int,
        characterId: Int,
        text: String,
        tone: Tone = .neutral,
    ) {
        self.bookId = bookId
        self.chapterId = chapterId
        self.characterId = characterId
        self.text = text
        self.tone = tone
    }
}
