// HTTP client for the TingShu server.
//
// All endpoints live under the user-configured base URL (see SettingsView —
// default is the Mac Mini on the LAN). Everything is async/await on top of
// URLSession. No retry logic at this layer — PlaybackService layers its own
// retry over TTS calls; book downloads use HTTP Range via URLSession's
// built-in resume data when the user retries.
import Foundation

enum APIError: Error, LocalizedError {
    case invalidBaseURL
    case server(status: Int, body: String)
    case network(underlying: Error)
    case decoding(underlying: Error)
    case badResponse

    var errorDescription: String? {
        // No truncation, no UX softening — surface the raw underlying
        // detail so debugging doesn't require reading the server log.
        switch self {
        case .invalidBaseURL:
            return "APIError.invalidBaseURL — 服务端地址不合法"
        case .server(let status, let body):
            return "APIError.server status=\(status) body=\(body)"
        case .network(let underlying):
            let nsErr = underlying as NSError
            return "APIError.network domain=\(nsErr.domain) code=\(nsErr.code) "
                + "desc=\(underlying.localizedDescription) "
                + "userInfo=\(nsErr.userInfo)"
        case .decoding(let underlying):
            return "APIError.decoding type=\(type(of: underlying)) "
                + "desc=\(String(describing: underlying))"
        case .badResponse:
            return "APIError.badResponse — 响应格式异常或空流"
        }
    }
}

/// Snapshot of chapter-meta response. Post-refactor (§2.3) the body
/// itself includes the per-chapter character snapshot — no separate
/// version header to reconcile.
struct ChapterMetaResponse: Sendable {
    let meta: ChapterMeta
}

/// Snapshot of a TTS response — audio bytes + `X-Speaker-*` headers so the
/// cache key includes the matched speaker.
struct TTSResponse: Sendable {
    let audioData: Data
    let speakerId: String?
    let speakerGender: String?
    let speakerAge: String?
}

actor APIClient {
    private var baseURL: URL
    private var username: String
    private var password: String
    private let session: URLSession
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder

    init(baseURL: URL, username: String = "", password: String = "", session: URLSession = .shared) {
        self.baseURL = baseURL
        self.username = username
        self.password = password
        self.session = session
        self.decoder = JSONDecoder()
        self.encoder = JSONEncoder()
    }

    func updateBaseURL(_ url: URL) {
        self.baseURL = url
    }

    func updateCredentials(username: String, password: String) {
        self.username = username
        self.password = password
    }

    /// Attach the Bearer-auth header to ``request`` if credentials are
    /// set. Wire format: ``Authorization: Bearer <base64(user:pass)>`` —
    /// matches the server's ``app/api/auth.py`` decoder. Empty
    /// credentials = no header (server side has ``auth.enabled=false``
    /// in that case).
    private func applyAuth(_ request: inout URLRequest) {
        guard !username.isEmpty || !password.isEmpty else { return }
        let raw = "\(username):\(password)".data(using: .utf8) ?? Data()
        let token = raw.base64EncodedString()
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    }

    // MARK: - Books

    func listBooks() async throws -> [BookListItem] {
        // Book list is a fast call — don't let it wait the 60 s default.
        // If the server is unreachable the user should see a friendly
        // connection-failed state quickly, not a minute-long spinner.
        let response: BookListResponse = try await getJSON(path: "/api/books", timeout: 10)
        return response.books
    }

    /// Upload a book file. ``filename`` drives server-side format detection
    /// (.txt vs .epub) — pass the original name, not a sanitised one.
    func uploadBook(data: Data, filename: String) async throws -> UploadResponse {
        let url = try resolve(path: "/api/books/upload")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        applyAuth(&request)

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        body.appendString("--\(boundary)\r\n")
        body.appendString("Content-Disposition: form-data; name=\"file\"; filename=\"\(filename)\"\r\n")
        body.appendString("Content-Type: application/octet-stream\r\n\r\n")
        body.append(data)
        body.appendString("\r\n--\(boundary)--\r\n")
        request.httpBody = body

        return try await sendAndDecode(request: request)
    }

    /// Download the zipped book archive (meta.json + chapters/*.txt).
    /// Returns the zip bytes; caller unzips into the local library dir.
    func downloadBookArchive(bookId: String) async throws -> Data {
        let url = try resolve(path: "/api/books/\(bookId)/download")
        var request = URLRequest(url: url)
        applyAuth(&request)
        let (data, response) = try await session.data(for: request)
        try ensureOK(response: response, body: data)
        return data
    }

    /// Delete a book server-side. See docs/technical-plan.md §2.2.1 for
    /// cleanup semantics. Throws ``APIError.server`` with 404 if the book
    /// is already gone.
    func deleteBook(bookId: String) async throws {
        let url = try resolve(path: "/api/books/\(bookId)")
        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"
        request.timeoutInterval = 15
        applyAuth(&request)
        let (data, response) = try await session.data(for: request)
        try ensureOK(response: response, body: data)
    }

    // MARK: - Server-side TTS cache

    /// Wipe the server's TTS audio cache (``server/data/tts_cache/``).
    /// Returns when the server has finished — populated caches with
    /// tens of thousands of files can take several seconds, so the
    /// timeout is generous. The directory survives; only ``.m4a`` and
    /// stale ``.tmp`` files are removed.
    func clearServerTTSCache() async throws {
        let url = try resolve(path: "/api/tts/cache")
        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"
        request.timeoutInterval = 60
        applyAuth(&request)
        let (data, response) = try await session.data(for: request)
        try ensureOK(response: response, body: data)
    }

    // MARK: - Book characters (cumulative roster — see server §2.3)

    /// List the cumulative book character roster (narrator slots
    /// filtered out server-side). Empty list if no chapter has been
    /// analysed yet — the player-settings screen surfaces that as "尚
    /// 无识别到的角色". 404 only on unknown book.
    func bookCharacters(bookId: String) async throws -> [Character] {
        return try await getJSON(path: "/api/books/\(bookId)/characters", timeout: 10)
    }

    /// Patch one character's matcher inputs (gender / age / personality).
    /// Only the fields set on ``update`` are sent; the server preserves
    /// the rest. Server may block briefly on the per-book lock if a
    /// chapter analysis is mid-merge — never fails on contention.
    func updateBookCharacter(
        bookId: String, characterId: Int, update: CharacterUpdate,
    ) async throws -> Character {
        let url = try resolve(path: "/api/books/\(bookId)/characters/\(characterId)")
        var request = URLRequest(url: url)
        request.httpMethod = "PATCH"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        // Match the server's worst-case lock-wait (a chapter merge
        // typically completes in <1 s but the surrounding LLM calls
        // outside the lock can stretch the perceived wait).
        request.timeoutInterval = 30
        applyAuth(&request)
        request.httpBody = try encoder.encode(update)
        return try await sendAndDecode(request: request)
    }

    // MARK: - Chapter metadata (SSE)

    /// Consume the chapter-meta SSE stream and return the final
    /// ``meta`` event's payload. The server emits ``heartbeat`` events
    /// every ~5s while LLM analysis runs, then a single ``meta``
    /// (success) or ``error`` (failure) event — see
    /// ``server/app/api/chapter_meta_stream.py`` for the wire format.
    ///
    /// The 300 s timeout is the **inter-byte** gap, not the total wall
    /// clock. Heartbeats reset the timer; only a genuinely silent server
    /// trips it. LLM jobs in practice complete in 30–60 s.
    func chapterMeta(bookId: String, chapterId: Int) async throws -> ChapterMetaResponse {
        let url = try resolve(path: "/api/books/\(bookId)/chapters/\(chapterId)/meta")
        var request = URLRequest(url: url)
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.timeoutInterval = 300
        applyAuth(&request)

        let (bytes, response) = try await session.bytes(for: request)
        // 4xx errors (404 book/chapter not found) come back as regular
        // JSON, not SSE. Read the body and surface the detail.
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard 200..<300 ~= http.statusCode else {
            var body = Data()
            for try await byte in bytes { body.append(byte) }
            let bodyString = String(data: body, encoding: .utf8) ?? "<non-utf8>"
            throw APIError.server(status: http.statusCode, body: bodyString)
        }

        var eventName = "message"
        var dataLines: [String] = []

        // Try to dispatch a pending event. Returns the parsed
        // ChapterMetaResponse on a meta event, or throws on error.
        // Returns nil for heartbeats / unknowns so the loop can keep
        // going.
        func dispatch() throws -> ChapterMetaResponse? {
            defer {
                eventName = "message"
                dataLines = []
            }
            let payload = dataLines.joined(separator: "\n")
            switch eventName {
            case "meta":
                guard let payloadData = payload.data(using: .utf8) else {
                    throw APIError.badResponse
                }
                do {
                    let meta = try decoder.decode(ChapterMeta.self, from: payloadData)
                    return ChapterMetaResponse(meta: meta)
                } catch {
                    throw APIError.decoding(underlying: error)
                }
            case "error":
                let detail = parseSSEErrorDetail(payload) ?? payload
                throw APIError.server(status: 200, body: detail)
            case "heartbeat":
                // Liveness signal — discard.
                return nil
            default:
                // Unknown event name: ignore (forward-compat with
                // future server-side events like "progress").
                return nil
            }
        }

        // **Why we don't use `bytes.lines`**: iOS's AsyncLineSequence
        // does NOT yield empty lines between consecutive line
        // terminators, so the `\n\n` event separator in SSE silently
        // disappears. Two adjacent events' `data:` lines then accumulate
        // into one payload, and the meta event's JSONDecoder rejects it
        // with "Unexpected character '{' after top-level value around
        // line 2, column 1". We split on `\n` ourselves to preserve the
        // blank lines that delimit events. (Also handles `\r\n` by
        // trimming a trailing CR per byte.)
        var byteBuffer: [UInt8] = []
        for try await byte in bytes {
            if byte == 0x0A {  // '\n'
                if byteBuffer.last == 0x0D {  // strip trailing '\r'
                    byteBuffer.removeLast()
                }
                let line = String(bytes: byteBuffer, encoding: .utf8) ?? ""
                byteBuffer.removeAll(keepingCapacity: true)
                if line.isEmpty {
                    if let resp = try dispatch() { return resp }
                } else if line.hasPrefix("event:") {
                    eventName = String(line.dropFirst("event:".count))
                        .trimmingCharacters(in: .whitespaces)
                } else if line.hasPrefix("data:") {
                    // SSE: leading single space after `data:` is part of
                    // the delimiter, not the payload. Drop one space if
                    // present.
                    var rest = Substring(line.dropFirst("data:".count))
                    if rest.first == " " { rest = rest.dropFirst() }
                    dataLines.append(String(rest))
                }
                // Other line types (id:, retry:, comment lines) ignored.
            } else {
                byteBuffer.append(byte)
            }
        }

        // Trailing un-terminated bytes (no final \n) — treat as one last
        // line so a server-correct event of the form `event: X\ndata:
        // Y\n\n` doesn't strand the final event.
        if !byteBuffer.isEmpty {
            if byteBuffer.last == 0x0D { byteBuffer.removeLast() }
            let line = String(bytes: byteBuffer, encoding: .utf8) ?? ""
            if line.hasPrefix("data:") {
                var rest = Substring(line.dropFirst("data:".count))
                if rest.first == " " { rest = rest.dropFirst() }
                dataLines.append(String(rest))
            } else if line.hasPrefix("event:") {
                eventName = String(line.dropFirst("event:".count))
                    .trimmingCharacters(in: .whitespaces)
            }
        }
        if !dataLines.isEmpty || eventName != "message" {
            if let resp = try dispatch() { return resp }
        }

        // Truly empty stream — server cut us off before any event.
        throw APIError.badResponse
    }

    /// Pull the ``detail`` field out of an SSE error payload, falling
    /// back to nil so the caller can surface the raw JSON.
    private func parseSSEErrorDetail(_ payload: String) -> String? {
        guard let data = payload.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let detail = obj["detail"] as? String else { return nil }
        return detail
    }

    // MARK: - TTS

    func synthesize(request tts: TTSRequest) async throws -> TTSResponse {
        let url = try resolve(path: "/api/tts")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 60
        applyAuth(&request)
        request.httpBody = try encoder.encode(tts)

        let (data, response) = try await session.data(for: request)
        try ensureOK(response: response, body: data)
        let http = response as? HTTPURLResponse
        return TTSResponse(
            audioData: data,
            speakerId: http?.value(forHTTPHeaderField: "X-Speaker-Id"),
            speakerGender: http?.value(forHTTPHeaderField: "X-Speaker-Gender"),
            speakerAge: http?.value(forHTTPHeaderField: "X-Speaker-Age")
        )
    }

    // MARK: - helpers

    private func resolve(path: String) throws -> URL {
        guard let url = URL(string: path, relativeTo: baseURL)?.absoluteURL else {
            throw APIError.invalidBaseURL
        }
        return url
    }

    private func getJSON<T: Decodable>(path: String, timeout: TimeInterval? = nil) async throws -> T {
        let url = try resolve(path: path)
        var request = URLRequest(url: url)
        if let timeout = timeout { request.timeoutInterval = timeout }
        applyAuth(&request)
        let (data, response) = try await session.data(for: request)
        try ensureOK(response: response, body: data)
        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            throw APIError.decoding(underlying: error)
        }
    }

    private func sendAndDecode<T: Decodable>(request: URLRequest) async throws -> T {
        let (data, response) = try await session.data(for: request)
        try ensureOK(response: response, body: data)
        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            throw APIError.decoding(underlying: error)
        }
    }

    private func ensureOK(response: URLResponse, body: Data) throws {
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard 200..<300 ~= http.statusCode else {
            let bodyString = String(data: body, encoding: .utf8) ?? "<non-utf8 body>"
            throw APIError.server(status: http.statusCode, body: bodyString)
        }
    }
}

// MARK: - Data helpers

private extension Data {
    mutating func appendString(_ string: String) {
        append(Data(string.utf8))
    }
}
