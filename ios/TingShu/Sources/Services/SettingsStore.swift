// App-wide user preferences. Backed by UserDefaults for simplicity —
// these are per-device, not synced to the server (see §2.1.2).
import Foundation
import SwiftUI

@MainActor
final class SettingsStore: ObservableObject {
    // Persisted keys.
    private enum Keys {
        static let serverBaseURL = "serverBaseURL"
        static let serverUsername = "serverUsername"
        static let serverPassword = "serverPassword"
        static let darkMode = "darkMode"
        static let fontSize = "fontSize"
        static let gainDB = "gainDB"
        static let playbackSpeed = "playbackSpeed"
        static let leftHandMode = "leftHandMode"
        static let audioCacheLimitMB = "audioCacheLimitMB"
        static let narratorCharacterId = "narratorCharacterId"
    }

    /// Predefined narrator voices on the server (must match
    /// ``server/app/core/narrator.py:NARRATOR_SPEAKERS``). The settings
    /// page uses this to render a picker; ``PlaybackService`` reads
    /// ``narratorCharacterId`` to substitute for sentences whose
    /// ``character_id`` is 0.
    static let narratorOptions: [(id: Int, name: String)] = [
        (0, "男旁白"),
        (1, "女旁白"),
    ]

    @Published var serverBaseURL: String {
        didSet { UserDefaults.standard.set(serverBaseURL, forKey: Keys.serverBaseURL) }
    }
    /// Bearer-auth credentials. The wire format is
    /// ``Authorization: Bearer <base64(username:password)>`` — we
    /// compute the token offline at request time (see
    /// ``APIClient.applyAuth``). Empty strings mean "no auth"; the
    /// server side mirrors this with ``auth.enabled=false``.
    @Published var serverUsername: String {
        didSet { UserDefaults.standard.set(serverUsername, forKey: Keys.serverUsername) }
    }
    @Published var serverPassword: String {
        didSet { UserDefaults.standard.set(serverPassword, forKey: Keys.serverPassword) }
    }
    @Published var darkMode: Int {  // 0=system, 1=dark, 2=light
        didSet { UserDefaults.standard.set(darkMode, forKey: Keys.darkMode) }
    }
    @Published var fontSize: Int {
        didSet { UserDefaults.standard.set(fontSize, forKey: Keys.fontSize) }
    }
    @Published var gainDB: Double {
        didSet { UserDefaults.standard.set(gainDB, forKey: Keys.gainDB) }
    }
    @Published var playbackSpeed: Double {
        didSet { UserDefaults.standard.set(playbackSpeed, forKey: Keys.playbackSpeed) }
    }
    @Published var leftHandMode: Bool {
        didSet { UserDefaults.standard.set(leftHandMode, forKey: Keys.leftHandMode) }
    }
    @Published var audioCacheLimitMB: Int {
        didSet { UserDefaults.standard.set(audioCacheLimitMB, forKey: Keys.audioCacheLimitMB) }
    }
    /// Which predefined narrator voice to use for sentences whose
    /// ``character_id`` is 0 (旁白). Default 0 = male narrator. The
    /// PlaybackService substitutes this id into TTS requests; the
    /// server has the actual speaker_id mapping.
    @Published var narratorCharacterId: Int {
        didSet { UserDefaults.standard.set(narratorCharacterId, forKey: Keys.narratorCharacterId) }
    }

    init() {
        let d = UserDefaults.standard
        // Default to localhost for simulator + same-machine dev runs. On a
        // real device, users must change this to their Mac's LAN IP in
        // Settings. The Gemini proxy address in server/config.yaml is
        // unrelated — don't confuse it with the server's own address.
        self.serverBaseURL = d.string(forKey: Keys.serverBaseURL) ?? "http://localhost:8000"
        self.serverUsername = d.string(forKey: Keys.serverUsername) ?? ""
        self.serverPassword = d.string(forKey: Keys.serverPassword) ?? ""
        self.darkMode = d.integer(forKey: Keys.darkMode)
        self.fontSize = d.object(forKey: Keys.fontSize) as? Int ?? 18
        self.gainDB = d.object(forKey: Keys.gainDB) as? Double ?? 0.0
        self.playbackSpeed = d.object(forKey: Keys.playbackSpeed) as? Double ?? 1.0
        self.leftHandMode = d.bool(forKey: Keys.leftHandMode)
        self.audioCacheLimitMB = d.object(forKey: Keys.audioCacheLimitMB) as? Int ?? 500
        self.narratorCharacterId = d.object(forKey: Keys.narratorCharacterId) as? Int ?? 0
    }

    var parsedServerURL: URL? { URL(string: serverBaseURL) }

    var colorScheme: ColorScheme? {
        switch darkMode {
        case 1: return .dark
        case 2: return .light
        default: return nil
        }
    }

    var appVersion: String {
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
        let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?"
        return "\(version) (\(build))"
    }
}
