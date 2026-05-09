// App entrypoint. Wires the three shared stores + audio session and
// mounts the bookshelf as the root.
import SwiftUI
import AVFoundation
import MediaPlayer

@main
struct TingShuApp: App {
    @StateObject private var settings: SettingsStore
    @StateObject private var bookStore: BookStore
    @StateObject private var playback: PlaybackService
    private let api: APIClient
    private let cache: TTSCache

    init() {
        let settingsStore = SettingsStore()
        let baseURL = settingsStore.parsedServerURL ?? URL(string: "http://localhost:8000")!
        let client = APIClient(
            baseURL: baseURL,
            username: settingsStore.serverUsername,
            password: settingsStore.serverPassword,
        )
        let cacheLimitMB = settingsStore.audioCacheLimitMB
        let ttsCache = TTSCache(maxBytes: cacheLimitMB * 1024 * 1024)
        let progressStore = ProgressStore.makeDefault()
        let store = BookStore(
            api: client, settings: settingsStore, progressStore: progressStore,
        )

        _settings = StateObject(wrappedValue: settingsStore)
        _bookStore = StateObject(wrappedValue: store)
        _playback = StateObject(wrappedValue: PlaybackService(
            api: client, store: store, cache: ttsCache, settings: settingsStore,
            progressStore: progressStore,
        ))
        self.api = client
        self.cache = ttsCache

        #if canImport(UIKit)
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playback, mode: .default, options: [])
        try? session.setActive(true)
        UIApplication.shared.beginReceivingRemoteControlEvents()
        #endif
    }

    var body: some Scene {
        WindowGroup {
            BookshelfView()
                .environmentObject(settings)
                .environmentObject(bookStore)
                .environmentObject(playback)
                .preferredColorScheme(settings.colorScheme)
                .onChange(of: settings.serverBaseURL) { _, newValue in
                    guard let url = URL(string: newValue) else { return }
                    Task { await api.updateBaseURL(url) }
                }
                .onChange(of: settings.serverUsername) { _, newValue in
                    let pwd = settings.serverPassword
                    Task { await api.updateCredentials(username: newValue, password: pwd) }
                }
                .onChange(of: settings.serverPassword) { _, newValue in
                    let user = settings.serverUsername
                    Task { await api.updateCredentials(username: user, password: newValue) }
                }
        }
    }
}
