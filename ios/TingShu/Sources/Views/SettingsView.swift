import SwiftUI

struct SettingsView: View {
    /// How this view is being shown — drives whether we provide our own
    /// NavigationStack + "完成" toolbar button (sheet from the
    /// bookshelf) or rely on the host nav bar's back button (push from
    /// the player). Only affects chrome; the form body is identical.
    enum Presentation {
        case sheet
        case push
    }

    let presentation: Presentation

    @EnvironmentObject var settings: SettingsStore
    @EnvironmentObject var playback: PlaybackService
    @Environment(\.dismiss) private var dismiss

    /// Cached size readout. Refreshed in `.task` and after a clear.
    /// `nil` while the first read is in flight.
    @State private var cacheSizeBytes: Int? = nil
    @State private var showClearConfirm = false
    @State private var isClearing = false
    /// One-shot error blurb if the server-side wipe fails. The local
    /// part still happens regardless — surfacing the partial outcome
    /// is more useful than rolling it back.
    @State private var clearError: String?

    init(presentation: Presentation = .sheet) {
        self.presentation = presentation
    }

    var body: some View {
        switch presentation {
        case .sheet:
            // Modal entry — own NavigationStack so the title bar +
            // "完成" exist regardless of caller. Default for the
            // bookshelf gear button.
            NavigationStack {
                formContent
                    .navigationTitle("设置")
                    #if os(iOS)
                    .navigationBarTitleDisplayMode(.inline)
                    #endif
                    .toolbar {
                        ToolbarItem(placement: .topTrailing) {
                            Button("完成") { dismiss() }
                        }
                    }
            }
        case .push:
            // Pushed inside an existing NavigationStack (player view).
            // No extra NavigationStack — that would nest nav bars and
            // break the system back button. No "完成" — the host nav
            // bar already shows "‹ Back".
            formContent
                .navigationTitle("设置")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
        }
    }

    @ViewBuilder
    private var formContent: some View {
        Form {
            Section("连接") {
                let urlField = TextField("服务端地址", text: $settings.serverBaseURL)
                    .autocorrectionDisabled()
                #if os(iOS)
                urlField
                    .textInputAutocapitalization(.never)
                    .keyboardType(.URL)
                #else
                urlField
                #endif
                let userField = TextField("用户名", text: $settings.serverUsername)
                    .autocorrectionDisabled()
                #if os(iOS)
                userField
                    .textInputAutocapitalization(.never)
                #else
                userField
                #endif
                SecureField("密码", text: $settings.serverPassword)
            }

            Section("播放") {
                VStack(alignment: .leading) {
                    Text("音量增益：\(Int(settings.gainDB)) dB")
                    Slider(value: $settings.gainDB, in: -12...20, step: 1)
                }
                VStack(alignment: .leading) {
                    Text("语速：\(String(format: "%.1f", settings.playbackSpeed))x")
                    Slider(value: $settings.playbackSpeed, in: 0.5...2.0, step: 0.1)
                }
                Stepper(
                    "音频缓存上限：\(settings.audioCacheLimitMB) MB",
                    value: $settings.audioCacheLimitMB, in: 100...2000, step: 100
                )
                Picker("旁白音色", selection: $settings.narratorCharacterId) {
                    ForEach(SettingsStore.narratorOptions, id: \.id) { option in
                        Text(option.name).tag(option.id)
                    }
                }
            }

            Section("缓存") {
                HStack {
                    Text("已用空间")
                    Spacer()
                    Text(cacheSizeText)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
                Button(role: .destructive) {
                    showClearConfirm = true
                } label: {
                    if isClearing {
                        HStack {
                            ProgressView()
                            Text("清除中…")
                        }
                    } else {
                        Text("清除音频缓存")
                    }
                }
                .disabled(isClearing || (cacheSizeBytes ?? 0) == 0)
            }

            Section("显示") {
                Picker("暗色模式", selection: $settings.darkMode) {
                    Text("跟随系统").tag(0)
                    Text("深色").tag(1)
                    Text("浅色").tag(2)
                }
                Stepper(
                    "字号：\(settings.fontSize)",
                    value: $settings.fontSize, in: 12...30, step: 2
                )
                VStack(alignment: .leading, spacing: 8) {
                    Text("惯用手")
                    // Two-option binary state — segmented picker reads
                    // more directly than a toggle (a toggle's "on/off"
                    // doesn't map to a left/right concept). The
                    // underlying ``leftHandMode`` Bool stays unchanged
                    // so the player view's bar-order logic doesn't move.
                    Picker("惯用手", selection: $settings.leftHandMode) {
                        Text("左手").tag(true)
                        Text("右手").tag(false)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                }
            }

            Section("关于") {
                HStack {
                    Text("版本")
                    Spacer()
                    Text(settings.appVersion).foregroundStyle(.secondary)
                }
            }
        }
        .task { refreshCacheSize() }
        .confirmationDialog(
            "清除音频缓存",
            isPresented: $showClearConfirm,
            titleVisibility: .visible,
        ) {
            Button("仅本地", role: .destructive) {
                clearCache(includeServer: false)
            }
            Button("本地 + 服务端", role: .destructive) {
                clearCache(includeServer: true)
            }
            Button("取消", role: .cancel) { }
        } message: {
            Text("仅本地：删除本机已下载的音频文件。\n"
                 + "本地 + 服务端：同时清空服务端 data/tts_cache 下的所有合成结果。\n"
                 + "下次播放时需要重新合成。")
        }
        .alert(
            "清除失败",
            isPresented: Binding(
                get: { clearError != nil },
                set: { if !$0 { clearError = nil } },
            ),
            actions: { Button("好") { clearError = nil } },
            message: { Text(clearError ?? "") },
        )
    }

    /// Human-readable form of the cache size readout. Uses ByteCountFormatter
    /// so locale-correct units fall out for free (e.g. "12.3 MB" / "523 KB").
    private var cacheSizeText: String {
        guard let bytes = cacheSizeBytes else { return "—" }
        return ByteCountFormatter.string(
            fromByteCount: Int64(bytes), countStyle: .file,
        )
    }

    private func refreshCacheSize() {
        // currentSizeBytes is nonisolated; safe to call from MainActor.
        cacheSizeBytes = playback.cache.currentSizeBytes()
    }

    /// Wipe local cache, optionally also the server's. Local always
    /// runs first so a server failure (network down, auth wrong)
    /// doesn't leave the user with neither side cleared. The cleared
    /// state still reflects what actually happened.
    private func clearCache(includeServer: Bool) {
        isClearing = true
        Task {
            try? await playback.cache.clear()
            var failure: String?
            if includeServer {
                do {
                    try await playback.api.clearServerTTSCache()
                } catch {
                    failure = (error as? LocalizedError)?.errorDescription
                        ?? String(describing: error)
                }
            }
            await MainActor.run {
                refreshCacheSize()
                isClearing = false
                if let failure = failure {
                    clearError = "服务端清除失败（本地已清空）：\(failure)"
                }
            }
        }
    }
}
