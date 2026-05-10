// Player-page settings — pushed from the gear button in the player's
// bottom bar. Two sections directly on this page:
//
// - "播放" — the subset of app-wide settings worth surfacing during
//   active playback (volume gain, speed, narrator voice). Bound
//   directly to ``SettingsStore`` so changes here behave identically
//   to the same controls in the bookshelf settings page.
// - "显示" — font size, same source of truth.
//
// Per-book character voice editing lives on its own pushed page
// (``CharacterListView``) reached via the "修改角色音色" row. The
// roster can be hundreds of names long; a dedicated screen with a
// pinned search bar + full-bleed list is more usable than crammed
// into a Form section.
import SwiftUI

struct PlayerSettingsView: View {
    let book: LocalBook

    @EnvironmentObject var settings: SettingsStore
    @EnvironmentObject var playback: PlaybackService

    var body: some View {
        Form {
            playbackSection
            displaySection
            bookSection
        }
        .navigationTitle("播放设置")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
    }

    private var playbackSection: some View {
        Section("播放") {
            VStack(alignment: .leading) {
                Text("音量增益：\(Int(settings.gainDB)) dB")
                Slider(value: $settings.gainDB, in: -12...20, step: 1)
            }
            VStack(alignment: .leading) {
                Text("语速：\(String(format: "%.1f", settings.playbackSpeed))x")
                Slider(value: $settings.playbackSpeed, in: 0.5...2.0, step: 0.1)
            }
            Picker("旁白音色", selection: $settings.narratorCharacterId) {
                ForEach(SettingsStore.narratorOptions, id: \.id) { option in
                    Text(option.name).tag(option.id)
                }
            }
        }
    }

    private var displaySection: some View {
        Section("显示") {
            Stepper(
                "字号：\(settings.fontSize)",
                value: $settings.fontSize, in: 12...30, step: 2
            )
        }
    }

    private var bookSection: some View {
        // Single static link — legacy ``NavigationLink { dest } label:``
        // is fine here. The eager-destination quirk that bit the old
        // per-row character list doesn't reproduce when there's only
        // one link with a fixed identity in this view.
        Section("本书") {
            NavigationLink {
                CharacterListView(book: book)
            } label: {
                Label("修改角色音色", systemImage: "person.wave.2")
            }
        }
    }
}
