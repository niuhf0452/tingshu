// Per-book character roster — pushed from the "修改角色音色" row in
// PlayerSettingsView. Search bar pinned to the top of the list (not
// the floating navigation-bar drawer style), tap a row to push the
// edit view.
//
// State-driven push (``navigationDestination(item:)``) is used instead
// of per-row NavigationLink because:
// - The root NavigationStack in BookshelfView is path-typed ``[String]``,
//   so ``NavigationLink(value: Character)`` can't find a destination
//   inside this pushed view.
// - The legacy ``NavigationLink { destination } label:`` form in a
//   Form + ForEach combination was eagerly building one destination
//   per row and routing taps to the wrong row's destination instance.
import SwiftUI

struct CharacterListView: View {
    let book: LocalBook

    @EnvironmentObject var playback: PlaybackService

    @State private var characters: [Character] = []
    @State private var loadState: LoadState = .idle
    @State private var searchText = ""
    @State private var editingCharacter: Character?

    private enum LoadState: Equatable {
        case idle
        case loading
        case loaded
        case failed(String)
    }

    var body: some View {
        VStack(spacing: 0) {
            SearchField(text: $searchText, placeholder: "搜索角色")
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(.bar)
            content
        }
        .navigationTitle("修改角色音色")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .navigationDestination(item: $editingCharacter) { character in
            CharacterEditView(book: book, character: character) { updated in
                applyUpdated(updated)
            }
        }
        .task { await loadCharacters() }
    }

    @ViewBuilder
    private var content: some View {
        switch loadState {
        case .idle, .loading:
            VStack {
                Spacer()
                ProgressView()
                Spacer()
            }
        case .failed(let message):
            VStack(spacing: 8) {
                Spacer()
                Text("加载失败").foregroundStyle(.secondary)
                Text(message)
                    .font(.caption).foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)
                Button("重试") { Task { await loadCharacters() } }
                    .buttonStyle(.borderedProminent)
                Spacer()
            }
        case .loaded:
            if characters.isEmpty {
                VStack(spacing: 4) {
                    Spacer()
                    Text("尚未识别到角色").foregroundStyle(.secondary)
                    Text("章节分析后会出现在这里")
                        .font(.caption).foregroundStyle(.tertiary)
                    Spacer()
                }
            } else {
                listContent
            }
        }
    }

    private var listContent: some View {
        let filtered = filteredCharacters
        return List {
            if filtered.isEmpty {
                Text("没有匹配 “\(searchText)” 的角色")
                    .foregroundStyle(.secondary)
            } else {
                ForEach(filtered) { character in
                    Button {
                        editingCharacter = character
                    } label: {
                        HStack {
                            CharacterRow(character: character)
                            Spacer()
                            Image(systemName: "chevron.right")
                                .font(.footnote.weight(.semibold))
                                .foregroundStyle(.tertiary)
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        #if os(iOS)
        .listStyle(.insetGrouped)
        #endif
    }

    private var filteredCharacters: [Character] {
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return characters }
        return characters.filter { $0.name.localizedCaseInsensitiveContains(query) }
    }

    private func applyUpdated(_ updated: Character) {
        if let i = characters.firstIndex(where: { $0.id == updated.id }) {
            characters[i] = updated
        }
    }

    private func loadCharacters() async {
        loadState = .loading
        do {
            let list = try await playback.api.bookCharacters(bookId: book.bookId)
            characters = list.sorted(by: { $0.id < $1.id })
            loadState = .loaded
        } catch {
            loadState = .failed(
                (error as? LocalizedError)?.errorDescription
                ?? String(describing: error),
            )
        }
    }
}

// MARK: - search field

/// Compact pill search input. Built locally instead of using
/// ``.searchable`` because the user wants the field pinned to the top
/// of the list, not floating in / out with the nav bar drawer.
private struct SearchField: View {
    @Binding var text: String
    let placeholder: String

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
            field
            if !text.isEmpty {
                Button {
                    text = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(searchFieldBackground)
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    @ViewBuilder
    private var field: some View {
        let tf = TextField(placeholder, text: $text)
            .textFieldStyle(.plain)
            .autocorrectionDisabled()
        #if os(iOS)
        tf.textInputAutocapitalization(.never)
        #else
        tf
        #endif
    }

    private var searchFieldBackground: Color {
        #if canImport(UIKit)
        return Color(uiColor: .secondarySystemBackground)
        #else
        return Color.gray.opacity(0.15)
        #endif
    }
}

// MARK: - shared character display

/// Two-line character cell. Shared with the in-edit-view header (top
/// section showing read-only name + identity) — reusing one struct
/// keeps the visual identity of a character consistent across the
/// list, edit, and any future "now playing" surface.
struct CharacterRow: View {
    let character: Character

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(character.name).font(.body)
            HStack(spacing: 8) {
                Text(genderLabel(character.gender))
                Text("·")
                Text(ageLabel(character.age))
                if !character.personality.isEmpty {
                    Text("·")
                    Text(character.personality
                        .map(personalityLabel).joined(separator: "/"))
                        .lineLimit(1)
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            if !character.identity.isEmpty {
                Text(character.identity)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .lineLimit(2)
            }
        }
    }
}

// MARK: - i18n helpers

func genderLabel(_ g: Gender) -> String {
    switch g {
    case .male: return "男"
    case .female: return "女"
    case .neutral: return "中性"
    }
}

func ageLabel(_ a: Age) -> String {
    switch a {
    case .child: return "儿童"
    case .teen: return "少年"
    case .youth: return "青年"
    case .adult: return "成人"
    case .elder: return "老年"
    }
}

func personalityLabel(_ p: Personality) -> String {
    switch p {
    case .calm: return "沉稳"
    case .gentle: return "温柔"
    case .cheerful: return "开朗"
    case .serious: return "严肃"
    case .cold: return "冷淡"
    case .fierce: return "凶悍"
    case .determined: return "坚定"
    case .timid: return "怯弱"
    case .playful: return "顽皮"
    case .mature: return "成熟"
    case .naive: return "天真"
    case .wise: return "睿智"
    case .arrogant: return "傲慢"
    case .kind: return "和善"
    case .cunning: return "狡黠"
    case .brave: return "勇敢"
    case .melancholy: return "忧郁"
    case .passionate: return "热情"
    }
}
