// Edit one book character's matcher inputs (gender / age / personality).
// Pushed from the player-settings character list.
//
// Save semantics:
// - Only fields the user actually changed are sent in the PATCH body.
//   Identity / name are read-only here — the LLM uses them to recognise
//   the character across chapters, so a user rename would silently
//   break re-analysis. Voice selection is fully driven by the matcher
//   inputs, so they're the only useful editable fields.
// - The server takes the per-book lock during save, sharing it with
//   chapter analysis. This view's save call therefore can wait briefly
//   on contention but won't fail — the spec calls for "wait, never
//   lose data".
import SwiftUI

struct CharacterEditView: View {
    let book: LocalBook
    /// The roster value as the row was tapped. Acts as the baseline for
    /// the diff we send: only fields whose draft differs from this go
    /// into the PATCH.
    let original: Character
    /// Called with the server's authoritative response so the parent
    /// list refreshes its cached row without a round-trip GET.
    let onSaved: (Character) -> Void

    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var playback: PlaybackService

    @State private var draftGender: Gender
    @State private var draftAge: Age
    /// Personality is multi-select. Use a Set for O(1) toggle and to
    /// dedupe — the wire format is a list, but the order doesn't carry
    /// semantic meaning (the matcher uses set intersection).
    @State private var draftPersonality: Set<Personality>
    @State private var saveState: SaveState = .idle
    @State private var errorMessage: String?

    private enum SaveState: Equatable {
        case idle
        case saving
    }

    init(
        book: LocalBook, character: Character,
        onSaved: @escaping (Character) -> Void,
    ) {
        self.book = book
        self.original = character
        self.onSaved = onSaved
        _draftGender = State(initialValue: character.gender)
        _draftAge = State(initialValue: character.age)
        _draftPersonality = State(initialValue: Set(character.personality))
    }

    var body: some View {
        Form {
            // Read-only identity panel — gives context for the picks.
            Section {
                LabeledContent("名称", value: original.name)
                if !original.identity.isEmpty {
                    LabeledContent("身份") {
                        Text(original.identity)
                            .multilineTextAlignment(.trailing)
                    }
                }
            }

            Section("性别") {
                Picker("性别", selection: $draftGender) {
                    ForEach(Gender.allCases, id: \.self) { g in
                        Text(genderLabel(g)).tag(g)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }

            Section("年龄") {
                Picker("年龄", selection: $draftAge) {
                    ForEach(Age.allCases, id: \.self) { a in
                        Text(ageLabel(a)).tag(a)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }

            Section {
                ForEach(Personality.allCases, id: \.self) { p in
                    Button {
                        togglePersonality(p)
                    } label: {
                        HStack {
                            Text(personalityLabel(p))
                                .foregroundStyle(.primary)
                            Spacer()
                            if draftPersonality.contains(p) {
                                Image(systemName: "checkmark")
                                    .foregroundStyle(.tint)
                            }
                        }
                    }
                }
            } header: {
                Text("性格")
            } footer: {
                Text("可多选。匹配器以性格交集多者优先。")
            }

            if let err = errorMessage {
                Section {
                    Text(err)
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }
        }
        .navigationTitle(original.name)
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .toolbar {
            ToolbarItem(placement: .topTrailing) {
                Button(saveState == .saving ? "保存中…" : "保存") {
                    Task { await save() }
                }
                .disabled(!hasChanges || saveState == .saving)
            }
        }
        .interactiveDismissDisabled(saveState == .saving)
    }

    private func togglePersonality(_ p: Personality) {
        if draftPersonality.contains(p) {
            draftPersonality.remove(p)
        } else {
            draftPersonality.insert(p)
        }
    }

    private var hasChanges: Bool {
        draftGender != original.gender
            || draftAge != original.age
            || draftPersonality != Set(original.personality)
    }

    /// Build a partial update with **only** the fields that differ —
    /// matches the spec ("保存时只传递修改的角色信息"). For personality
    /// we send the full new list when it changed (set vs set), since a
    /// partial diff (added/removed) doesn't fit the server's set-
    /// replace semantics.
    private func buildUpdate() -> CharacterUpdate {
        var update = CharacterUpdate()
        if draftGender != original.gender { update.gender = draftGender }
        if draftAge != original.age { update.age = draftAge }
        if draftPersonality != Set(original.personality) {
            // Sorted for a stable wire payload (helps server logs and
            // any test that asserts on the request body).
            update.personality = draftPersonality.sorted { $0.rawValue < $1.rawValue }
        }
        return update
    }

    private func save() async {
        guard hasChanges else { return }
        saveState = .saving
        errorMessage = nil
        let update = buildUpdate()
        do {
            let updated = try await playback.api.updateBookCharacter(
                bookId: book.bookId,
                characterId: original.id,
                update: update,
            )
            // Server has new attrs → matcher will pick a different
            // speaker on the next TTS request, so the locally cached
            // audio for this book is stale. Drop it (and the in-memory
            // prefetch URLs that reference it) before the user goes
            // back to the player — otherwise the next sentence either
            // replays the old voice or crashes on a missing file.
            await playback.invalidateBookAudio(bookId: book.bookId)
            onSaved(updated)
            saveState = .idle
            dismiss()
        } catch {
            errorMessage = (error as? LocalizedError)?.errorDescription
                ?? String(describing: error)
            saveState = .idle
        }
    }
}
