// Root screen: shows the user's library (books imported into their
// personal server), provides import + pull-to-refresh, and navigates
// to the player on tap.
//
// Design spec: docs/technical-plan.md §3.3.
import SwiftUI
import UniformTypeIdentifiers

struct BookshelfView: View {
    @EnvironmentObject var store: BookStore
    @Environment(\.scenePhase) private var scenePhase
    @State private var showImporter = false
    @State private var showSettings = false
    @State private var path: [String] = []
    @State private var refreshTrigger = 0
    @State private var bookPendingDelete: ShelfBook?

    var body: some View {
        NavigationStack(path: $path) {
            bookshelfContent
                .navigationTitle("听书")
                .toolbar {
                    ToolbarItem(placement: .topLeading) {
                        Button { showSettings = true } label: {
                            Image(systemName: "gearshape")
                        }
                    }
                    // No global sync spinner — by design (see
                    // docs/technical-plan.md §3.3 "进行中状态的指示").
                    // Background refreshes are silent; the only
                    // in-progress signal is per-row, on the books that
                    // are actually uploading / processing / downloading.
                    // Pull-to-refresh keeps SwiftUI's built-in spinner.
                    ToolbarItem(placement: .topTrailing) {
                        Button { showImporter = true } label: {
                            Image(systemName: "plus")
                        }
                    }
                }
                .sheet(isPresented: $showSettings) { SettingsView() }
                .fileImporter(
                    isPresented: $showImporter,
                    allowedContentTypes: importContentTypes,
                    allowsMultipleSelection: false,
                    onCompletion: handleImport
                )
                .navigationDestination(for: String.self) { bookId in
                    PlayerLoader(bookId: bookId)
                }
                .refreshable { await store.refresh() }
                .task(id: refreshTrigger) { await store.refresh() }
                .onChange(of: scenePhase) { oldPhase, newPhase in
                    // Only refresh when transitioning *into* `.active`
                    // (e.g. backgrounded → foreground). The `path.isEmpty`
                    // gate ensures we don't refresh when the player view
                    // is on top — the user wants the bookshelf-only
                    // semantics (per docs/technical-plan.md §3.3
                    // "刷新触发条件"), and refreshing while the user is
                    // actively reading risks the zombie-cleanup pass
                    // deleting the book they're listening to.
                    guard newPhase == .active,
                          oldPhase != .active,
                          path.isEmpty else { return }
                    Task { await store.refresh() }
                }
                .alert(
                    "错误",
                    isPresented: Binding(
                        get: { store.lastError != nil },
                        set: { if !$0 { store.lastError = nil } }
                    ),
                    actions: { Button("确定") { store.lastError = nil } },
                    message: { Text(store.lastError ?? "") }
                )
                .alert(
                    "删除《\(bookPendingDelete?.title ?? "")》？",
                    isPresented: Binding(
                        get: { bookPendingDelete != nil },
                        set: { if !$0 { bookPendingDelete = nil } }
                    ),
                    presenting: bookPendingDelete,
                    actions: { book in
                        Button("取消", role: .cancel) { bookPendingDelete = nil }
                        Button("删除", role: .destructive) {
                            let id = book.bookId
                            bookPendingDelete = nil
                            Task { await store.deleteBook(bookId: id) }
                        }
                    },
                    message: { _ in
                        Text("将同时删除服务端文件和本地下载。此操作无法撤销。")
                    }
                )
        }
    }

    @ViewBuilder
    private var bookshelfContent: some View {
        Group {
            if store.books.isEmpty {
                emptyState
            } else {
                VStack(spacing: 0) {
                    if let err = store.connectionError {
                        ConnectionErrorBanner(message: err) {
                            Task { await store.refresh() }
                        }
                    }
                    bookList
                }
            }
        }
        // App-wide warm light theme — matches the player page.
        .background(Color.readerBackground)
    }

    @ViewBuilder
    private var bookList: some View {
        List {
            ForEach(store.books) { book in
                BookRow(book: book)
                    .listRowBackground(Color.readerBackground)
                    .contentShape(Rectangle())
                    .onTapGesture { onTapBook(book) }
                    .swipeActions(edge: .trailing) {
                        Button(role: .destructive) {
                            bookPendingDelete = book
                        } label: {
                            Label("删除", systemImage: "trash")
                        }
                    }
            }
        }
        .listStyle(.plain)
        // Hide the List's default white-in-light-mode scroll background
        // so our `.background(Color.readerBackground)` shows through.
        // `listRowBackground` above keeps each row tinted too — without
        // it, only the gaps between rows would warm up.
        .scrollContentBackground(.hidden)
    }

    @ViewBuilder
    private var emptyState: some View {
        VStack(spacing: 12) {
            Spacer()
            if let err = store.connectionError {
                Image(systemName: "wifi.exclamationmark")
                    .font(.system(size: 60))
                    .foregroundStyle(.orange)
                Text("连接服务端失败")
                    .font(.title3)
                    .foregroundStyle(.primary)
                Text(err)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 32)
                Button {
                    Task { await store.refresh() }
                } label: {
                    Label("重试", systemImage: "arrow.clockwise")
                }
                .buttonStyle(.borderedProminent)
                .padding(.top, 8)
                Text("可在 设置 中修改服务端地址")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                Image(systemName: "books.vertical")
                    .font(.system(size: 60))
                    .foregroundStyle(.secondary)
                Text("书架为空")
                    .font(.title3)
                    .foregroundStyle(.secondary)
                Text("点击右上角 + 号导入 TXT 或 EPUB 文件")
                    .font(.footnote)
                    .foregroundStyle(.tertiary)
            }
            Spacer()
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }

    private var importContentTypes: [UTType] {
        var types: [UTType] = [.plainText]
        if let epub = UTType(filenameExtension: "epub") {
            types.append(epub)
        }
        return types
    }

    private func onTapBook(_ book: ShelfBook) {
        switch book.displayStatus {
        case .ready:
            path.append(book.bookId)
        case .processing, .uploading:
            store.lastError = "服务端正在处理该书，请稍后下拉刷新"
        case .downloading:
            store.lastError = "正在下载该书，请稍候"
        case .failed:
            store.lastError = "该书处理失败，请在服务端检查日志"
        }
    }

    private func handleImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }
            Task { await performUpload(url: url) }
        case .failure(let error):
            store.lastError = "选取文件失败：\(error.localizedDescription)"
        }
    }

    private func performUpload(url: URL) async {
        // FileImporter URLs come with a security scope.
        let needsStop = url.startAccessingSecurityScopedResource()
        defer { if needsStop { url.stopAccessingSecurityScopedResource() } }
        do {
            // Read the whole file off the main thread — a multi-MB TXT would
            // otherwise block the UI for several hundred ms on the simulator.
            let data = try await Task.detached(priority: .userInitiated) {
                try Data(contentsOf: url)
            }.value
            _ = await store.uploadBook(data: data, filename: url.lastPathComponent)
            refreshTrigger &+= 1
        } catch {
            store.lastError = "读取文件失败：\(error.localizedDescription)"
        }
    }
}

/// Loads a ``LocalBook`` off the main thread before handing it to
/// ``PlayerView``. Without this, the navigation destination closure
/// would do sync disk I/O on every re-render, which stacks up into
/// perceivable UI freezes for big books.
///
/// If the book isn't on disk (e.g. user navigated from stale state
/// before a delete propagated), we show an explicit error — we do NOT
/// fall back to `store.refresh()` which would make the player view
/// hang on network for up to 10s.
private struct PlayerLoader: View {
    @EnvironmentObject var store: BookStore
    @Environment(\.dismiss) private var dismiss
    let bookId: String
    @State private var book: LocalBook?
    @State private var loadFinished = false

    var body: some View {
        Group {
            if let book {
                PlayerView(book: book)
            } else if loadFinished {
                VStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.system(size: 44))
                        .foregroundStyle(.orange)
                    Text("书籍未下载到本机")
                        .font(.headline)
                    Text("请返回书架等待下载完成")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    Button("返回") { dismiss() }
                        .buttonStyle(.borderedProminent)
                        .padding(.top, 8)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ProgressView()
            }
        }
        .task(id: bookId) {
            book = await store.localBook(bookId: bookId)
            loadFinished = true
        }
    }
}

/// Thin banner shown above the book list when we have books but the
/// latest refresh failed (e.g. server briefly unreachable). Lets the
/// user retry without forcing a full-screen empty state.
private struct ConnectionErrorBanner: View {
    let message: String
    let retry: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "wifi.exclamationmark")
                .foregroundStyle(.orange)
            Text("连接服务端失败")
                .font(.footnote.weight(.medium))
            Spacer()
            Button("重试", action: retry)
                .font(.footnote.weight(.medium))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color.orange.opacity(0.12))
    }
}


private struct BookRow: View {
    let book: ShelfBook

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            coverPlaceholder
            VStack(alignment: .leading, spacing: 4) {
                Text(book.title)
                    .font(.headline)
                    .lineLimit(2)
                if !book.author.isEmpty {
                    Text(book.author)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                HStack(spacing: 6) {
                    statusBadge
                    if isInProgress {
                        ProgressView()
                            .progressViewStyle(.circular)
                            .controlSize(.mini)
                    }
                    if book.chapterCount > 0 {
                        Text("\(book.chapterCount) 章")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                    if let hint = inProgressHint {
                        Text(hint)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            Spacer()
            if book.displayStatus == .ready {
                Image(systemName: "chevron.right")
                    .font(.footnote)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(.vertical, 6)
    }

    private var isInProgress: Bool {
        switch book.displayStatus {
        case .uploading, .processing, .downloading: return true
        case .ready, .failed: return false
        }
    }

    private var inProgressHint: String? {
        switch book.displayStatus {
        case .uploading: return "正在上传并识别章节…"
        case .processing: return "服务端处理中"
        case .downloading: return "正在下载到本机"
        case .ready, .failed: return nil
        }
    }

    private var coverPlaceholder: some View {
        RoundedRectangle(cornerRadius: 6, style: .continuous)
            .fill(Color.accentColor.opacity(0.15))
            .frame(width: 56, height: 72)
            .overlay(
                Text(book.title.prefix(1))
                    .font(.title2.weight(.semibold))
                    .foregroundStyle(.tint)
            )
    }

    private var statusBadge: some View {
        let (label, color): (String, Color) = {
            switch book.displayStatus {
            case .ready: return ("已导入", .green)
            case .processing: return ("处理中", .orange)
            case .uploading: return ("上传中", .orange)
            case .downloading: return ("下载中", .blue)
            case .failed: return ("失败", .red)
            }
        }()
        return Text(label)
            .font(.caption2.weight(.medium))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15), in: Capsule())
            .foregroundStyle(color)
    }
}
