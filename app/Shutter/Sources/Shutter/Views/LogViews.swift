import AppKit
import SwiftUI

/// Reads only the end of the engine log, so a large log stays fast.
enum LogTail {
    static func lines(_ url: URL, max: Int = 50, bytes: UInt64 = 64 * 1024) -> [String] {
        guard let h = try? FileHandle(forReadingFrom: url) else { return [] }
        defer { try? h.close() }
        guard let end = try? h.seekToEnd() else { return [] }
        let start = end > bytes ? end - bytes : 0
        try? h.seek(toOffset: start)
        guard let data = try? h.readToEnd() else { return [] }
        var parts = String(decoding: data, as: UTF8.self).components(separatedBy: "\n")
        if start > 0, !parts.isEmpty { parts.removeFirst() }       // the first line may be cut
        while parts.last == "" { parts.removeLast() }
        return Array(parts.suffix(max))
    }
}

/// The last lines of the engine log in a fixed-height box, pinned to the newest line, refreshed every 2 s.
struct RecentLogBox: View {
    let fallback: [String]          // what this app captured when it started the engine
    @State private var lines: [String] = []
    private let timer = Timer.publish(every: 2, on: .main, in: .common).autoconnect()

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView([.vertical]) {
                LazyVStack(alignment: .leading, spacing: 2) {
                    if lines.isEmpty {
                        Text("No output yet. The log fills when the engine starts.")
                            .font(Theme.Fonts.row)
                            .foregroundStyle(Theme.Palette.muted)
                    }
                    ForEach(Array(lines.enumerated()), id: \.offset) { item in
                        Text(item.element.isEmpty ? " " : item.element)
                            .font(Theme.Fonts.mono)
                            .foregroundStyle(Theme.Palette.ink2)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .fixedSize(horizontal: false, vertical: true)
                            .id(item.offset)
                    }
                }
                .textSelection(.enabled)
                .padding(Theme.Space.s3)
            }
            .frame(height: 220)
            .background(Theme.Palette.chrome)
            .onChange(of: lines) { _, l in if !l.isEmpty { proxy.scrollTo(l.count - 1, anchor: .bottom) } }
            .onAppear { reload(); if !lines.isEmpty { proxy.scrollTo(lines.count - 1, anchor: .bottom) } }
            .onReceive(timer) { _ in reload() }
        }
    }

    private func reload() {
        let t = LogTail.lines(EngineProcess.logURL, max: 50)
        let new = t.isEmpty ? Array(fallback.suffix(50)) : t
        if new != lines { lines = new }
    }
}

/// The whole engine log, with search and View in Finder.
struct FullLogSheet: View {
    @Environment(\.dismiss) private var dismiss
    @State private var all: [String] = []
    @State private var query = ""

    private var shown: [String] {
        let q = query.trimmingCharacters(in: .whitespaces)
        return q.isEmpty ? all : all.filter { $0.localizedCaseInsensitiveContains(q) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s4) {
            HStack(spacing: Theme.Space.s3) {
                Text("Engine log")
                    .font(Theme.Fonts.title)
                    .foregroundStyle(Theme.Palette.ink)
                Spacer(minLength: Theme.Space.s3)
                TextField("", text: $query, prompt: Text("Search the log"))
                    .kitField()
                    .frame(width: 220)
            }
            Text("\(shown.count) of \(all.count) lines · \(EngineProcess.logURL.path)")
                .font(Theme.Fonts.meta)
                .foregroundStyle(Theme.Palette.muted)
                .lineLimit(1)
                .truncationMode(.middle)
            ScrollView([.vertical, .horizontal]) {
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(shown.enumerated()), id: \.offset) { item in
                        Text(item.element.isEmpty ? " " : item.element)
                            .font(Theme.Fonts.mono)
                            .foregroundStyle(Theme.Palette.ink2)
                            .fixedSize()
                    }
                }
                .textSelection(.enabled)
                .padding(Theme.Space.s3)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Theme.Palette.chrome)
            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous)
                .strokeBorder(Theme.Palette.line, lineWidth: Theme.Metric.hairline))
            HStack(spacing: Theme.Space.s3) {
                Button("View in Finder") { NSWorkspace.shared.activateFileViewerSelecting([EngineProcess.logURL]) }
                    .buttonStyle(.kit(.line))
                Button("Reload") { load() }
                    .buttonStyle(.kit(.line))
                Spacer(minLength: Theme.Space.s3)
                Button("Done") { dismiss() }
                    .buttonStyle(.kit(.ink))
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(Theme.Space.s6)
        .frame(minWidth: 760, idealWidth: 860, minHeight: 520, idealHeight: 620)
        .background(PaperBackground(glows: false))
        .onAppear(perform: load)
    }

    private func load() {
        let text = (try? String(contentsOf: EngineProcess.logURL, encoding: .utf8)) ?? ""
        var parts = text.components(separatedBy: "\n")
        while parts.last == "" { parts.removeLast() }
        all = parts
    }
}
