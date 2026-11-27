import AppKit
import SwiftUI

/// The owner's Admin page: aggregated Help The Dev Improve reports. Only reachable when AdminStore.verified.
struct AdminView: View {
    @ObservedObject var admin = AdminStore.shared

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.s5) {
                HStack(alignment: .top) {
                    PageTitle(title: "Admin",
                              subtitle: subtitle)
                    Spacer(minLength: Theme.Space.s4)
                    KitSegmented(options: [KitOption(value: 7, label: "7 days"), KitOption(value: 30, label: "30 days")],
                                 selection: Binding(get: { admin.days }, set: { admin.setDays($0) }))
                        .fixedSize()
                    Button(admin.loading ? "Refreshing…" : "Refresh") { Task { await admin.refresh() } }
                        .buttonStyle(.kit(.line, small: true))
                        .disabled(admin.loading)
                }

                if let err = admin.lastError {
                    KitBanner(symbol: "wifi.exclamationmark", title: "Could not refresh", detail: err)
                }

                if let s = admin.summary {
                    installs(s)
                    models(s.models ?? [])
                    macs(s.macs ?? [])
                    versions(s.versions ?? [])
                    issues(s.issues ?? [])
                } else {
                    EmptyStateView(title: admin.loading ? "Loading reports…" : "No reports loaded",
                                   message: "Press Refresh to ask the telemetry server for the latest numbers.")
                }
            }
            .padding(.horizontal, Theme.Space.s6)
            .padding(.top, Theme.Space.s5)
            .padding(.bottom, Theme.Space.s6)
        }
        .task { await admin.refresh() }
    }

    private var subtitle: String {
        var s = "Anonymous reports from installs that switched on Help The Dev Improve."
        if let since = admin.summary?.since { s += " Since \(since)." }
        return s
    }

    // MARK: Sections

    private func installs(_ s: AdminSummary) -> some View {
        section("Opted-in installs reporting this week") {
            VStack(alignment: .leading, spacing: Theme.Space.s1) {
                Text(s.usageReportsThisWeek.map(String.init) ?? "–")
                    .font(Theme.Fonts.big)
                    .foregroundStyle(Theme.Palette.ink)
                    .monospacedDigit()
                Text("One report per install per week, no IDs.")
                    .font(Theme.Fonts.meta)
                    .foregroundStyle(Theme.Palette.muted)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .kitCard()
        }
    }

    private func models(_ rows: [AdminSummary.Model]) -> some View {
        section("Models", count: rows.count) {
            if rows.isEmpty {
                empty("No model use reported in this period.")
            } else {
                table(header: ["Model", "Requests", "Finished", "Success"],
                      rows: rows.map { [$0.model ?? "–", num($0.requests), num($0.finished), pct($0.successPct)] })
            }
        }
    }

    private func macs(_ rows: [AdminSummary.Mac]) -> some View {
        section("Macs", count: rows.count) {
            if rows.isEmpty {
                empty("No Macs reported in this period.")
            } else {
                table(header: ["Chip", "Memory", "Reports"],
                      rows: rows.map { [$0.chip ?? "–", $0.memoryGb.map { "\(Self.trim($0)) GB" } ?? "–", num($0.reports)] })
            }
        }
    }

    private func versions(_ rows: [AdminSummary.Version]) -> some View {
        section("Versions", count: rows.count) {
            if rows.isEmpty {
                empty("No versions reported in this period.")
            } else {
                table(header: ["Shuttrr version", "Reports"],
                      rows: rows.map { [$0.appVersion ?? "–", num($0.reports)] })
            }
        }
    }

    private func issues(_ rows: [AdminSummary.Issue]) -> some View {
        section("Crashes and failures", count: rows.count) {
            if rows.isEmpty {
                empty("No crashes or failures reported in this period.")
            } else {
                VStack(alignment: .leading, spacing: Theme.Space.s3) {
                    ForEach(Array(rows.enumerated()), id: \.offset) { item in
                        AdminIssueCard(issue: item.element)
                    }
                }
            }
        }
    }

    // MARK: Pieces

    private func section<Content: View>(_ title: String, count: Int? = nil,
                                        @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.s3) {
            SectionHeader(title: title, count: count)
            content()
        }
    }

    private func empty(_ text: String) -> some View {
        Text(text)
            .font(Theme.Fonts.row)
            .foregroundStyle(Theme.Palette.ink2)
            .frame(maxWidth: .infinity, alignment: .leading)
            .kitCard()
    }

    /// First column wide and left-aligned, numbers right-aligned.
    private func table(header: [String], rows: [[String]]) -> some View {
        VStack(spacing: 0) {
            line(header, header: true)
            ForEach(Array(rows.enumerated()), id: \.offset) { item in
                Rectangle().fill(Theme.Palette.line).frame(height: Theme.Metric.hairline)
                line(item.element, header: false)
            }
        }
        .padding(.horizontal, Theme.Space.s4)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassCard(padding: 0)
    }

    private func line(_ cells: [String], header: Bool) -> some View {
        HStack(spacing: Theme.Space.s4) {
            ForEach(Array(cells.enumerated()), id: \.offset) { item in
                let first = item.offset == 0
                Text(item.element)
                    .font(header ? Theme.Fonts.metaMedium : (first ? Theme.Fonts.row : Theme.Fonts.mono))
                    .foregroundStyle(header ? Theme.Palette.muted : Theme.Palette.ink)
                    .monospacedDigit()
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                    .frame(maxWidth: first ? .infinity : 110, alignment: first ? .leading : .trailing)
            }
        }
        .frame(height: header ? Theme.Metric.tableHeaderHeight : Theme.Metric.kvRowHeight)
    }

    private func num(_ v: Int?) -> String { v.map(String.init) ?? "–" }
    private func pct(_ v: Double?) -> String { v.map { "\(Self.trim($0))%" } ?? "–" }

    static func trim(_ v: Double) -> String {
        v.rounded() == v ? String(Int(v)) : String(format: "%.1f", v)
    }
}

/// One grouped crash or failure, with its latest stack behind a disclosure.
struct AdminIssueCard: View {
    let issue: AdminSummary.Issue
    @State private var open = false

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s2) {
            HStack(alignment: .firstTextBaseline, spacing: Theme.Space.s2) {
                KitBadge(text: issue.kind ?? "issue", kind: issue.kind == "app_crash" ? .danger : .outline)
                if let t = issue.errorType, !t.isEmpty {
                    Text(t).font(Theme.Fonts.monoMedium).foregroundStyle(Theme.Palette.ink)
                }
                Spacer(minLength: Theme.Space.s2)
                Text(times)
                    .font(Theme.Fonts.metaSemibold)
                    .foregroundStyle(Theme.Palette.ink)
                    .monospacedDigit()
            }
            if let m = issue.message, !m.isEmpty {
                Text(m)
                    .font(Theme.Fonts.row)
                    .foregroundStyle(Theme.Palette.ink2)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
            }
            Text(meta)
                .font(Theme.Fonts.meta)
                .foregroundStyle(Theme.Palette.muted)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
            if let trace = issue.latestTrace, !trace.isEmpty {
                Button {
                    withAnimation(Motion.snappy) { open.toggle() }
                } label: {
                    HStack(spacing: Theme.Space.s1) {
                        Image(systemName: "chevron.right")
                            .font(.system(size: 10, weight: .semibold))
                            .rotationEffect(.degrees(open ? 90 : 0))
                        Text(open ? "Hide latest stack" : "Show latest stack")
                            .font(Theme.Fonts.metaMedium)
                    }
                    .foregroundStyle(Theme.Palette.ink2)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                if open {
                    ScrollView([.vertical, .horizontal]) {
                        Text(trace)
                            .font(Theme.Fonts.mono)
                            .foregroundStyle(Theme.Palette.ink)
                            .textSelection(.enabled)
                            .fixedSize()
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(Theme.Space.s3)
                    }
                    .frame(maxHeight: 240)
                    .background(RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                        .fill(Theme.Palette.white))
                    .overlay(RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                        .strokeBorder(Theme.Palette.line, lineWidth: Theme.Metric.hairline))
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .kitCard()
    }

    private var times: String {
        guard let n = issue.times else { return "" }
        return n == 1 ? "1 time" : "\(n) times"
    }

    private var meta: String {
        var parts: [String] = []
        switch (issue.firstDay, issue.lastDay) {
        case let (f?, l?) where f == l: parts.append("Seen \(f)")
        case let (f?, l?): parts.append("First \(f), last \(l)")
        case let (f?, nil): parts.append("First \(f)")
        case let (nil, l?): parts.append("Last \(l)")
        default: break
        }
        if let v = issue.versions, !v.isEmpty { parts.append("Versions \(v)") }
        if let m = issue.models, !m.isEmpty { parts.append("Models \(m)") }
        return parts.isEmpty ? " " : parts.joined(separator: " · ")
    }
}

/// The hidden Admin Token… sheet (Settings, with Option held). Never shows the stored token.
struct AdminTokenSheet: View {
    @ObservedObject var admin = AdminStore.shared
    @Environment(\.dismiss) private var dismiss
    @State private var token = ""
    @State private var busy = false
    @State private var result: String?
    @State private var good = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: Theme.Space.s3) {
                Text("Admin Token")
                    .font(Theme.Fonts.section)
                    .foregroundStyle(Theme.Palette.ink)
                Text("Stored in this Mac's Keychain. Shuttrr checks it with the telemetry server.")
                    .font(Theme.Fonts.row)
                    .foregroundStyle(Theme.Palette.ink2)
                    .fixedSize(horizontal: false, vertical: true)
                SecureField("Token", text: $token)
                    .kitField()
                    .disabled(busy)
                    .onSubmit(save)
                if let result {
                    Text(result)
                        .font(Theme.Fonts.meta)
                        .foregroundStyle(good ? Theme.Palette.good : Theme.Palette.danger)
                }
            }
            .padding(Theme.Space.s5)
            Hairline()
            HStack {
                Button("Remove") {
                    admin.removeToken()
                    token = ""
                    good = false
                    result = "Removed."
                }
                .buttonStyle(.kit(.line))
                .disabled(busy)
                Spacer()
                Button("Close") { dismiss() }
                    .buttonStyle(.kit(.line))
                    .keyboardShortcut(.cancelAction)
                Button(busy ? "Checking…" : "Save") { save() }
                    .buttonStyle(.kit(.ink))
                    .keyboardShortcut(.defaultAction)
                    .disabled(busy || token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            .padding(Theme.Space.s4)
        }
        .frame(width: 440)
        .background(Theme.Palette.content)
        .preferredColorScheme(.light)
    }

    private func save() {
        guard !busy else { return }
        busy = true
        result = nil
        let t = token
        Task {
            let outcome = await admin.saveToken(t)
            busy = false
            token = ""
            good = outcome == .accepted
            switch outcome {
            case .accepted: result = "Accepted."
            case .notAccepted: result = "Not accepted."
            case .unreachable: result = "Saved, but the server could not be reached to check it."
            case .couldNotSave: result = "Could not save it to the Keychain."
            }
        }
    }
}

/// Watches the Option key while the view is on screen (for the hidden Admin Token… button in Settings).
struct OptionKeyWatcher: ViewModifier {
    @Binding var held: Bool
    @State private var monitor: Any?

    func body(content: Content) -> some View {
        content
            .onAppear {
                held = NSEvent.modifierFlags.contains(.option)
                monitor = NSEvent.addLocalMonitorForEvents(matching: .flagsChanged) { e in
                    held = e.modifierFlags.contains(.option)
                    return e
                }
            }
            .onDisappear {
                if let monitor { NSEvent.removeMonitor(monitor) }
                monitor = nil
                held = false
            }
    }
}
