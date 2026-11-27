import AppKit
import SwiftUI

/// "Help The Dev Improve": the shared wording, the What's Sent sheet and the one-time question.
enum TelemetryCopy {
    static let toggle = "Send anonymous usage and crash reports"
    static let explanation = "If you agree, Shuttrr sends a short anonymous report now and then: which model you use the most, and what went wrong when something crashes. Nothing in it says who you are. It stays off unless you switch it on, and you can change it any time in Settings."
}

/// What is sent and what never is, as two short bulleted lists.
struct TelemetryLists: View {
    let info: TelemetryInfo?

    var body: some View {
        let t = info ?? TelemetryInfo()
        VStack(alignment: .leading, spacing: Theme.Space.s4) {
            list("What is sent", t.whatOrFallback, symbol: "checkmark", tint: Theme.Palette.good)
            list("Never sent", t.neverOrFallback, symbol: "xmark", tint: Theme.Palette.muted)
        }
    }

    private func list(_ title: String, _ items: [String], symbol: String, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.s2) {
            Text(title)
                .font(Brand.sans(14, weight: 600))
                .foregroundStyle(Theme.Palette.ink)
            ForEach(items, id: \.self) { item in
                HStack(alignment: .firstTextBaseline, spacing: Theme.Space.s2) {
                    Image(systemName: symbol)
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(tint)
                        .frame(width: 14)
                    Text(item)
                        .font(Theme.Fonts.row)
                        .foregroundStyle(Theme.Palette.ink2)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }
}

/// Settings › See What's Sent: the two lists, the last payload as it was sent, and when.
struct TelemetryDetailsSheet: View {
    @EnvironmentObject private var store: EngineStore
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        let t = store.telemetry
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: Theme.Space.s2) {
                Text("What Shuttrr sends")
                    .font(Theme.Fonts.section)
                    .foregroundStyle(Theme.Palette.ink)
                Text(t?.enabled == true ? "Help The Dev Improve is on." : "Help The Dev Improve is off. Nothing is sent.")
                    .font(Theme.Fonts.row)
                    .foregroundStyle(Theme.Palette.ink2)
            }
            .padding(Theme.Space.s5)
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.Space.s5) {
                    TelemetryLists(info: t)
                    VStack(alignment: .leading, spacing: Theme.Space.s2) {
                        HStack(alignment: .firstTextBaseline) {
                            Text("Last report sent")
                                .font(Brand.sans(14, weight: 600))
                                .foregroundStyle(Theme.Palette.ink)
                            Spacer(minLength: Theme.Space.s3)
                            if let at = t?.lastSentAt, t?.lastPayloadText != nil {
                                Text(Self.relative(at))
                                    .font(Theme.Fonts.meta)
                                    .foregroundStyle(Theme.Palette.muted)
                            }
                        }
                        if let text = t?.lastPayloadText {
                            ScrollView([.vertical, .horizontal]) {
                                Text(text)
                                    .font(Theme.Fonts.mono)
                                    .foregroundStyle(Theme.Palette.ink)
                                    .textSelection(.enabled)
                                    .fixedSize()
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(Theme.Space.s3)
                            }
                            .frame(height: 200)
                            .background(RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                                .fill(Theme.Palette.white))
                            .overlay(RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                                .strokeBorder(Theme.Palette.line, lineWidth: Theme.Metric.hairline))
                        } else {
                            Text("Nothing has been sent yet.")
                                .font(Theme.Fonts.row)
                                .foregroundStyle(Theme.Palette.ink2)
                        }
                        if let n = t?.queuedIssues, n > 0 {
                            Text(n == 1 ? "1 issue is waiting to be sent." : "\(n) issues are waiting to be sent.")
                                .font(Theme.Fonts.meta)
                                .foregroundStyle(Theme.Palette.muted)
                        }
                    }
                }
                .padding(.horizontal, Theme.Space.s5)
                .padding(.bottom, Theme.Space.s5)
            }
            .frame(maxHeight: 460)
            Hairline()
            HStack {
                Spacer()
                Button("Done") { dismiss() }
                    .buttonStyle(.kit(.ink))
                    .keyboardShortcut(.defaultAction)
            }
            .padding(Theme.Space.s4)
        }
        .frame(width: 560)
        .background(Theme.Palette.content)
        .preferredColorScheme(.light)
        .onAppear { store.refreshTelemetry() }
    }

    static func relative(_ unix: Double, now: Date = Date()) -> String {
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .full
        return f.localizedString(for: Date(timeIntervalSince1970: unix), relativeTo: now)
    }
}

/// Asked once, over the main window, of people who finished set-up before this option existed.
/// Saving or closing marks it as asked in the engine, so it never shows again.
struct TelemetryAskSheet: View {
    @EnvironmentObject private var store: EngineStore
    @State private var on = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: Theme.Space.s3) {
                Text("Help The Dev Improve")
                    .font(Theme.Fonts.section)
                    .foregroundStyle(Theme.Palette.ink)
                Text(TelemetryCopy.explanation)
                    .font(Theme.Fonts.row)
                    .foregroundStyle(Theme.Palette.ink2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(Theme.Space.s5)
            VStack(alignment: .leading, spacing: Theme.Space.s4) {
                TelemetryLists(info: store.telemetry)
                Toggle(isOn: $on) {
                    Text(TelemetryCopy.toggle)
                        .font(Brand.sans(14, weight: 600))
                        .foregroundStyle(Theme.Palette.ink)
                }
                .toggleStyle(.switch)
                .tint(Theme.Palette.violet)
                .padding(.top, Theme.Space.s2)
            }
            .padding(.horizontal, Theme.Space.s5)
            .padding(.bottom, Theme.Space.s5)
            Hairline()
            HStack {
                Spacer()
                Button("Close") { store.closeTelemetryAsk(nil) }
                    .buttonStyle(.kit(.line))
                    .keyboardShortcut(.cancelAction)
                Button("Save") { store.closeTelemetryAsk(on) }
                    .buttonStyle(.kit(.ink))
                    .keyboardShortcut(.defaultAction)
            }
            .padding(Theme.Space.s4)
        }
        .frame(width: 520)
        .background(Theme.Palette.content)
        .preferredColorScheme(.light)
    }
}
