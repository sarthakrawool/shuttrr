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

/// The happy palette of the "help the dev" question: sunny cream and butter yellow with a warm orange accent,
/// on purpose unlike the app's ink-on-paper look so the question feels like a friendly note, not a setting.
enum Sunny {
    static let cream = Color(red: 1.00, green: 0.968, blue: 0.878)      // #FFF7E0
    static let butter = Color(red: 1.00, green: 0.886, blue: 0.533)     // #FFE288
    static let orange = Color(red: 0.925, green: 0.553, blue: 0.180)    // #EC8D2E
    static let orangeDeep = Color(red: 0.820, green: 0.443, blue: 0.098) // #D17119
    static let ink = Color(red: 0.231, green: 0.165, blue: 0.094)       // #3B2A18, warm dark brown
    static let ink2 = Color(red: 0.420, green: 0.333, blue: 0.231)      // #6B553B
    static let paper = Color(red: 1.00, green: 0.992, blue: 0.965)      // #FFFDF6, opaque card for the lists
    static let teal = Color(red: 0.165, green: 0.455, blue: 0.427)      // #2A746D (AA on cream)

    static let background = LinearGradient(colors: [cream, butter.opacity(0.75)], startPoint: .top, endPoint: .bottom)

    /// The gouache illustration (Z-Image Turbo, made with Shuttrr; Apache-2.0 model). Shipped in Resources/Brand.
    static let image: NSImage? = {
        guard let url = Brand.resourceDirectory?.appendingPathComponent("help-the-dev.png") else { return nil }
        return NSImage(contentsOf: url)
    }()
}

/// The question itself: the little character, a friendly headline, why it is safe, and two clear answers.
/// Used by the one-time popup after an update and by the set-up window's last-but-one page.
struct HelpTheDevCard: View {
    @EnvironmentObject private var store: EngineStore
    let busy: Bool
    let answer: (Bool) -> Void
    // Testing: SHUTTER_TEST_WHATS_SENT=open shows the lists, =bottom also scrolls them to the last line
    @State private var showWhat = ProcessInfo.processInfo.environment["SHUTTER_TEST_WHATS_SENT"] != nil

    var body: some View {
        VStack(spacing: 0) {
            if !showWhat, let image = Sunny.image {   // the lists take its place, so the popup keeps its size
                Image(nsImage: image)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fit)          // never cropped
                    .frame(maxWidth: 220, maxHeight: 200)
                    .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
                    .shadow(color: Sunny.orange.opacity(0.18), radius: 14, y: 6)
                    .accessibilityLabel("A small painted character looking up hopefully")
                    .padding(.bottom, 18)
            }
            Text("Would you like to help the dev?")
                .font(Brand.display(26, weight: 560))
                .foregroundStyle(Sunny.ink)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            Text("Shuttrr is made by one person. If you say yes, it sends a tiny anonymous note now and then: which models you use, and what went wrong if something crashes.")
                .font(Brand.sans(14))
                .foregroundStyle(Sunny.ink2)
                .multilineTextAlignment(.center)
                .lineSpacing(2)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 10)
            HStack(spacing: 8) {
                Image(systemName: "lock.fill").font(.system(size: 11, weight: .semibold))
                Text("No names, files or prompts. Off unless you say yes, and you can change your mind any time in Settings.")
                    .font(Brand.sans(12.5, weight: 500))
                    .fixedSize(horizontal: false, vertical: true)
            }
            .foregroundStyle(Sunny.teal)
            .padding(.horizontal, 14)
            .padding(.vertical, 9)
            .background(Capsule(style: .continuous).fill(Color.white.opacity(0.65)))
            .padding(.top, 14)
            VStack(spacing: 10) {
                Button { answer(true) } label: {
                    HStack(spacing: 8) {
                        if busy { ProgressView().controlSize(.small).tint(.white) }
                        Text("Yes, happy to help")
                    }
                    .font(Brand.sans(15, weight: 650))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity, minHeight: 42)
                    .background(Capsule(style: .continuous)
                        .fill(LinearGradient(colors: [Sunny.orange, Sunny.orangeDeep], startPoint: .top, endPoint: .bottom)))
                    .shadow(color: Sunny.orangeDeep.opacity(0.30), radius: 8, y: 3)
                    .contentShape(Capsule())
                }
                .buttonStyle(.plain)
                .keyboardShortcut(.defaultAction)
                .disabled(busy)
                Button { answer(false) } label: {
                    Text("No thanks")
                        .font(Brand.sans(14, weight: 600))
                        .foregroundStyle(Sunny.ink2)
                        .frame(maxWidth: .infinity, minHeight: 38)
                        .background(Capsule(style: .continuous).fill(Color.white.opacity(0.55)))
                        .overlay(Capsule(style: .continuous).strokeBorder(Sunny.ink2.opacity(0.18), lineWidth: 1))
                        .contentShape(Capsule())
                }
                .buttonStyle(.plain)
                .keyboardShortcut(.cancelAction)
                .disabled(busy)
            }
            .frame(maxWidth: 300)
            .padding(.top, 20)
            Button {
                withAnimation(.easeInOut(duration: 0.2)) { showWhat.toggle() }
            } label: {
                HStack(spacing: 4) {
                    Text(showWhat ? "Hide what's sent" : "See exactly what's sent").underline()
                    Image(systemName: showWhat ? "chevron.up" : "chevron.down").font(.system(size: 9, weight: .bold))
                }
                .font(Brand.sans(12.5, weight: 600))
                .foregroundStyle(Sunny.orangeDeep)
            }
            .buttonStyle(.plain)
            .padding(.top, 12)
            if showWhat {
                // in place of the picture and about its height, so the popup never outgrows a small window
                ScrollViewReader { proxy in
                    ScrollView {
                        VStack(spacing: 0) {
                            SunnyLists(info: store.telemetry)
                            Color.clear.frame(height: 1).id("end")
                        }
                    }
                    .onAppear {
                        if ProcessInfo.processInfo.environment["SHUTTER_TEST_WHATS_SENT"] == "bottom" {
                            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { withAnimation { proxy.scrollTo("end", anchor: .bottom) } }
                        }
                    }
                }
                    .scrollIndicators(.automatic)
                    .scrollBounceBehavior(.basedOnSize)
                    .frame(maxHeight: 190)
                    .padding(.top, 12)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(.horizontal, 32)
        .padding(.top, 28)
        .padding(.bottom, 24)
        .frame(maxWidth: .infinity)
    }
}

/// What is sent and what never is, in the popup's own opaque warm colours (the same in light and dark mode).
struct SunnyLists: View {
    let info: TelemetryInfo?

    var body: some View {
        let t = info ?? TelemetryInfo()
        VStack(alignment: .leading, spacing: 14) {
            list("What is sent", t.whatOrFallback, symbol: "checkmark.circle.fill", tint: Sunny.teal)
            list("Never sent", t.neverOrFallback, symbol: "xmark.circle.fill", tint: Sunny.orangeDeep)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 16, style: .continuous).fill(Sunny.paper))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous)
            .strokeBorder(Sunny.ink.opacity(0.10), lineWidth: 1))
    }

    private func list(_ title: String, _ items: [String], symbol: String, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(title)
                .font(Brand.sans(13.5, weight: 700))
                .foregroundStyle(Sunny.ink)
            ForEach(items, id: \.self) { item in
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Image(systemName: symbol)
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(tint)
                        .frame(width: 14)
                    Text(item)
                        .font(Brand.sans(13))
                        .foregroundStyle(Sunny.ink)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }
}

/// Asked once, over the main window, of people who finished set-up before this option existed.
/// Either answer (or Esc) marks it as asked in the engine, so it never shows again. Nothing is on until "Yes".
struct TelemetryAskSheet: View {
    @EnvironmentObject private var store: EngineStore

    var body: some View {
        HelpTheDevCard(busy: false) { yes in store.closeTelemetryAsk(yes) }
            .frame(width: 460)
            .fixedSize(horizontal: false, vertical: true)
            .background(Sunny.background)
            .environment(\.colorScheme, .light)
    }
}
