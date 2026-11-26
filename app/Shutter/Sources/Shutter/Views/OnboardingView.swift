import AppKit
import SwiftUI

/// Which page of the first-run window is showing. Lives in the AppDelegate so "Set-up…" on the Models page can
/// reopen the window at the start.
@MainActor
final class OnboardingFlow: ObservableObject {
    enum Page: Int, CaseIterable, Sendable { case welcome, setup, model, ready }
    static let windowID = "onboarding"

    nonisolated static let startPage: Page? = {
        let names: [String: Page] = ["welcome": .welcome, "setup": .setup, "model": .model, "ready": .ready]
        return names[ProcessInfo.processInfo.environment["SHUTTER_ONBOARDING_PAGE"] ?? ""]
    }()

    @Published var page: Page = OnboardingFlow.startPage ?? .welcome
    /// Testing: SHUTTER_ONBOARDING_PAGE=welcome|setup|model|ready opens the window on that page
    /// (setup also starts the installer), so every page can be checked without clicking through.
    let autoRunSetup: Bool

    nonisolated init() {
        autoRunSetup = Self.startPage == .setup
    }

    func reset() { page = .welcome }
}

/// The first-run window: Welcome → Setting up → Choose a model → Ready. One primary button per page.
struct OnboardingView: View {
    @EnvironmentObject private var store: EngineStore
    @EnvironmentObject private var engine: EngineProcess
    @EnvironmentObject private var setup: SetupRunner
    @EnvironmentObject private var flow: OnboardingFlow
    @Environment(\.openWindow) private var openWindow
    @Environment(\.dismissWindow) private var dismissWindow

    /// About 960 x 800 points, clamped to the screen's visible frame; the window can be resized from there.
    static let minWidth: CGFloat = 640
    static let minHeight: CGFloat = 560
    static var width: CGFloat { clamped(960, \.width, margin: 80, floor: minWidth) }
    static var height: CGFloat { clamped(800, \.height, margin: 60, floor: minHeight) }
    private static func clamped(_ want: CGFloat, _ side: KeyPath<CGSize, CGFloat>, margin: CGFloat,
                                floor: CGFloat) -> CGFloat {
        guard let vis = NSScreen.main?.visibleFrame.size else { return want }
        return max(floor, min(want, vis[keyPath: side] - margin))
    }
    /// Clears the close, minimise and zoom buttons (the content runs under the hidden title bar).
    static let headerTop: CGFloat = 52

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
                .padding(.top, Self.headerTop)
                .padding(.bottom, Theme.Space.s5)
            Group {
                switch flow.page {
                case .welcome: WelcomePage(onSetUp: startSetup, onSkip: { go(.model) })
                case .setup: SetupPage(onRetry: startSetup)
                case .model: ModelPage(onContinue: { go(.ready) })
                case .ready: ReadyPage(onClose: close, onOpen: openDashboard)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .transition(.opacity.combined(with: .offset(y: 6)))
            .id(flow.page)
        }
        .padding(.horizontal, Theme.Space.s7)
        .padding(.bottom, Theme.Space.s6)
        .frame(minWidth: Self.minWidth, idealWidth: Self.width, maxWidth: .infinity,
               minHeight: Self.minHeight, idealHeight: Self.height, maxHeight: .infinity)
        .background(PaperBackground())
        .background(PaperWindow())
        .background(OnboardingWindowSizer())
        .animation(Motion.gentle, value: flow.page)
        .onAppear {
            if flow.autoRunSetup && flow.page == .setup && setup.phase == .idle { setup.run() }
        }
        .onChange(of: setup.phase) { _, phase in
            if phase == .finished { setupFinished() }
        }
    }

    private var header: some View {
        HStack(alignment: .center) {
            HStack(spacing: Theme.Space.s2) {
                Image(systemName: "camera.aperture")
                    .font(.system(size: Theme.Size.section, weight: .medium))
                    .foregroundStyle(Theme.Palette.ink)      // the logo is black and white, never coloured
                Text("Shuttrr.").foregroundColor(Theme.Palette.ink)
                    .font(Theme.Fonts.brand)
            }
            Spacer()
            HStack(spacing: Theme.Space.s1) {
                ForEach(OnboardingFlow.Page.allCases, id: \.self) { p in
                    Capsule()
                        .fill(p.rawValue <= flow.page.rawValue ? Theme.Palette.ink : Theme.Palette.fillStrong)
                        .frame(width: p == flow.page ? 22 : 10, height: 4)
                }
            }
            .animation(Motion.snappy, value: flow.page)
            .accessibilityElement()
            .accessibilityLabel("Step \(flow.page.rawValue + 1) of \(OnboardingFlow.Page.allCases.count)")
        }
    }

    private func go(_ page: OnboardingFlow.Page) {
        flow.page = page
    }

    private func startSetup() {
        go(.setup)
        setup.run()
    }

    private func setupFinished() {
        Task { @MainActor in
            if !(await EngineProbe.answers()) && !engine.isRunning {
                engine.start()
            }
            store.refreshNow()
            try? await Task.sleep(nanoseconds: 700_000_000)
            if flow.page == .setup { go(.model) }
        }
    }

    private func close() {
        dismissWindow(id: OnboardingFlow.windowID)
    }

    private func openDashboard() {
        openWindow(id: AppDelegate.dashboardID)
        NSApp.activate(ignoringOtherApps: true)
        dismissWindow(id: OnboardingFlow.windowID)
    }
}

/// Does anything answer on the engine port? Short timeout: the engine is local.
enum EngineProbe {
    static func answers(timeout: TimeInterval = 1.5) async -> Bool {
        guard let url = try? APIClient.current().url("/v1/system") else { return false }
        var req = URLRequest(url: url)
        req.timeoutInterval = timeout
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = timeout
        config.timeoutIntervalForResource = timeout + 1
        do {
            let (_, resp) = try await URLSession(configuration: config).data(for: req)
            return (resp as? HTTPURLResponse).map { $0.statusCode < 500 } ?? false
        } catch {
            return false
        }
    }
}

// MARK: - Shared pieces

private struct OnboardTitle: View {
    let title: String
    let text: String?

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s3) {
            Text(title)
                .font(Brand.display(32, weight: 500))
                .foregroundStyle(Theme.Palette.ink)
                .fixedSize(horizontal: false, vertical: true)
            if let text {
                Text(text)
                    .font(Brand.sans(15))
                    .foregroundStyle(Theme.Palette.ink2)
                    .lineSpacing(3)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

private struct NumberedPoint: View {
    let number: String
    let title: String
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: Theme.Space.s4) {
            Text(number)
                .font(Theme.Fonts.monoMedium)
                .foregroundStyle(Theme.Palette.ink)
                .frame(width: 28, height: 28)
                .background(Circle().fill(Theme.Palette.white))
                .overlay(Circle().strokeBorder(Theme.Palette.lineStrong, lineWidth: Theme.Metric.hairline))
            VStack(alignment: .leading, spacing: Theme.Space.s1) {
                Text(title)
                    .font(Brand.sans(14, weight: 600))
                    .foregroundStyle(Theme.Palette.ink)
                Text(text)
                    .font(Theme.Fonts.row)
                    .foregroundStyle(Theme.Palette.ink2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.top, 4)
            Spacer(minLength: 0)
        }
    }
}

/// A page: the content scrolls when it is taller than the window, the footer always stays in view.
private struct PageLayout<Content: View, FooterContent: View>: View {
    @ViewBuilder var content: Content
    @ViewBuilder var footer: FooterContent

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 0) { content }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.bottom, Theme.Space.s2)
            }
            .scrollBounceBehavior(.basedOnSize)
            .scrollIndicators(.automatic)
            .frame(maxHeight: .infinity)
            footer
        }
    }
}

private struct Footer<Leading: View, Trailing: View>: View {
    @ViewBuilder var leading: Leading
    @ViewBuilder var trailing: Trailing

    var body: some View {
        HStack(spacing: Theme.Space.s3) {
            leading
            Spacer(minLength: Theme.Space.s3)
            trailing
        }
        .padding(.top, Theme.Space.s5)
    }
}

// MARK: - Pages

private struct WelcomePage: View {
    let onSetUp: () -> Void
    let onSkip: () -> Void
    private let installed = SetupRunner.engineInstalled

    var body: some View {
        PageLayout {
            OnboardTitle(title: "Welcome to Shuttrr",
                       text: "Shuttrr runs big video and voice models on your Mac in far less memory than they normally need. Set-up takes a few minutes and you never open Terminal.")
            VStack(alignment: .leading, spacing: Theme.Space.s4) {
                NumberedPoint(number: "1", title: "Check this Mac", text: "Chip, memory and free disk space.")
                NumberedPoint(number: "2", title: "Install the engine",
                              text: "About 2 GB in a hidden folder in your home folder. No password needed.")
                NumberedPoint(number: "3", title: "Choose a model",
                              text: "Download one to make your first video. It stays on this Mac.")
            }
            .padding(.top, Theme.Space.s6)
        } footer: {
            Footer {
                if installed {
                    Button("Run Set-up Again", action: onSetUp)
                        .buttonStyle(.kit(.line))
                }
            } trailing: {
                if installed {
                    Button("Continue", action: onSkip)
                        .buttonStyle(.kit(.ink))
                        .keyboardShortcut(.defaultAction)
                } else {
                    Button("Set Up Shuttrr", action: onSetUp)
                        .buttonStyle(.kit(.ink))
                        .keyboardShortcut(.defaultAction)
                }
            }
        }
    }
}

private struct SetupPage: View {
    @EnvironmentObject private var setup: SetupRunner
    let onRetry: () -> Void
    @State private var showLog = false

    private var failure: String? {
        if case .failed(let reason) = setup.phase { return reason }
        return nil
    }

    var body: some View {
        PageLayout {
            OnboardTitle(title: failure == nil ? "Setting up Shuttrr" : "Set-up stopped", text: nil)
            ProgressLine(value: setup.phase == .finished ? 1 : setup.progress, width: nil,
                         height: Theme.Metric.progressHeightWide,
                         tint: failure == nil ? nil : Theme.Palette.danger.opacity(0.7))
                .padding(.top, Theme.Space.s4)
            if let failure {
                Text(failure)
                    .font(Brand.sans(14, weight: 500))
                    .foregroundStyle(Theme.Palette.danger)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
                    .padding(.top, Theme.Space.s3)
            }
            VStack(alignment: .leading, spacing: Theme.Space.s3) {
                ForEach(setup.steps) { step in
                    StepRow(step: step)
                }
            }
            .padding(.top, Theme.Space.s5)
            if showLog {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 1) {
                            ForEach(Array(setup.logLines.enumerated()), id: \.offset) { i, line in
                                Text(line.isEmpty ? " " : line)
                                    .font(Brand.mono(10.5))
                                    .foregroundStyle(Theme.Palette.ink2)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .id(i)
                            }
                        }
                        .padding(Theme.Space.s3)
                        .textSelection(.enabled)
                    }
                    .frame(height: 150)
                    .background(RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                        .fill(Theme.Palette.white))
                    .overlay(RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                        .strokeBorder(Theme.Palette.line, lineWidth: Theme.Metric.hairline))
                    .onChange(of: setup.logLines.count) { _, n in
                        if n > 0 { proxy.scrollTo(n - 1, anchor: .bottom) }
                    }
                    .onAppear {
                        if !setup.logLines.isEmpty { proxy.scrollTo(setup.logLines.count - 1, anchor: .bottom) }
                    }
                }
                .padding(.top, Theme.Space.s4)
            }
        } footer: {
            Footer {
                Button(showLog ? "Hide Details" : "Show Details") {
                    withAnimation(Motion.smooth) { showLog.toggle() }
                }
                .buttonStyle(.kit(.line))
            } trailing: {
                if failure != nil {
                    Button("Try Again", action: onRetry)
                        .buttonStyle(.kit(.ink))
                        .keyboardShortcut(.defaultAction)
                } else {
                    Button(setup.phase == .finished ? "Done" : "Setting Up…") {}
                        .buttonStyle(.kit(.ink))
                        .disabled(true)
                }
            }
        }
    }
}

private struct StepRow: View {
    let step: SetupRunner.Step

    var body: some View {
        HStack(alignment: .top, spacing: Theme.Space.s3) {
            icon
                .frame(width: 18, height: 18)
                .padding(.top, 1)
            VStack(alignment: .leading, spacing: 2) {
                Text(step.title)
                    .font(Brand.sans(14, weight: step.state == .pending ? 400 : 600))
                    .foregroundStyle(step.state == .pending ? Theme.Palette.muted : Theme.Palette.ink)
                if !step.detail.isEmpty && step.state != .pending {
                    Text(step.detail)
                        .font(Theme.Fonts.meta)
                        .foregroundStyle(step.state == .failed ? Theme.Palette.danger : Theme.Palette.ink2)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 0)
        }
        .animation(Motion.smooth, value: step.state)
    }

    @ViewBuilder private var icon: some View {
        switch step.state {
        case .pending:
            Circle().strokeBorder(Theme.Palette.lineStrong, lineWidth: 1.5)
        case .running:
            ProgressView().controlSize(.small).scaleEffect(0.8)
        case .done:
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(Theme.Palette.good)
        case .failed:
            Image(systemName: "xmark.circle.fill")
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(Theme.Palette.danger)
        }
    }
}

private struct ModelPage: View {
    @EnvironmentObject private var store: EngineStore
    let onContinue: () -> Void
    @State private var waitedLong = false
    /// A Download button on this page was pressed: once the download runs, move on by ourselves.
    @State private var startedHere = false
    @State private var advancing = false

    private var anyInstalled: Bool { store.catalog.contains { $0.installed } }
    private var downloading: [CatalogModel] { store.catalog.filter { $0.isDownloading } }
    /// Rows still offering a choice; a model that is downloading becomes one quiet line instead.
    private var choices: [CatalogModel] { store.catalog.filter { !$0.isDownloading } }

    var body: some View {
        PageLayout {
            OnboardTitle(title: "Choose a model",
                         text: "Models are large, so the download takes a while. You can close this window; it carries on.")
            Group {
                if !store.connection.isConnected {
                    waitingLine("Starting the engine…")
                } else if store.catalog.isEmpty {
                    if waitedLong {
                        Text("The engine running on this Mac has no models to download. It may be an older copy started by hand. Quit it, then open this window again.")
                            .font(Theme.Fonts.row)
                            .foregroundStyle(Theme.Palette.ink2)
                            .fixedSize(horizontal: false, vertical: true)
                    } else {
                        waitingLine("Reading the list of models…")
                    }
                } else {
                    VStack(alignment: .leading, spacing: Theme.Space.s3) {
                        ForEach(downloading) { model in
                            BackgroundDownloadLine(model: model)
                        }
                        SectionedModelList(models: choices)
                    }
                }
            }
            .padding(.top, Theme.Space.s5)
            .animation(Motion.smooth, value: downloading.map(\.id))
        } footer: {
            Footer {
                EmptyView()
            } trailing: {
                Button(buttonTitle, action: onContinue)
                    .buttonStyle(.kit(.ink))
                    .keyboardShortcut(.defaultAction)
            }
        }
        .task {
            store.refreshNow()
            try? await Task.sleep(nanoseconds: 10_000_000_000)
            waitedLong = true
        }
        .onChange(of: store.catalogPending) { _, pending in
            if !pending.isEmpty { startedHere = true }
        }
        .onChange(of: store.catalogDownloading) { _, running in
            guard running, startedHere, !advancing else { return }
            advancing = true
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: 1_100_000_000)
                onContinue()
            }
        }
    }

    private var buttonTitle: String {
        if store.catalogDownloading || anyInstalled { return "Continue" }
        return "Skip for Now"
    }

    private func waitingLine(_ text: String) -> some View {
        HStack(spacing: Theme.Space.s3) {
            ProgressView().controlSize(.small)
            Text(text)
                .font(Theme.Fonts.row)
                .foregroundStyle(Theme.Palette.ink2)
        }
    }
}

/// One line for a model that is already downloading: it carries on by itself, nothing to press here.
private struct BackgroundDownloadLine: View {
    let model: CatalogModel

    var body: some View {
        HStack(alignment: .top, spacing: Theme.Space.s3) {
            Image(systemName: "arrow.down.circle.fill")
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(Brand.gradient)
                .padding(.top, 1)
            VStack(alignment: .leading, spacing: 2) {
                Text(model.name)
                    .font(Brand.sans(14, weight: 600))
                    .foregroundStyle(Theme.Palette.ink)
                Text("Downloading in the background. You can carry on.")
                    .font(Theme.Fonts.row)
                    .foregroundStyle(Theme.Palette.ink2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, Theme.Space.s5)
        .padding(.vertical, Theme.Space.s4)
        .background(RoundedRectangle(cornerRadius: Theme.Radius.control + 2, style: .continuous)
            .fill(Theme.Palette.white.opacity(0.7)))
        .overlay(RoundedRectangle(cornerRadius: Theme.Radius.control + 2, style: .continuous)
            .strokeBorder(Theme.Palette.line, lineWidth: Theme.Metric.hairline))
        .transition(.opacity)
    }
}

/// The live download on the Ready page: name, a slim bar, GB done of total and percent.
private struct LiveDownloadCard: View {
    let model: CatalogModel

    var body: some View {
        let d = model.download
        let total = (d?.totalBytes ?? 0) > 0 ? (d?.totalBytes ?? 0) : model.sizeBytes
        VStack(alignment: .leading, spacing: Theme.Space.s2) {
            HStack(alignment: .firstTextBaseline) {
                Text(model.name)
                    .font(Brand.sans(14, weight: 600))
                    .foregroundStyle(Theme.Palette.ink)
                Spacer(minLength: Theme.Space.s3)
                Text("\(Fmt.gb(d?.doneBytes ?? 0)) of \(Fmt.gb(total)) · \(Fmt.percent(d?.fraction ?? 0))")
                    .font(Theme.Fonts.mono)
                    .foregroundStyle(Theme.Palette.ink2)
                    .monospacedDigit()
            }
            ProgressLine(value: d?.fraction ?? 0, width: nil, height: Theme.Metric.progressHeight)
            Text("Your model is downloading in the background. You can close this window; Shuttrr tells you on the Models page when it is ready.")
                .font(Theme.Fonts.meta)
                .foregroundStyle(Theme.Palette.ink2)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, Theme.Space.s5)
        .padding(.vertical, Theme.Space.s4)
        .background(RoundedRectangle(cornerRadius: Theme.Radius.control + 2, style: .continuous)
            .fill(Theme.Palette.white))
        .overlay(RoundedRectangle(cornerRadius: Theme.Radius.control + 2, style: .continuous)
            .strokeBorder(Theme.Palette.line, lineWidth: Theme.Metric.hairline))
    }
}

private struct ReadyPage: View {
    @EnvironmentObject private var store: EngineStore
    let onClose: () -> Void
    let onOpen: () -> Void

    private var stateLine: String {
        if store.catalogDownloading {
            return "Everything is set up. Your model is still on its way."
        }
        if store.catalog.contains(where: { $0.installed }) {
            return "Everything is installed. You can make your first video now."
        }
        return "There is no model yet. Download one any time from the Models page."
    }

    var body: some View {
        PageLayout {
            OnboardTitle(title: "Shuttrr is ready", text: stateLine)
            let live = store.catalog.filter { $0.isDownloading }
            ForEach(live) { model in
                LiveDownloadCard(model: model)
                    .padding(.top, Theme.Space.s5)
            }
            VStack(alignment: .leading, spacing: live.isEmpty ? Theme.Space.s4 : Theme.Space.s3) {
                NumberedPoint(number: "1", title: "It lives in the menu bar",
                              text: "Click the aperture icon at the top of the screen.")
                NumberedPoint(number: "2", title: "Other apps can use it",
                              text: "They send requests to http://127.0.0.1:8484 on this Mac.")
                NumberedPoint(number: "3", title: "It protects your Mac",
                              text: "A request waits rather than push the Mac into swap.")
            }
            .padding(.top, live.isEmpty ? Theme.Space.s6 : Theme.Space.s5)
        } footer: {
            Footer {
                EmptyView()
            } trailing: {
                Button("Close", action: onClose)
                    .buttonStyle(.kit(.line))
                Button("Open Shuttrr", action: onOpen)
                    .buttonStyle(.kit(.ink))
                    .keyboardShortcut(.defaultAction)
            }
        }
        .onAppear { UserDefaults.standard.set(true, forKey: SetupRunner.onboardingDoneKey) }
    }
}


/// Opens the first-run window at its full size even when macOS restores a smaller frame saved by an older version
/// (0.2.0 was 640 x 600). Done once per window, so a size the person picks by resizing is kept while it is open.
private struct OnboardingWindowSizer: NSViewRepresentable {
    @MainActor private static var sized = Set<ObjectIdentifier>()

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async { Self.size(view.window) }
        return view
    }

    func updateNSView(_ view: NSView, context: Context) {
        DispatchQueue.main.async { Self.size(view.window) }
    }

    @MainActor private static func size(_ window: NSWindow?) {
        guard let window, !sized.contains(ObjectIdentifier(window)) else { return }
        sized.insert(ObjectIdentifier(window))
        let want = NSSize(width: OnboardingView.width, height: OnboardingView.height)
        let f = window.frame
        guard f.width < want.width - 1 || f.height < want.height - 1 else { return }
        let vis = window.screen?.visibleFrame ?? NSScreen.main?.visibleFrame ?? f
        let size = NSSize(width: max(f.width, want.width), height: max(f.height, want.height))
        let origin = NSPoint(x: vis.midX - size.width / 2, y: vis.midY - size.height / 2)
        window.setFrame(NSRect(origin: origin, size: size), display: true, animate: false)
    }
}
