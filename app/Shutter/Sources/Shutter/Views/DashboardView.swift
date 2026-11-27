import AppKit
import SwiftUI

enum DashboardSection: String, CaseIterable, Identifiable, Hashable {
    case models = "Models"
    case queue = "Requests"
    case memory = "Memory"
    case server = "Server"
    case settings = "Settings"

    var id: DashboardSection { self }

    var symbol: String {
        switch self {
        case .queue: return "arrow.left.arrow.right"
        case .models: return "cube"
        case .memory: return "memorychip"
        case .server: return "server.rack"
        case .settings: return "gearshape"
        }
    }

    var shortcut: KeyEquivalent {
        switch self {
        case .queue: return "2"
        case .models: return "1"
        case .memory: return "3"
        case .server: return "4"
        case .settings: return "5"
        }
    }
}

enum QueueFilter: String, CaseIterable, Hashable {
    case all, active, finished, failed

    var label: String { rawValue.capitalized }

    func includes(_ state: JobState) -> Bool {
        switch self {
        case .all: return true
        case .active: return state.isLive
        case .finished: return state == .done || state == .cancelled
        case .failed: return state == .failed
        }
    }
}

/// The main window, laid out like the web dashboard (the reference design):
/// full-height sidebar | toolbar (search + filter left, engine status right) / content / status bar.
struct DashboardView: View {
    @EnvironmentObject private var store: EngineStore
    @EnvironmentObject private var engine: EngineProcess

    /// Testing: SHUTTER_SECTION=models|requests|memory|server|settings opens the window on that page.
    @State private var section: DashboardSection = DashboardSection.allCases.first {
        $0.rawValue.lowercased() == (ProcessInfo.processInfo.environment["SHUTTER_SECTION"] ?? "").lowercased()
    } ?? .models
    @State private var query = ""
    @State private var filter: QueueFilter = .all
    @State private var selectedJob: Job.ID?
    @State private var showingNewJob = false
    @State private var newJobModel: String?
    @FocusState private var searchFocused: Bool
    @ObservedObject private var quit = QuitConfirm.shared

    var body: some View {
        HStack(spacing: 0) {
            SidebarView(section: $section, starting: starting)
            .frame(width: Theme.Metric.sidebarWidth)
            // the licence of a model about to download (Fish Drama 3), unless the Download a Model sheet shows it;
            // on its own view because SwiftUI shows only one of several sheets chained on the same view
            .sheet(item: Binding(get: { showingDownloadsElsewhere ? nil : store.licenceAsk },
                                 set: { store.licenceAsk = $0 })) { m in
                LicenceSheet(model: m).environmentObject(store)
            }
            .background(Theme.Palette.sidebarVeil)

            Hairline(vertical: true).ignoresSafeArea()

            VStack(spacing: 0) {
                toolbar
                Hairline()
                if let err = store.lastError {
                    KitBanner(symbol: "exclamationmark.triangle", title: "That did not work", detail: err) {
                        Button("Dismiss") { store.lastError = nil }
                            .buttonStyle(.kit(.onBanner))
                    }
                    .padding(.horizontal, Theme.Space.s6)
                    .padding(.top, Theme.Space.s4)
                    .sectionTransition()
                }
                Group {
                    if starting {
                        EngineStartingView(section: section, detail: startingDetail)
                            .transition(.opacity)
                    } else {
                        content
                            .id(section)
                            .transition(.opacity.combined(with: .offset(y: 6)))
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .animation(Motion.smooth, value: starting)
                Hairline()
                statusBar
            }
            .background(Theme.Palette.content)
            .animation(Motion.gentle, value: section)
            .animation(Motion.smooth, value: store.lastError != nil)
        }
        .overlay {
            if let failure = store.loadFailure {
                ZStack {
                    Theme.Palette.scrim.ignoresSafeArea()
                        .onTapGesture { store.loadFailure = nil }
                    LoadFailureCard(failure: failure)
                        .padding(Theme.Space.s6)
                }
                .transition(.opacity)
            }
        }
        .animation(Motion.smooth, value: store.loadFailure)
        .task {
            // Testing: SHUTTER_TEST_SHEET=newrequest:<model id> opens the New request sheet for that model.
            if let s = ProcessInfo.processInfo.environment["SHUTTER_TEST_SHEET"], s.hasPrefix("newrequest") {
                for _ in 0..<60 where !store.connection.isConnected { try? await Task.sleep(nanoseconds: 500_000_000) }
                try? await Task.sleep(nanoseconds: 3_000_000_000)   // the launch closes and reopens the window
                if Task.isCancelled { return }
                let parts = s.split(separator: ":", maxSplits: 1)
                newJobModel = parts.count > 1 ? String(parts[1]) : nil
                try? await Task.sleep(nanoseconds: 200_000_000)   // let the sheet read the model first
                showingNewJob = true
                FileHandle.standardError.write(Data("[test] New request sheet for \(newJobModel ?? "-")\n".utf8))
                return
            }
            // Testing: SHUTTER_TEST_LICENCE=<catalog id> asks that model's licence once the catalog is in.
            if let lid = ProcessInfo.processInfo.environment["SHUTTER_TEST_LICENCE"], !lid.isEmpty {
                for _ in 0..<60 where store.catalog.isEmpty { try? await Task.sleep(nanoseconds: 500_000_000) }
                store.licenceAsk = store.catalog.first { $0.id == lid }
            }
            // Testing: SHUTTER_TEST_LOAD=<model id> presses Load on that model once the engine answers.
            guard let id = ProcessInfo.processInfo.environment["SHUTTER_TEST_LOAD"], !id.isEmpty else { return }
            for _ in 0..<60 where !store.connection.isConnected { try? await Task.sleep(nanoseconds: 500_000_000) }
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            store.loadModel(id)
        }
        .background(PaperBackground())
        .background(PaperWindow())
        .tint(Theme.Palette.violet)
        .preferredColorScheme(.light)
        .ignoresSafeArea(.container, edges: .top)
        .frame(minWidth: Theme.Metric.minWindowWidth, minHeight: Theme.Metric.minWindowHeight)
        .background(shortcuts)
        .sheet(isPresented: $showingNewJob) {
            NewJobSheet(initialModel: newJobModel)
                .environmentObject(store)
        }
        .sheet(isPresented: Binding(get: { quit.asking }, set: { if !$0 && quit.asking { quit.reply(false) } })) {
            ConfirmSheet(title: "A request is still running. Quit and stop it?",
                         message: "Quitting Shuttrr also stops its engine, so the request stops where it is.",
                         confirmTitle: "Quit", destructive: true,
                         onCancel: { quit.reply(false) }, onConfirm: { quit.reply(true) })
        }
        .onChange(of: query) { _, newValue in
            if !newValue.isEmpty { section = .queue }
        }
        // Opening Requests clears the red failed badge; failures that land while it is open count as seen too.
        .onChange(of: section) { _, s in store.requestsOpen = s == .queue }
        .onAppear { store.requestsOpen = section == .queue }
        // files deleted (or re-downloaded) while Shuttrr was in the background: re-scan when it comes back, and
        // whenever the Models page opens
        .onReceive(NotificationCenter.default.publisher(for: NSWindow.didBecomeKeyNotification)) { _ in
            store.rescanModels()
        }
        .onChange(of: section) { _, s in if s == .models { store.rescanModels() } }
    }

    /// Skeleton placeholders while the engine starts (until it first answers, it exits, or 90 s pass).
    private var starting: Bool {
        // Testing: SHUTTER_TEST_SKELETON=1 keeps the placeholders on (for screenshots)
        if ProcessInfo.processInfo.environment["SHUTTER_TEST_SKELETON"] != nil { return true }
        return store.isStarting(engineExited: engine.lastExit != nil && !engine.isRunning)
    }

    private var startingDetail: String {
        store.startingFor < 20 ? "Starting engine…"
            : "Still starting. The first start after an update can take a minute."
    }

    @ObservedObject private var downloadsSheet = DownloadSheetState.shared
    private var showingDownloadsElsewhere: Bool { downloadsSheet.open }

    private func openNewJob(_ model: String?) {
        guard store.connection.isConnected else { return }
        newJobModel = model
        showingNewJob = true
    }

    // MARK: Toolbar

    private var toolbar: some View {
        HStack(spacing: Theme.Space.s3) {
            searchField
            if section == .queue {
                KitSegmented(options: QueueFilter.allCases.map { KitOption(value: $0, label: $0.label) },
                             selection: $filter)
                    .fixedSize()
            }
            Spacer(minLength: Theme.Space.s4)
            EngineStatusView()
        }
        .padding(.horizontal, Theme.Space.s5)
        .frame(height: Theme.Metric.toolbarHeight)
    }

    private var searchField: some View {
        HStack(spacing: Theme.Space.s2) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: Theme.Size.row))
                .foregroundStyle(Theme.Palette.muted)
            TextField("Search requests", text: $query)
                .textFieldStyle(.plain)
                .font(Theme.Fonts.row)
                .focused($searchFocused)
                .onExitCommand {
                    query = ""
                    searchFocused = false
                }
            HStack(spacing: Theme.Space.s1) {
                KeyCap(key: "⌘", filled: true)
                KeyCap(key: "F", filled: true)
            }
        }
        .padding(.horizontal, Theme.Space.s3)
        .frame(width: Theme.Metric.searchWidth, height: Theme.Metric.searchHeight)
        .background(RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous).fill(Theme.Palette.fill))
        .overlay(RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
            .strokeBorder(searchFocused ? Theme.Palette.violet : Theme.Palette.line, lineWidth: searchFocused ? 2 : Theme.Metric.hairline))
    }

    // MARK: Content

    @ViewBuilder
    private var content: some View {
        switch section {
        case .queue:
            QueueView(query: query, filter: filter, selectedJob: $selectedJob,
                      onNewJob: { openNewJob(nil) },
                      onShowAll: { query = ""; filter = .all })
        case .models:
            ModelsView(onNewJob: { openNewJob($0) })
        case .memory:
            MemoryView()
        case .server:
            ServerView()
        case .settings:
            SettingsView()
        }
    }

    // MARK: Status bar

    private var statusBar: some View {
        HStack(spacing: Theme.Space.s4) {
            Text(statusText)
                .font(Theme.Fonts.meta)
                .foregroundStyle(Theme.Palette.muted)
                .lineLimit(1)
                .truncationMode(.middle)
                .textSelection(.enabled)
            Spacer(minLength: Theme.Space.s4)
            KeyHint(keys: ["⌘", "N"], label: "New request")
            KeyHint(keys: ["⌘", "F"], label: "Search")
        }
        .padding(.horizontal, Theme.Space.s3)
        .frame(height: Theme.Metric.statusBarHeight)
    }

    private var statusText: String {
        if section == .queue, let id = selectedJob, let job = store.jobs.first(where: { $0.id == id }) {
            var parts = [job.id, job.request.model, job.state.label]
            if !job.message.isEmpty { parts.append(job.message) } else if job.state == .running { parts.append(job.detailLine) }
            if let out = job.outputPath { parts.append(out) }
            if let img = job.request.image { parts.append("image " + img) }
            if let audio = job.request.audio { parts.append("audio " + audio) }
            return parts.joined(separator: " · ")
        }
        switch store.connection {
        case .connecting: return "Connecting…"
        case .disconnected(let why): return starting ? "Starting engine…" : why
        case .connected:
            let n = store.jobs.filter { filter.includes($0.state) && QueueView.matches($0, query) }.count
            let shown = (query.isEmpty && filter == .all) ? "" : " shown"
            return "\(n) request\(n == 1 ? "" : "s")\(shown) · \(APIClient.current().base)"
        }
    }

    // MARK: Keyboard (⌘N new job, ⌘F search, ⌘1–4 sections)

    private var shortcuts: some View {
        ZStack {
            Button("New request") { openNewJob(nil) }
                .keyboardShortcut("n", modifiers: .command)
            Button("Search requests") {
                section = .queue
                searchFocused = true
            }
            .keyboardShortcut("f", modifiers: .command)
            ForEach(DashboardSection.allCases) { s in
                Button(s.rawValue) { section = s }
                    .keyboardShortcut(s.shortcut, modifiers: .command)
            }
        }
        .buttonStyle(.plain)
        .opacity(0)
        .frame(width: 0, height: 0)
        .accessibilityHidden(true)
    }
}

// MARK: - Sidebar

struct SidebarView: View {
    @EnvironmentObject private var store: EngineStore
    @Binding var section: DashboardSection
    var starting = false
    @Namespace private var selection

    init(section: Binding<DashboardSection>, starting: Bool = false) {
        self._section = section
        self.starting = starting
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s0) {
            HStack(spacing: Theme.Space.s2) {
                Image(systemName: "camera.aperture")
                    .font(.system(size: Theme.Size.section, weight: .medium))
                    .foregroundStyle(Theme.Palette.ink)      // the logo is black and white, never coloured
                Text("Shuttrr.").foregroundColor(Theme.Palette.ink)
                    .font(Theme.Fonts.brand)
            }
            .padding(.horizontal, Theme.Space.s2)
            .frame(height: Theme.Space.s6)
            .padding(.top, Theme.Space.s5)        // clears the window's traffic lights
            .padding(.bottom, Theme.Space.s2)

            ForEach(DashboardSection.allCases) { s in
                SidebarRow(symbol: s.symbol, title: s.rawValue, isOn: section == s, namespace: selection) {
                    if starting && s != .server && s != .settings {
                        SkeletonBlock(width: 22, height: 10, radius: 5)
                    } else if s == .queue && store.unseenFailures > 0 {
                        FailedBadge(count: store.unseenFailures)
                    } else if let t = countText(for: s) { t }
                } action: {
                    section = s
                }
            }

            Spacer(minLength: 0)
            EngineUpgradeRow()
            UpdateAvailableRow()
            SidebarVersionFooter()
        }
        .padding(.vertical, Theme.Space.s4)
        .padding(.horizontal, Theme.Space.s3)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    /// Grey count; the queue count turns green while a job runs (meaning, not decoration).
    private func countText(for s: DashboardSection) -> Text? {
        guard let sys = store.system else { return nil }
        let on = section == s      // the selected row is accent-filled: its count turns white
        switch s {
        case .queue:
            // counted from the request list, which leaves out Loads from the Models page
            let active = store.jobs.filter { $0.state == .queued || $0.state == .waitingMemory || $0.state == .running }.count
            guard active > 0 else { return nil }
            let running = store.runningJob != nil
            return Text("\(active)")
                .font(Theme.Fonts.monoMedium)
                .foregroundColor(on ? Theme.Palette.onAccent : (running ? Theme.Palette.good : Theme.Palette.muted))
                .monospacedDigit()
        case .models:
            guard !sys.pool.entries.isEmpty else { return nil }
            return Text("\(sys.pool.entries.count)").font(Theme.Fonts.mono).foregroundColor(on ? Theme.Palette.onAccent : Theme.Palette.muted).monospacedDigit()
        case .memory:
            return Text(Fmt.gb(MemoryParts(sys).free)).font(Theme.Fonts.mono).foregroundColor(on ? Theme.Palette.onAccent : Theme.Palette.muted).monospacedDigit()
        case .server:
            return nil
        case .settings:
            return nil
        }
    }
}

/// Red circle with the number of requests that failed since Requests was last opened.
struct FailedBadge: View {
    let count: Int

    var body: some View {
        Text("\(count)")
            .font(Theme.Fonts.metaSemibold)
            .monospacedDigit()
            .foregroundStyle(Color.white)
            .padding(.horizontal, count > 9 ? Theme.Space.s1 : 0)
            .frame(minWidth: Theme.Metric.badgeHeight, minHeight: Theme.Metric.badgeHeight)
            .background(Capsule().fill(Theme.Palette.danger))
            .help("\(count) request\(count == 1 ? "" : "s") failed since you last opened Requests")
            .accessibilityLabel("\(count) failed")
    }
}

struct SidebarRow<Trailing: View>: View {
    let symbol: String
    let title: String
    let isOn: Bool
    /// When set, the selection pill slides between rows that share the namespace.
    var namespace: Namespace.ID? = nil
    @ViewBuilder var trailing: () -> Trailing
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: Theme.Space.s3) {
                Image(systemName: symbol)
                    .font(.system(size: Theme.Size.row))
                    .foregroundStyle(isOn ? Theme.Palette.amber : Theme.Palette.ink2)
                    .frame(width: Theme.Metric.icon + Theme.Space.s1)
                Text(title)
                    .font(isOn ? Theme.Fonts.rowMedium : Theme.Fonts.row)
                    .foregroundStyle(isOn ? Theme.Palette.onAccent : Theme.Palette.ink)
                Spacer(minLength: Theme.Space.s2)
                trailing()
            }
            .padding(.horizontal, Theme.Space.s3)
            .frame(height: Theme.Metric.sidebarRowHeight)
            .background {
                if isOn {
                    if let namespace {
                        Capsule().fill(Theme.Palette.ink).matchedGeometryEffect(id: "selection", in: namespace)
                    } else {
                        Capsule().fill(Theme.Palette.ink)
                    }
                } else if hovering {
                    Capsule().fill(Theme.Palette.fill)
                }
            }
            .contentShape(Rectangle())
            .animation(Motion.snappy, value: isOn)
            .animation(Motion.quick, value: hovering)
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .accessibilityAddTraits(isOn ? .isSelected : [])
    }
}

// MARK: - Engine status (toolbar, right)

struct EngineStatusView: View {
    @EnvironmentObject private var store: EngineStore
    @EnvironmentObject private var engine: EngineProcess

    var body: some View {
        HStack(spacing: Theme.Space.s3) {
            HStack(spacing: Theme.Space.s2) {
                PulseDot(color: dotColor, active: store.runningJob != nil, size: Theme.Metric.dot)
                Text("Engine").foregroundColor(Theme.Palette.muted)
                    + Text(" " + word).fontWeight(.semibold).foregroundColor(Theme.Palette.ink)
            }
            if let v = store.system?.version, !v.isEmpty {
                Text("v" + v).foregroundStyle(Theme.Palette.muted)
            }
            if store.queuePaused {
                Text("Paused").foregroundStyle(Theme.Palette.muted)
            }
        }
        .font(Theme.Fonts.meta)
        .lineLimit(1)
        .fixedSize()
        .help(engine.isRunning ? "Started by Shuttrr" : "Started outside Shuttrr")
    }

    private var starting: Bool { store.isStarting(engineExited: engine.lastExit != nil && !engine.isRunning) }

    private var word: String {
        if starting && !store.connection.isConnected { return "starting" }
        switch store.connection {
        case .connected: return "live"
        case .connecting: return "connecting"
        case .disconnected: return "offline"
        }
    }

    private var dotColor: Color {
        if starting && !store.connection.isConnected { return Theme.Palette.muted }
        switch store.connection {
        case .connected: return Theme.Palette.good
        case .connecting: return Theme.Palette.muted
        case .disconnected: return Theme.Palette.danger
        }
    }
}

// MARK: - Banner shown only when something blocks the queue

struct BlockingBanner: View {
    @EnvironmentObject private var store: EngineStore
    @EnvironmentObject private var engine: EngineProcess

    var body: some View {
        if case .disconnected(let why) = store.connection {
            KitBanner(symbol: "bolt.horizontal",
                      title: "Cannot reach the engine",
                      detail: "\(why). Start it here or with “shutter serve”; this window reconnects by itself.") {
                if !engine.isRunning {
                    Button("Start engine") {
                        engine.start()
                        store.refreshNow()
                    }
                    .buttonStyle(.kit(.onBanner))
                }
            }
        } else if let svc = store.busyProtected.first {
            let more = store.busyProtected.count > 1 ? " (+\(store.busyProtected.count - 1) more)" : ""
            KitBanner(symbol: "pause.fill",
                      title: "Requests are waiting for a protected service",
                      detail: "\(svc.name): \(svc.busy ?? "busy")\(more)")
        } else if store.queuePaused {
            KitBanner(symbol: "pause.fill",
                      title: "Requests are paused",
                      detail: "Running jobs finish; nothing new starts until you resume.") {
                Button("Resume") { store.setQueuePaused(false) }
                    .buttonStyle(.kit(.onBanner))
            }
        }
    }
}
