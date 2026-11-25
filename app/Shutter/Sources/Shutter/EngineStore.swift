import Foundation
import SwiftUI

enum ConnectionState: Equatable {
    case connecting
    case connected
    case disconnected(String)

    var isConnected: Bool { self == .connected }
}

/// Live view of the engine: polls /v1/system and /v1/jobs every 2 s, follows the SSE
/// stream for smooth progress, and backs off (up to 30 s) while the engine is down.
@MainActor
final class EngineStore: ObservableObject {
    @Published private(set) var connection: ConnectionState = .connecting
    @Published private(set) var system: SystemInfo?
    @Published private(set) var jobs: [Job] = []
    @Published private(set) var models: [ModelInfo] = []
    /// Apps that sent requests lately (empty when the engine has no /v1/clients).
    @Published private(set) var clients: [ClientInfo] = []
    private var lastClientsFetch = Date.distantPast
    @Published private(set) var history: [MemoryPoint] = []
    @Published var lastError: String?
    /// Fast / Low memory (GET /v1/settings). Nil until the engine answers; an engine that
    /// has no /v1/settings keeps it nil and the Mode control stays hidden.
    @Published private(set) var settings: EngineSettings?

    /// Models the engine can download (empty when the engine has no /v1/catalog).
    @Published private(set) var catalog: [CatalogModel] = []
    @Published private(set) var catalogRestartNeeded = false
    /// Ids whose Download or Stop button was just pressed (until the engine answers).
    @Published private(set) var catalogPending: Set<String> = []
    private var lastCatalogFetch = Date.distantPast

    private var pollTask: Task<Void, Never>?
    private var eventsTask: Task<Void, Never>?
    private var eventsGeneration = 0
    private var lastEventsStart = Date.distantPast
    private var failures = 0
    private var wakeRequested = false
    private var lastModelsFetch = Date.distantPast
    private let maxHistory = 300 // 10 minutes at 2 s
    private var lastSettingsFetch = Date.distantPast
    /// Every local change gets a number; a reply or event only replaces local state once
    /// the newest change has been answered, so a slow reply cannot pull the thumb back.
    private var settingsSeq = 0
    private var settingsAcked = 0
    private var settingsPostTask: Task<Void, Never>?

    nonisolated init() {}

    // MARK: Derived state

    var runningJob: Job? { jobs.first { $0.state == .running } }
    var waitingJob: Job? { jobs.first { $0.state == .waitingMemory } }
    /// The job the panel should show: running first, then one waiting for memory.
    var currentJob: Job? { runningJob ?? waitingJob }

    var hasActiveJob: Bool {
        jobs.contains { $0.state == .running || $0.state == .waitingMemory || $0.state == .queued }
    }

    var queuePauseSupported: Bool { system?.queue.paused != nil }
    var queuePaused: Bool { system?.queue.paused ?? false }

    var busyProtected: [ProtectedService] {
        (system?.guardInfo.protected ?? []).filter { ($0.busy ?? "").isEmpty == false }
    }

    var installedModels: [ModelInfo] { models.filter { $0.installed } }

    /// True when the only thing to run is the built-in test model: time for the set-up card.
    var needsFirstModel: Bool {
        !catalog.isEmpty && !models.contains { $0.installed && $0.family != "dummy" }
    }
    var catalogDownloading: Bool { catalog.contains { $0.isDownloading } }

    /// SF Symbol for the menu bar: changes while a job runs or memory is refused.
    var menuBarSymbol: String {
        if waitingJob != nil || system?.memory.pressure == "critical" { return "memorychip" }
        if runningJob != nil { return "record.circle" }
        return "camera.aperture"
    }

    // MARK: Lifecycle

    func start() {
        guard pollTask == nil else { return }
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let delay = await self?.pollOnce() else { return }
                await self?.waitBeforeNextPoll(seconds: delay)
            }
        }
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
        eventsTask?.cancel()
        eventsTask = nil
    }

    /// Poll again right away (after an action, or when the engine was just started).
    func refreshNow() {
        failures = 0
        wakeRequested = true
    }

    private func waitBeforeNextPoll(seconds: Double) async {
        var waited = 0.0
        while waited < seconds && !wakeRequested && !Task.isCancelled {
            try? await Task.sleep(nanoseconds: 250_000_000)
            waited += 0.25
        }
        wakeRequested = false
    }

    /// One poll. Returns how long to wait before the next one.
    private func pollOnce() async -> Double {
        let api = APIClient.current()
        do {
            let sys = try await api.system()
            let list = try await api.jobs(limit: 50)
            system = sys
            jobs = list.filter { !$0.isModelOperation }
            failures = 0
            if connection != .connected {
                connection = .connected
                lastModelsFetch = .distantPast
            }
            appendHistory(sys.memory)
            let modelsEvery: Double = models.contains { $0.isLoading || $0.isWaiting } || !modelPending.isEmpty
                || !loadsStarted.isEmpty ? 1.5 : 30
            // the In memory section and the model rows come from the same pool: refetch the rows when it changes
            let poolKeys = sys.pool.entries.map(\.key).sorted()
            let poolChanged = poolKeys != lastPoolKeys
            lastPoolKeys = poolKeys
            if poolChanged || Date().timeIntervalSince(lastModelsFetch) > modelsEvery {
                lastModelsFetch = Date()
                if let m = try? await api.models() { setModels(m) }
            }
            if settings == nil || Date().timeIntervalSince(lastSettingsFetch) > 30 {
                lastSettingsFetch = Date()
                let seq = settingsSeq
                if let s = try? await api.settings(), seq == settingsSeq, settingsAcked == settingsSeq {
                    settings = s
                }
            } else if let brief = sys.settings {
                applyBrief(brief)
            }
            // Every poll while a download runs (for the progress bar), otherwise now and then.
            let catalogEvery: Double = catalogDownloading ? 1.5 : (needsFirstModel ? 20 : 120)
            if Date().timeIntervalSince(lastCatalogFetch) > catalogEvery {
                lastCatalogFetch = Date()
                if let list = try? await api.catalog() { applyCatalog(list) }
            }
            if Date().timeIntervalSince(lastClientsFetch) > 6 {
                lastClientsFetch = Date()
                clients = (try? await api.clients()) ?? []
            }
            startEventsIfNeeded()
            return 2
        } catch {
            failures += 1
            connection = .disconnected(Self.describe(error))
            system = nil
            eventsTask?.cancel()
            eventsTask = nil
            // 2, 4, 8, 16, 30, 30 ... seconds
            return min(30, 2 * pow(2, Double(min(failures - 1, 5))))
        }
    }

    private func appendHistory(_ m: MemoryInfo) {
        guard m.total > 0 else { return }
        history.append(MemoryPoint(date: Date(),
                                   availableGB: Double(m.available) / Fmt.gib,
                                   swapGB: Double(m.swapUsed) / Fmt.gib))
        if history.count > maxHistory {
            history.removeFirst(history.count - maxHistory)
        }
    }

    private static func describe(_ error: Error) -> String {
        if let u = error as? URLError {
            switch u.code {
            case .cannotConnectToHost, .networkConnectionLost, .cannotFindHost:
                return "Engine not running"
            case .timedOut:
                return "Engine not responding"
            default:
                break
            }
        }
        return error.localizedDescription
    }

    // MARK: Events (SSE)

    private func startEventsIfNeeded() {
        // At most one attempt every 10 s, so a missing /v1/events is not hammered.
        guard eventsTask == nil, Date().timeIntervalSince(lastEventsStart) > 10 else { return }
        lastEventsStart = Date()
        eventsGeneration += 1
        let generation = eventsGeneration
        eventsTask = Task { [weak self] in
            await self?.followEvents()
            // Stream ended: allow the next successful poll to reconnect, unless a newer
            // stream has already replaced this one.
            guard let self, self.eventsGeneration == generation else { return }
            self.eventsTask = nil
        }
    }

    private func followEvents() async {
        guard let url = try? APIClient.current().eventsURL() else { return }
        var req = URLRequest(url: url)
        req.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        do {
            let (bytes, response) = try await APIClient.streamSession.bytes(for: req)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else { return }
            for try await line in bytes.lines {
                if Task.isCancelled { return }
                guard line.hasPrefix("data:") else { continue }
                let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
                guard let data = payload.data(using: .utf8),
                      let event = try? JSONDecoder().decode(EngineEvent.self, from: data) else { continue }
                handle(event)
            }
        } catch {
            // Dropped stream; polling carries on and restarts it.
        }
    }

    private func handle(_ event: EngineEvent) {
        if event.kind == "job.progress", let id = event.jobId,
           let i = jobs.firstIndex(where: { $0.id == id }) {
            if let p = event.data["progress"]?.doubleValue { jobs[i].progress = p }
            if let s = event.data["stage"]?.stringValue { jobs[i].stage = s }
            if let e = event.data["eta_s"]?.doubleValue { jobs[i].etaS = e }
            jobs[i].stateRaw = JobState.running.rawValue
        } else if event.kind == "settings.changed" {
            if let mode = event.data["mode"]?.stringValue, let eff = event.data["efficiency"]?.doubleValue {
                applyBrief(mode: mode, efficiency: eff)
            }
        } else if event.kind.hasPrefix("job.") || event.kind.hasPrefix("model.") {
            wakeRequested = true
        }
    }

    // MARK: Modes (docs/MODES.md)

    private func applyBrief(_ b: SettingsBrief) { applyBrief(mode: b.mode, efficiency: b.efficiency) }

    /// Follow a change made elsewhere (dashboard, CLI), unless a local change is in flight.
    private func applyBrief(mode: String, efficiency: Double) {
        guard var s = settings, settingsAcked == settingsSeq else { return }
        guard s.mode != mode || abs(s.efficiency - efficiency) > 1e-6 else { return }
        s.mode = mode
        s.efficiency = efficiency
        s.rung = s.rungID(mode: mode, efficiency: efficiency)
        settings = s
    }

    func setMode(_ mode: String) {
        guard var s = settings, s.mode != mode else { return }
        s.mode = mode
        s.rung = s.rungID(mode: mode, efficiency: s.efficiency)
        settings = s
        postSettings(after: 0)
    }

    /// Set the efficiency (0...1). `commit` sends now (release / key press); otherwise the
    /// POST waits for 250 ms of quiet so a drag sends one request, not dozens.
    func setEfficiency(_ value: Double, commit: Bool = true) {
        guard var s = settings else { return }
        let v = (min(1, max(0, value)) * 10_000).rounded() / 10_000
        s.efficiency = v
        s.rung = s.rungID(mode: s.mode, efficiency: v)
        settings = s
        postSettings(after: commit ? 0 : 0.25)
    }

    /// Send the current mode and efficiency (both, so a debounce cancelled by a newer
    /// change never drops a value).
    private func postSettings(after delay: Double) {
        settingsSeq += 1
        let seq = settingsSeq
        settingsPostTask?.cancel()
        settingsPostTask = Task { [weak self] in
            if delay > 0 {
                try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                if Task.isCancelled { return }
            }
            guard let self, let current = self.settings else { return }
            do {
                let s = try await APIClient.current().updateSettings(mode: current.mode, efficiency: current.efficiency)
                if seq == self.settingsSeq { self.settings = s }
                self.settingsAcked = max(self.settingsAcked, seq)
            } catch {
                self.settingsAcked = max(self.settingsAcked, seq)
                if seq == self.settingsSeq {
                    self.lastError = error.localizedDescription
                    self.lastSettingsFetch = .distantPast // fetch the engine's values again
                    self.wakeRequested = true
                }
            }
        }
    }

    /// "Fast" / "Low memory · Lower memory" for a job row; nil when the job carries no mode.
    func modeText(for job: Job) -> String? {
        guard let info = job.modeInfo else { return nil }
        if info.mode == "fast" {
            if let r = info.rung, let s = settings, r != s.ladder.first?.id {
                return "Fast · \(s.label(forRung: r))"
            }
            return "Fast"
        }
        var rungID = info.rung
        if rungID == nil, let s = settings, let e = info.efficiency {
            rungID = s.rungID(mode: "low_memory", efficiency: e)
        }
        guard let r = rungID else { return "Low memory" }
        return "Low memory · \(settings?.label(forRung: r) ?? r)"
    }

    /// The most recent finished Fast job that stepped down to fit, if any.
    var lastSteppedDown: (job: Job, rung: String)? {
        let done = jobs.filter { $0.state == .done && $0.finishedAt != nil }
            .sorted { ($0.finishedAt ?? 0) > ($1.finishedAt ?? 0) }
        guard let last = done.first, let info = last.modeInfo, info.mode == "fast",
              info.steppedDownFrom != nil, let r = info.rung else { return nil }
        return (last, r)
    }

    // MARK: Actions

    func submit(_ request: NewJobRequest) async -> Bool {
        do {
            _ = try await APIClient.current().submit(request)
            refreshNow()
            return true
        } catch {
            lastError = error.localizedDescription
            return false
        }
    }

    func cancel(_ job: Job) {
        Task {
            do {
                _ = try await APIClient.current().cancel(id: job.id)
            } catch {
                lastError = error.localizedDescription
            }
            refreshNow()
        }
    }

    /// Model ids whose Load / Unload / Cancel was just pressed (until the engine answers).
    @Published private(set) var modelPending: Set<String> = []

    /// The red popup in the main window (memory short, or a load that failed).
    @Published var loadFailure: LoadFailure?
    /// Models this app asked to load: a failure reported for them opens the popup.
    private var loadsStarted: Set<String> = []
    private var lastPoolKeys: [String] = []

    func loadModel(_ id: String, wait: Bool = false) {
        loadFailure = nil
        modelAction(id) { [weak self] api in
            if let refusal = try await api.loadModel(id: id, wait: wait) {
                let name = self?.models.first { $0.id == id }?.displayName ?? refusal.modelName
                self?.loadFailure = LoadFailure.memory(refusal, name: refusal.modelName.isEmpty ? name : refusal.modelName)
            } else {
                self?.loadsStarted.insert(id)
            }
        }
    }

    func unloadModel(_ id: String) {
        loadsStarted.remove(id)
        modelAction(id) { try await $0.unloadModel(id: id) }
    }

    /// Opens the popup for a load this app started that the engine reports as failed.
    private func noticeLoadFailures(_ list: [ModelInfo]) {
        for m in list where loadsStarted.contains(m.id) && !m.loadError.isEmpty {
            loadsStarted.remove(m.id)
            loadFailure = LoadFailure.other(modelID: m.id, name: m.displayName, reason: m.loadError)
        }
        for m in list where m.isLoaded { loadsStarted.remove(m.id) }
    }

    private func setModels(_ list: [ModelInfo]) {
        models = list
        noticeLoadFailures(list)
    }

    private func modelAction(_ id: String, _ call: @escaping (APIClient) async throws -> Void) {
        guard !modelPending.contains(id) else { return }
        modelPending.insert(id)
        Task {
            let api = APIClient.current()
            do {
                try await call(api)
                lastError = nil
            } catch {
                lastError = error.localizedDescription
            }
            if let m = try? await api.models() { setModels(m) }
            lastModelsFetch = Date()
            modelPending.remove(id)
            refreshNow()
        }
    }

    /// Unload one component, or every unpinned one when `key` is nil.
    func release(key: String?) {
        Task {
            let api = APIClient.current()
            do {
                _ = try await api.releasePool(key: key)
            } catch {
                lastError = error.localizedDescription
            }
            // refresh the model rows at once, so their Load / Unload button matches the In memory section
            if let m = try? await api.models() { setModels(m) }
            lastModelsFetch = Date()
            refreshNow()
        }
    }

    func setQueuePaused(_ paused: Bool) {
        Task {
            do {
                try await APIClient.current().setQueuePaused(paused)
            } catch {
                lastError = error.localizedDescription
            }
            refreshNow()
        }
    }

    // MARK: Model downloads

    private func applyCatalog(_ list: CatalogList) {
        let wasDownloading = catalogDownloading
        catalog = list.models
        catalogRestartNeeded = list.restartNeeded
        if wasDownloading && !catalogDownloading {
            lastModelsFetch = .distantPast     // a download just ended: the new model shows up
            lastSettingsFetch = .distantPast
            refreshNow()
        }
    }

    func startDownload(_ id: String) { catalogAction(id) { try await $0.startDownload(id: id) } }
    func stopDownload(_ id: String) { catalogAction(id) { try await $0.stopDownload(id: id) } }

    private func catalogAction(_ id: String, _ call: @escaping (APIClient) async throws -> CatalogList) {
        guard !catalogPending.contains(id) else { return }
        catalogPending.insert(id)
        Task {
            do {
                applyCatalog(try await call(APIClient.current()))
            } catch {
                lastError = error.localizedDescription
            }
            catalogPending.remove(id)
            lastCatalogFetch = Date()
            refreshNow()
        }
    }

    func reloadModels() {
        Task {
            do {
                setModels(try await APIClient.current().models())
                lastModelsFetch = Date()
            } catch {
                lastError = error.localizedDescription
            }
        }
    }
}
