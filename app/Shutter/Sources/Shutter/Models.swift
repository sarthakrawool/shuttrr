import Foundation

// JSON shapes served by the Shutter engine (see shutter/types.py and server/api.py).
// Every field is optional on the wire: the server is still evolving, so decoding never
// fails on a missing or mistyped field. It falls back to a default instead.

// MARK: - Tolerant decoding helpers

extension KeyedDecodingContainer {
    /// Decodes `key` if present and well-typed, otherwise returns `fallback`.
    func value<T: Decodable>(_ key: Key, _ fallback: T) -> T {
        if let v = try? decodeIfPresent(T.self, forKey: key) {
            return v
        }
        return fallback
    }

    /// Decodes an optional value; a missing, null or mistyped field gives nil.
    func optional<T: Decodable>(_ key: Key) -> T? {
        if let v = try? decodeIfPresent(T.self, forKey: key) {
            return v
        }
        return nil
    }

    /// An optional integer that may arrive as a float; missing or mistyped gives nil.
    func optionalInt(_ key: Key) -> Int? {
        if let v = try? decodeIfPresent(Int.self, forKey: key) {
            return v
        }
        if let d = try? decodeIfPresent(Double.self, forKey: key), d.isFinite {
            return Int(d)
        }
        return nil
    }

    /// Integers that may arrive as floats (e.g. 1.7e10).
    func int(_ key: Key, _ fallback: Int = 0) -> Int {
        if let v = try? decodeIfPresent(Int.self, forKey: key) {
            return v
        }
        if let d = try? decodeIfPresent(Double.self, forKey: key), d.isFinite {
            return Int(d)
        }
        return fallback
    }
}

// MARK: - Free-form JSON (job.result, request.params, event.data)

enum JSONValue: Codable, Equatable, Sendable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() {
            self = .null
        } else if let b = try? c.decode(Bool.self) {
            self = .bool(b)
        } else if let d = try? c.decode(Double.self) {
            self = .number(d)
        } else if let s = try? c.decode(String.self) {
            self = .string(s)
        } else if let a = try? c.decode([JSONValue].self) {
            self = .array(a)
        } else if let o = try? c.decode([String: JSONValue].self) {
            self = .object(o)
        } else {
            self = .null
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let b): try c.encode(b)
        case .number(let d): try c.encode(d)
        case .string(let s): try c.encode(s)
        case .array(let a): try c.encode(a)
        case .object(let o): try c.encode(o)
        }
    }

    var stringValue: String? {
        switch self {
        case .string(let s): return s
        case .number(let d): return String(d)
        case .bool(let b): return b ? "true" : "false"
        default: return nil
        }
    }

    var doubleValue: Double? {
        switch self {
        case .number(let d): return d
        case .string(let s): return Double(s)
        case .bool(let b): return b ? 1 : 0
        default: return nil
        }
    }

    subscript(key: String) -> JSONValue? {
        if case .object(let o) = self { return o[key] }
        return nil
    }
}

// MARK: - /v1/system

struct MemoryInfo: Decodable, Sendable {
    var total: Int = 0
    var available: Int = 0
    var swapUsed: Int = 0
    var pressure: String = "normal"

    var used: Int { max(0, total - available) }

    enum CodingKeys: String, CodingKey { case total, available, swapUsed = "swap_used", pressure }

    init() {}
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        total = c.int(.total)
        available = c.int(.available)
        swapUsed = c.int(.swapUsed)
        pressure = c.value(.pressure, "normal")
    }
}

struct ProtectedService: Decodable, Identifiable, Sendable {
    var name: String = ""
    var busy: String?
    var id: String { name }

    enum CodingKeys: String, CodingKey { case name, busy }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        name = c.value(.name, "")
        busy = c.optional(.busy)
    }
}

struct GuardInfo: Decodable, Sendable {
    var headroomBytes: Int = 0
    /// True when the engine sizes the headroom from this Mac's memory (no headroom_gb set).
    var headroomAuto: Bool = false
    var protected: [ProtectedService] = []

    enum CodingKeys: String, CodingKey { case headroomBytes = "headroom_bytes", headroomAuto = "headroom_auto", protected }

    init() {}
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        headroomBytes = c.int(.headroomBytes)
        headroomAuto = c.value(.headroomAuto, false)
        protected = c.value(.protected, [])
    }
}

struct PoolEntry: Decodable, Identifiable, Sendable {
    var key: String = ""
    var sizeBytes: Int = 0
    var idleS: Double = 0
    var pinned: Bool = false
    /// "resident" (whole component in memory) or "streamed" (weights read from SSD as needed).
    var residency: String = "resident"
    var id: String { key }
    var isStreamed: Bool { residency == "streamed" }

    enum CodingKeys: String, CodingKey { case key, sizeBytes = "size_bytes", idleS = "idle_s", pinned, residency }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        key = c.value(.key, "")
        sizeBytes = c.int(.sizeBytes)
        idleS = c.value(.idleS, 0.0)
        pinned = c.value(.pinned, false)
        residency = c.value(.residency, "resident")
    }
}

/// Requests finished lately and the speed of the one running (`/v1/system` → `throughput`).
struct Throughput: Decodable, Sendable {
    struct Running: Decodable, Sendable {
        var step: Int = 0
        var steps: Int = 0
        var stepsPerS: Double?

        enum CodingKeys: String, CodingKey { case step, steps, stepsPerS = "steps_per_s" }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            step = c.int(.step)
            steps = c.int(.steps)
            stepsPerS = c.optional(.stepsPerS)
        }
    }

    var finished: Int = 0
    var meanSeconds: Double?
    var running: Running?

    enum CodingKeys: String, CodingKey { case finished, meanSeconds = "mean_seconds", running }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        finished = c.int(.finished)
        meanSeconds = c.optional(.meanSeconds)
        running = c.optional(.running)
    }

    /// "0.4 steps/s" or "2.5 s/step" for the running request; nil when the model reports no steps.
    var speedText: String? {
        guard let v = running?.stepsPerS, v > 0 else { return nil }
        return v >= 1 ? String(format: "%.1f steps/s", v) : String(format: "%.1f s/step", 1 / v)
    }
}

/// An app that sent requests lately (`/v1/clients`), e.g. "comfyui/0.46".
struct ClientInfo: Decodable, Identifiable, Sendable {
    var client: String = ""
    var requests: Int = 0
    var idleS: Double = 0
    var id: String { client }

    enum CodingKeys: String, CodingKey { case client, requests, idleS = "idle_s" }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        client = c.value(.client, "")
        requests = c.int(.requests)
        idleS = c.value(.idleS, 0.0)
    }
}

struct ClientList: Decodable, Sendable {
    var clients: [ClientInfo] = []
    enum CodingKeys: String, CodingKey { case clients }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        clients = c.value(.clients, [])
    }
}

struct PoolInfo: Decodable, Sendable {
    var totalBytes: Int = 0
    var entries: [PoolEntry] = []

    enum CodingKeys: String, CodingKey { case totalBytes = "total_bytes", entries }

    init() {}
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        totalBytes = c.int(.totalBytes)
        entries = c.value(.entries, [])
    }
}

struct QueueCounts: Decodable, Sendable {
    var queued: Int = 0
    var running: Int = 0
    var waitingMemory: Int = 0
    var done: Int = 0
    var failed: Int = 0
    /// Present only when the engine supports pausing the queue.
    var paused: Bool?

    enum CodingKeys: String, CodingKey {
        case queued, running, waitingMemory = "waiting_memory", done, failed, paused
    }

    init() {}
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        queued = c.int(.queued)
        running = c.int(.running)
        waitingMemory = c.int(.waitingMemory)
        done = c.int(.done)
        failed = c.int(.failed)
        paused = c.optional(.paused)
    }
}

struct GpuHolder: Decodable, Sendable {
    var pid: Int = 0
    var label: String = ""

    enum CodingKeys: String, CodingKey { case pid, label }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        pid = c.int(.pid)
        label = c.value(.label, "")
    }
}

struct GpuLockInfo: Decodable, Sendable {
    var held: Bool = false
    var holder: GpuHolder?

    enum CodingKeys: String, CodingKey { case held, holder }

    init() {}
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        held = c.value(.held, false)
        holder = c.optional(.holder)
    }
}

struct SystemInfo: Decodable, Sendable {
    var version: String = ""
    var memory = MemoryInfo()
    var guardInfo = GuardInfo()
    var pool = PoolInfo()
    var queue = QueueCounts()
    var gpuLock = GpuLockInfo()
    /// The engine-wide mode defaults; nil from an engine without modes.
    var settings: SettingsBrief?
    /// nil from an engine that does not report throughput.
    var throughput: Throughput?

    enum CodingKeys: String, CodingKey {
        case version, memory, guardInfo = "guard", pool, queue, gpuLock = "gpu_lock", settings, throughput
    }

    init() {}
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        version = c.value(.version, "")
        memory = c.value(.memory, MemoryInfo())
        guardInfo = c.value(.guardInfo, GuardInfo())
        pool = c.value(.pool, PoolInfo())
        queue = c.value(.queue, QueueCounts())
        gpuLock = c.value(.gpuLock, GpuLockInfo())
        settings = c.optional(.settings)
        throughput = c.optional(.throughput)
    }
}

// MARK: - /v1/models

struct ModelInfo: Decodable, Identifiable, Hashable, Sendable {
    var id: String = ""
    var family: String = ""
    var kind: String = ""
    var dir: String = ""
    var installed: Bool = false
    /// not_loaded, loading, waiting_for_memory, loaded ("" from an engine without load support)
    var state: String = ""
    var loadedBytes: Int = 0
    /// The memory guard's sentence while a load waits.
    var waiting: String = ""
    /// The queued load request, so Cancel can cancel it.
    var loadRequest: String = ""
    /// A name for people from the engine (catalog name, "Test model"); empty from older engines.
    var name: String = ""
    /// Why the last load failed (other than memory), until the next Load or Unload.
    var loadError: String = ""

    enum CodingKeys: String, CodingKey {
        case id, family, kind, dir, installed, state, waiting, name
        case loadedBytes = "loaded_bytes"
        case loadRequest = "load_request"
        case loadError = "load_error"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = c.value(.id, "")
        family = c.value(.family, "")
        kind = c.value(.kind, "")
        dir = c.value(.dir, "")
        installed = c.value(.installed, false)
        state = c.value(.state, "")
        loadedBytes = c.value(.loadedBytes, 0)
        waiting = c.value(.waiting, "")
        loadRequest = c.value(.loadRequest, "")
        name = c.value(.name, "")
        loadError = c.value(.loadError, "")
    }

    var supportsLoad: Bool { !state.isEmpty }
    var isLoaded: Bool { state == "loaded" }
    var isLoading: Bool { state == "loading" }
    var isWaiting: Bool { state == "waiting_for_memory" }
    /// A name for people: "dummy" is the built-in test model.
    var displayName: String { !name.isEmpty ? name : (family == "dummy" ? "Test model" : id) }
}

/// POST /v1/models/<id>/load refused by the memory guard (409 `refused`).
struct LoadRefusal: Decodable, Sendable, Equatable {
    var kind: String = ""
    var model: String = ""
    var modelName: String = ""
    var peakGB: Double = 0
    var headroomGB: Double = 0
    var availableGB: Double = 0
    var shortByGB: Double = 0
    var lowest: Bool = false
    var reason: String = ""

    enum CodingKeys: String, CodingKey {
        case kind, model, lowest, reason
        case modelName = "model_name"
        case peakGB = "peak_gb"
        case headroomGB = "headroom_gb"
        case availableGB = "available_gb"
        case shortByGB = "short_by_gb"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        kind = c.value(.kind, "")
        model = c.value(.model, "")
        modelName = c.value(.modelName, "")
        peakGB = c.value(.peakGB, 0)
        headroomGB = c.value(.headroomGB, 0)
        availableGB = c.value(.availableGB, 0)
        shortByGB = c.value(.shortByGB, 0)
        lowest = c.value(.lowest, false)
        reason = c.value(.reason, "")
    }
}

/// What the red load popup shows.
struct LoadFailure: Identifiable, Equatable {
    let id = UUID()
    let modelID: String
    let title: String
    let body: String
    let help: String
    /// Memory short: offer "Load when memory frees up"; otherwise offer View log.
    let isMemory: Bool

    static func memory(_ r: LoadRefusal, name: String) -> LoadFailure {
        func g(_ v: Double) -> String { String(format: "%.1f GB", v) }
        if r.kind != "memory" {
            return LoadFailure(modelID: r.model, title: "\(name) cannot load right now", body: r.reason,
                               help: "Try again in a moment.", isMemory: true)
        }
        let body = "It needs \(g(r.peakGB)) at its peak, plus \(g(r.headroomGB)) kept free for macOS. This Mac has "
            + "\(g(r.availableGB)) available right now, so it is \(g(r.shortByGB)) short."
        let help = r.lowest ? "Quit other apps to free memory." : "Quit other apps to free memory, or choose a lower memory setting."
        return LoadFailure(modelID: r.model, title: "Not enough memory to load \(name)", body: body, help: help,
                           isMemory: true)
    }

    static func other(modelID: String, name: String, reason: String) -> LoadFailure {
        let first = reason.split(separator: "\n").first.map(String.init) ?? reason
        return LoadFailure(modelID: modelID, title: "\(name) could not be loaded", body: first,
                           help: "The engine log has the details.", isMemory: false)
    }
}

struct ModelList: Decodable, Sendable {
    var models: [ModelInfo] = []
    enum CodingKeys: String, CodingKey { case models }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        models = c.value(.models, [])
    }
}

// MARK: - Model downloads (GET /v1/catalog)

struct CatalogDownload: Decodable, Hashable, Sendable {
    /// downloading, done, failed, stopped, cancelled
    var state: String = ""
    var doneBytes: Int = 0
    var totalBytes: Int = 0
    var error: String = ""

    enum CodingKeys: String, CodingKey {
        case state, error
        case doneBytes = "done_bytes"
        case totalBytes = "total_bytes"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        state = c.value(.state, "")
        doneBytes = c.value(.doneBytes, 0)
        totalBytes = c.value(.totalBytes, 0)
        error = c.value(.error, "")
    }

    var isRunning: Bool { state == "downloading" }
    var fraction: Double { totalBytes > 0 ? min(1, Double(doneBytes) / Double(totalBytes)) : 0 }
}

/// How a catalog model fits this Mac's memory (its total, from the engine's per-step estimates).
struct CatalogFit: Decodable, Hashable, Sendable {
    /// fits, low_memory, needs, unknown
    var level: String = "unknown"
    /// "Fits", "Fits in Low memory", "Needs 24 GB"; empty when unknown.
    var label: String = ""
    var detail: String = ""

    enum CodingKeys: String, CodingKey { case level, label, detail }

    init() {}

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        level = c.value(.level, "unknown")
        label = c.value(.label, "")
        detail = c.value(.detail, "")
    }
}

/// Free disk space a download needs; `label` is empty when it fits.
struct CatalogDisk: Decodable, Hashable, Sendable {
    var ok: Bool = true
    var needBytes: Int = 0
    var label: String = ""

    enum CodingKeys: String, CodingKey { case ok, label, needBytes = "need_bytes" }

    init() {}

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        ok = c.value(.ok, true)
        needBytes = c.value(.needBytes, 0)
        label = c.value(.label, "")
    }
}

/// The request "Try a Sample" sends with a model (sized to finish in about a minute).
struct CatalogSample: Decodable, Hashable, Sendable {
    var prompt: String = ""
    var width: Int = 512
    var height: Int = 320
    var frames: Int = 17
    var steps: Int = 12
    var seed: Int = 42

    enum CodingKeys: String, CodingKey { case prompt, width, height, frames, steps, seed }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        prompt = c.value(.prompt, "")
        width = c.value(.width, 512)
        height = c.value(.height, 320)
        frames = c.value(.frames, 17)
        steps = c.value(.steps, 12)
        seed = c.value(.seed, 42)
    }
}

struct CatalogModel: Decodable, Identifiable, Hashable, Sendable {
    var id: String = ""
    var name: String = ""
    var summary: String = ""
    var kind: String = ""
    var sizeBytes: Int = 0
    var minMemoryBytes: Int = 0
    var freeBytes: Int = 0
    var fitsThisMac: Bool = true
    var installed: Bool = false
    var dir: String = ""
    var notes: [String] = []
    var download: CatalogDownload?
    var fit = CatalogFit()
    var disk = CatalogDisk()
    /// The starter model for this Mac (under 24 GB: the smallest Wan).
    var recommended: Bool = false
    var sample: CatalogSample?

    enum CodingKeys: String, CodingKey {
        case id, name, summary, kind, installed, dir, notes, download, fit, disk, recommended, sample
        case sizeBytes = "size_bytes"
        case minMemoryBytes = "min_memory_bytes"
        case freeBytes = "free_bytes"
        case fitsThisMac = "fits_this_mac"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = c.value(.id, "")
        name = c.value(.name, "")
        summary = c.value(.summary, "")
        kind = c.value(.kind, "")
        sizeBytes = c.value(.sizeBytes, 0)
        minMemoryBytes = c.value(.minMemoryBytes, 0)
        freeBytes = c.value(.freeBytes, 0)
        fitsThisMac = c.value(.fitsThisMac, true)
        installed = c.value(.installed, false)
        dir = c.value(.dir, "")
        notes = c.value(.notes, [])
        download = c.optional(.download)
        fit = c.optional(.fit) ?? CatalogFit()
        disk = c.optional(.disk) ?? CatalogDisk()
        recommended = c.value(.recommended, false)
        sample = c.optional(.sample)
    }

    var isDownloading: Bool { download?.isRunning ?? false }
}

struct CatalogList: Decodable, Sendable {
    var models: [CatalogModel] = []
    /// The Rust engine reads its models once, so a new download needs an engine restart.
    var restartNeeded: Bool = false

    enum CodingKeys: String, CodingKey {
        case models
        case restartNeeded = "restart_needed"
    }

    init() {}

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        models = c.value(.models, [])
        restartNeeded = c.value(.restartNeeded, false)
    }
}

// MARK: - Jobs

enum JobState: String, Sendable {
    case queued, waitingMemory = "waiting_memory", running, done, failed, cancelled, unknown

    var isTerminal: Bool { self == .done || self == .failed || self == .cancelled }

    var label: String {
        switch self {
        case .queued: return "Queued"
        case .waitingMemory: return "Waiting for memory"
        case .running: return "Running"
        case .done: return "Done"
        case .failed: return "Failed"
        case .cancelled: return "Cancelled"
        case .unknown: return "Unknown"
        }
    }
}

struct JobRequestInfo: Decodable, Sendable {
    var model: String = ""
    var prompt: String = ""
    var image: String?
    var audio: String?
    var output: String?
    var framing: String = "auto"
    var durationS: Double?
    var seed: Int = 42
    var priority: Int = 0
    var params: [String: JSONValue] = [:]

    enum CodingKeys: String, CodingKey {
        case model, prompt, image, audio, output, framing, durationS = "duration_s", seed, priority, params
    }

    init() {}
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        model = c.value(.model, "")
        prompt = c.value(.prompt, "")
        image = c.optional(.image)
        audio = c.optional(.audio)
        output = c.optional(.output)
        framing = c.value(.framing, "auto")
        durationS = c.optional(.durationS)
        seed = c.int(.seed, 42)
        priority = c.int(.priority)
        params = c.value(.params, [:])
    }
}

struct Job: Decodable, Identifiable, Sendable {
    var id: String = ""
    var stateRaw: String = "queued"
    var createdAt: Double = 0
    var startedAt: Double?
    var finishedAt: Double?
    var progress: Double = 0
    var stage: String = ""
    var message: String = ""
    var result: [String: JSONValue] = [:]
    var node: String = "local"
    var request = JobRequestInfo()
    /// Optional, not in types.py yet: seconds remaining as estimated by the engine.
    var etaS: Double?

    enum CodingKeys: String, CodingKey {
        case id, stateRaw = "state", createdAt = "created_at", startedAt = "started_at"
        case finishedAt = "finished_at", progress, stage, message, result, node, request
        case etaS = "eta_s"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = c.value(.id, "")
        stateRaw = c.value(.stateRaw, "queued")
        createdAt = c.value(.createdAt, 0.0)
        startedAt = c.optional(.startedAt)
        finishedAt = c.optional(.finishedAt)
        progress = c.value(.progress, 0.0)
        stage = c.value(.stage, "")
        message = c.value(.message, "")
        result = c.value(.result, [:])
        node = c.value(.node, "local")
        request = c.value(.request, JobRequestInfo())
        etaS = c.optional(.etaS)
    }

    var state: JobState { JobState(rawValue: stateRaw) ?? .unknown }

    /// A Load from the Models page (params._action = "load"): it runs through the queue, but it is not a request,
    /// so the Requests list and its counts leave it out. The model's row shows its progress instead.
    var isModelOperation: Bool { request.params["_action"]?.stringValue == "load" }

    /// Path of the finished file (adapters must put it in result["output"]).
    var outputPath: String? {
        for k in ["output", "output_path", "path"] {
            if let s = result[k]?.stringValue, !s.isEmpty { return s }
        }
        return state == .done ? request.output : nil
    }

    /// Peak memory in bytes, if the engine recorded it.
    var peakBytes: Int? {
        for k in ["peak_bytes", "peak_memory_bytes", "peak_memory", "peak"] {
            if let v = result[k] {
                if let d = v.doubleValue { return Int(d) }
                if let d = v["bytes"]?.doubleValue { return Int(d) }
            }
        }
        return nil
    }

    /// Wall time in seconds: finished - started, or elapsed so far while running.
    var durationS: Double? {
        guard let start = startedAt, start > 0 else { return nil }
        let end = finishedAt ?? Date().timeIntervalSince1970
        return max(0, end - start)
    }

    /// Remaining seconds: engine value if given, else extrapolated from progress.
    var remainingS: Double? {
        if let e = etaS ?? result["eta_s"]?.doubleValue { return max(0, e) }
        guard state == .running, progress > 0.02, progress < 1, let elapsed = durationS else { return nil }
        return elapsed / progress * (1 - progress)
    }
}

struct JobList: Decodable, Sendable {
    var jobs: [Job] = []
    enum CodingKeys: String, CodingKey { case jobs }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        jobs = c.value(.jobs, [])
    }
}

/// Body of POST /v1/jobs (JobRequest.from_dict). Nil fields are omitted.
struct NewJobRequest: Encodable, Sendable {
    var model: String
    var prompt: String
    var image: String?
    var audio: String?
    var output: String?
    var framing: String = "auto"
    var durationS: Double?
    var seed: Int
    var priority: Int = 0
    /// Per-job overrides such as `mode` / `efficiency`; nil sends nothing (engine defaults).
    var params: [String: JSONValue]? = nil
    /// Size, length and steps (text-to-video models); nil sends nothing.
    var width: Int? = nil
    var height: Int? = nil
    var frames: Int? = nil
    var steps: Int? = nil

    enum CodingKeys: String, CodingKey {
        // `framing` is not sent: the engine no longer has it.
        case model, prompt, image, audio, output, durationS = "duration_s", seed, priority, params
        case width, height, frames, steps
    }
}

// MARK: - Events (SSE) and pool release

struct EngineEvent: Decodable, Sendable {
    var kind: String = ""
    var jobId: String?
    var data: [String: JSONValue] = [:]
    var ts: Double = 0

    enum CodingKeys: String, CodingKey { case kind, jobId = "job_id", data, ts }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        kind = c.value(.kind, "")
        jobId = c.optional(.jobId)
        data = c.value(.data, [:])
        ts = c.value(.ts, 0.0)
    }
}

struct ReleaseResponse: Decodable, Sendable {
    var released: [String] = []
    enum CodingKeys: String, CodingKey { case released }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        released = c.value(.released, [])
    }
}

/// One sample for the rolling memory chart.
struct MemoryPoint: Identifiable, Sendable {
    let id = UUID()
    let date: Date
    let availableGB: Double
    let swapGB: Double
}

// MARK: - Modes (/v1/settings, docs/MODES.md)

/// `{mode, efficiency, rung}` in /v1/system and in `settings.changed` events.
struct SettingsBrief: Decodable, Equatable, Sendable {
    var mode: String = "fast"
    var efficiency: Double = 0.5
    var rung: String = "resident"

    enum CodingKeys: String, CodingKey { case mode, efficiency, rung }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        mode = c.value(.mode, "fast")
        efficiency = c.value(.efficiency, 0.5)
        rung = c.value(.rung, "resident")
    }
}

/// One step of the low-memory ladder. `peakBytes` / `seconds` are measured (measured.json);
/// `estPeakBytes` / `estSeconds` are estimates, shown with an "estimate" tag.
struct ModeRung: Decodable, Identifiable, Equatable, Sendable {
    var at: Double = 0
    var id: String = ""
    var label: String = ""
    var detail: String = ""
    var peakBytes: Int?
    var seconds: Double?
    var estPeakBytes: Int?
    var estSeconds: Double?
    var identical: Bool = true

    enum CodingKeys: String, CodingKey {
        case at, id, label, detail, peakBytes = "peak_bytes", seconds
        case estPeakBytes = "est_peak_bytes", estSeconds = "est_seconds", identical
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        at = c.value(.at, 0.0)
        id = c.value(.id, "")
        label = c.value(.label, "")
        detail = c.value(.detail, "")
        peakBytes = c.optionalInt(.peakBytes)
        seconds = c.optional(.seconds)
        estPeakBytes = c.optionalInt(.estPeakBytes)
        estSeconds = c.optional(.estSeconds)
        identical = c.value(.identical, true)
    }
}

struct MeasuredFor: Decodable, Equatable, Sendable {
    var model: String = ""
    var size: String = ""

    enum CodingKeys: String, CodingKey { case model, size }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        model = c.value(.model, "")
        size = c.value(.size, "")
    }
}

/// GET /v1/settings.
struct EngineSettings: Decodable, Equatable, Sendable {
    var mode: String = "fast"
    var efficiency: Double = 0.5
    var rung: String = "resident"
    var ladder: [ModeRung] = []
    var measuredFor: MeasuredFor?

    enum CodingKeys: String, CodingKey { case mode, efficiency, rung, ladder, measuredFor = "measured_for" }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        mode = c.value(.mode, "fast")
        efficiency = c.value(.efficiency, 0.5)
        rung = c.value(.rung, "resident")
        ladder = c.value(.ladder, [])
        measuredFor = c.optional(.measuredFor)
    }

    var isLowMemory: Bool { mode == "low_memory" }

    /// The highest rung whose `at` is <= `efficiency` (docs/MODES.md).
    func index(for efficiency: Double) -> Int {
        var best = 0
        for (i, r) in ladder.enumerated() where r.at <= efficiency + 1e-9 { best = i }
        return best
    }

    func rung(id: String) -> ModeRung? { ladder.first { $0.id == id } }

    func label(forRung id: String) -> String { rung(id: id)?.label ?? id }

    /// The rung a mode runs at before fitting: Fast plans the first rung.
    func rungID(mode: String, efficiency: Double) -> String {
        guard !ladder.isEmpty else { return rung }
        return mode == "low_memory" ? ladder[index(for: efficiency)].id : ladder[0].id
    }

    /// Peak memory of a rung: measured first, else the estimate; nil when neither exists.
    func peak(of r: ModeRung) -> (bytes: Int, estimate: Bool)? {
        if let p = r.peakBytes { return (p, false) }
        if let p = r.estPeakBytes { return (p, true) }
        return nil
    }

    /// Time relative to Fast (the first rung), like with like (measured or estimated).
    func timeRatio(of r: ModeRung) -> (ratio: Double, estimate: Bool)? {
        guard let base = ladder.first else { return nil }
        if let a = r.seconds, let b = base.seconds, b > 0 { return (a / b, false) }
        if let a = r.estSeconds, let b = base.estSeconds, b > 0 { return (a / b, true) }
        return nil
    }
}

/// The mode a job ran (or will run) with: from `result` once finished, else the params the
/// engine copied in at submit.
struct JobModeInfo: Equatable, Sendable {
    var mode: String
    var rung: String?
    var efficiency: Double?
    var steppedDownFrom: String?
}

/// One stage of a finished job, when the result carries `stages`.
struct JobStage: Identifiable, Sendable {
    let id: Int
    let name: String
    let seconds: Double?
    let peakBytes: Int?
}

extension Job {
    var modeInfo: JobModeInfo? {
        guard let mode = result["mode"]?.stringValue ?? request.params["mode"]?.stringValue else { return nil }
        return JobModeInfo(mode: mode,
                           rung: result["rung"]?.stringValue,
                           efficiency: request.params["efficiency"]?.doubleValue,
                           steppedDownFrom: result["stepped_down_from"]?.stringValue)
    }

    /// `result.stages` as a list (`[{name, seconds, peak_bytes}]`) or a map (`{name: {...}}`).
    var stages: [JobStage] {
        guard let raw = result["stages"] else { return [] }
        var out: [JobStage] = []
        switch raw {
        case .array(let items):
            for (i, item) in items.enumerated() {
                let name = item["name"]?.stringValue ?? item["stage"]?.stringValue ?? item["id"]?.stringValue ?? "Stage \(i + 1)"
                out.append(JobStage(id: i, name: name, seconds: item["seconds"]?.doubleValue,
                                    peakBytes: item["peak_bytes"]?.doubleValue.map { Int($0) }))
            }
        case .object(let map):
            for (i, key) in map.keys.sorted().enumerated() {
                let item = map[key] ?? .null
                let secs: Double?
                if case .number(let d) = item { secs = d } else { secs = item["seconds"]?.doubleValue }
                out.append(JobStage(id: i, name: key, seconds: secs, peakBytes: item["peak_bytes"]?.doubleValue.map { Int($0) }))
            }
        default:
            break
        }
        return out
    }
}
