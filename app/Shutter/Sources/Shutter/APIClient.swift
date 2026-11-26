import Foundation

enum APIError: LocalizedError {
    case badURL(String)
    case badResponse
    case http(Int, String)

    var errorDescription: String? {
        switch self {
        case .badURL(let s): return "Invalid engine URL: \(s)"
        case .badResponse: return "The engine sent an unreadable response."
        case .http(let code, let msg): return msg.isEmpty ? "Engine returned HTTP \(code)." : "HTTP \(code): \(msg)"
        }
    }
}

/// Thin async client for the engine's HTTP API. Value type; build a fresh one with
/// `APIClient.current()` so a changed URL in Settings takes effect on the next call.
struct APIClient: Sendable {
    static let baseURLKey = "engineURL"
    static let defaultBaseURL = "http://127.0.0.1:8484"

    let base: String

    static func current() -> APIClient {
        let raw = UserDefaults.standard.string(forKey: baseURLKey) ?? defaultBaseURL
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.isEmpty { s = defaultBaseURL }
        while s.hasSuffix("/") { s.removeLast() }
        return APIClient(base: s)
    }

    /// Short timeouts: the engine is local, a slow answer means it is down or busy.
    static let session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 5
        config.timeoutIntervalForResource = 15
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        return URLSession(configuration: config)
    }()

    /// For calls that start a helper process on the engine side (model downloads).
    static let slowSession: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 40
        config.timeoutIntervalForResource = 60
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        return URLSession(configuration: config)
    }()

    /// Long-lived session for the SSE stream.
    static let streamSession: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 3600
        config.timeoutIntervalForResource = 7 * 24 * 3600
        return URLSession(configuration: config)
    }()

    func url(_ path: String) throws -> URL {
        guard let u = URL(string: base + path) else { throw APIError.badURL(base + path) }
        return u
    }

    // MARK: Endpoints

    func system() async throws -> SystemInfo {
        try await send("GET", "/v1/system")
    }

    func models() async throws -> [ModelInfo] {
        let list: ModelList = try await send("GET", "/v1/models")
        return list.models
    }

    /// Apps that sent requests in the last few minutes.
    func clients() async throws -> [ClientInfo] {
        let list: ClientList = try await send("GET", "/v1/clients")
        return list.clients
    }

    func jobs(limit: Int = 50) async throws -> [Job] {
        let list: JobList = try await send("GET", "/v1/jobs?limit=\(limit)")
        return list.jobs
    }

    func job(id: String) async throws -> Job {
        try await send("GET", "/v1/jobs/\(escape(id))")
    }

    func submit(_ request: NewJobRequest) async throws -> Job {
        let body = try JSONEncoder().encode(request)
        return try await send("POST", "/v1/jobs", body: body)
    }

    func cancel(id: String) async throws -> Job {
        try await send("DELETE", "/v1/jobs/\(escape(id))")
    }

    /// Unload one pool entry, or all unpinned entries when `key` is nil.
    /// POST /v1/models/<id>/load. Returns nil when the load was queued, or the guard's refusal (409) with numbers.
    /// `wait` queues it to wait for memory instead of refusing.
    func loadModel(id: String, wait: Bool = false) async throws -> LoadRefusal? {
        var req = URLRequest(url: try url("/v1/models/\(escape(id))/load"))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = Data((wait ? "{\"wait\": true}" : "{}").utf8)
        let (data, response) = try await Self.session.data(for: req)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        if http.statusCode == 409, let body = try? JSONDecoder().decode(LoadRefusedBody.self, from: data) {
            return body.refused
        }
        guard (200..<300).contains(http.statusCode) else {
            throw APIError.http(http.statusCode, Self.errorMessage(data))
        }
        return nil
    }

    private struct LoadRefusedBody: Decodable { let refused: LoadRefusal }

    /// POST /v1/models/<id>/unload: release the model (refused with a sentence while a request for it runs).
    func unloadModel(id: String) async throws {
        let _: JSONValue = try await send("POST", "/v1/models/\(escape(id))/unload", body: Data("{}".utf8))
    }

    func releasePool(key: String?) async throws -> [String] {
        let body = try JSONEncoder().encode(ReleaseBody(key: key))
        let r: ReleaseResponse = try await send("POST", "/v1/pool/release", body: body)
        return r.released
    }

    /// Not in the documented API yet: POST /v1/queue/pause and /v1/queue/resume.
    /// The UI offers it only when /v1/system reports `queue.paused`.
    func setQueuePaused(_ paused: Bool) async throws {
        let _: JSONValue = try await send("POST", paused ? "/v1/queue/pause" : "/v1/queue/resume", body: Data("{}".utf8))
    }

    /// GET /v1/settings: the default mode, efficiency and the low-memory ladder.
    func settings() async throws -> EngineSettings {
        try await send("GET", "/v1/settings")
    }

    /// POST /v1/settings; nil fields are left out (and left unchanged by the engine).
    func updateSettings(mode: String?, efficiency: Double?) async throws -> EngineSettings {
        let body = try JSONEncoder().encode(SettingsBody(mode: mode, efficiency: efficiency))
        return try await send("POST", "/v1/settings", body: body)
    }

    /// Models the engine can download by itself, with any download in progress.
    func catalog() async throws -> CatalogList {
        try await send("GET", "/v1/catalog", slow: true)
    }

    func startDownload(id: String, acceptLicence: Bool = false) async throws -> CatalogList {
        let q = acceptLicence ? "?accept_licence=1" : ""
        return try await send("POST", "/v1/catalog/\(escape(id))/download\(q)", body: Data("{}".utf8), slow: true)
    }

    func stopDownload(id: String) async throws -> CatalogList {
        try await send("DELETE", "/v1/catalog/\(escape(id))/download", slow: true)
    }

    /// GET /v1/models/<id>/request-options. Nil when the engine has no such endpoint (older engines answer 404).
    func requestOptions(model: String) async throws -> RequestOptions? {
        try await orNil404 { try await send("GET", "/v1/models/\(escape(model))/request-options") }
    }

    /// POST /v1/models/<id>/estimate: time, fit, warnings and issues for these values. Nil on an older engine.
    func estimate(model: String, values: RequestValues) async throws -> RequestEstimate? {
        let body = try JSONEncoder().encode(values)
        return try await orNil404 { try await send("POST", "/v1/models/\(escape(model))/estimate", body: body) }
    }

    /// PUT /v1/models/<id>/defaults: save the person's own starting values for this model.
    func saveDefaults(model: String, values: RequestValues) async throws -> RequestOptions {
        let body = try JSONEncoder().encode(values)
        return try await send("PUT", "/v1/models/\(escape(model))/defaults", body: body)
    }

    /// DELETE /v1/models/<id>/defaults: back to this Mac's starting values.
    func resetDefaults(model: String) async throws -> RequestOptions {
        try await send("DELETE", "/v1/models/\(escape(model))/defaults")
    }

    func eventsURL() throws -> URL { try url("/v1/events") }

    // MARK: Plumbing

    private struct ReleaseBody: Encodable {
        let key: String?
        enum CodingKeys: String, CodingKey { case key }
        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(key, forKey: .key) // explicit null when nil
        }
    }

    private struct SettingsBody: Encodable {
        let mode: String?
        let efficiency: Double?
    }

    private func orNil404<T>(_ call: () async throws -> T) async throws -> T? {
        do {
            return try await call()
        } catch APIError.http(404, _) {
            return nil
        }
    }

    private func escape(_ s: String) -> String {
        s.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? s
    }

    private func send<T: Decodable>(_ method: String, _ path: String, body: Data? = nil,
                                    slow: Bool = false) async throws -> T {
        var req = URLRequest(url: try url(path))
        req.httpMethod = method
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        if let body {
            req.httpBody = body
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        let (data, response) = try await (slow ? Self.slowSession : Self.session).data(for: req)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard (200..<300).contains(http.statusCode) else {
            throw APIError.http(http.statusCode, Self.errorMessage(data))
        }
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw APIError.badResponse
        }
    }

    /// Pulls a readable message out of {"detail": ...} / {"error": ...} bodies.
    private static func errorMessage(_ data: Data) -> String {
        if let v = try? JSONDecoder().decode(JSONValue.self, from: data) {
            for k in ["detail", "error", "message", "reason"] {
                if let s = v[k]?.stringValue { return s }
            }
        }
        let text = String(decoding: data.prefix(300), as: UTF8.self)
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
