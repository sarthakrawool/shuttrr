import Foundation
import Security

// The owner's private Admin page: the aggregated Help The Dev Improve reports from the telemetry Worker.
// Shuttrr is open source, so hiding the page is not the security: the Worker's read API needs a secret admin
// token. The app shows the page only when a token is in this Mac's Keychain AND the Worker accepts it.
// The token's value is never printed, logged or shown.

// MARK: - Summary (GET /v1/summary?days=7|30)

struct AdminSummary: Decodable, Equatable, Sendable {
    var days: Int?
    var since: String?
    var usageReportsThisWeek: Int?
    var models: [Model]?
    var macs: [Mac]?
    var versions: [Version]?
    var issues: [Issue]?

    struct Model: Decodable, Equatable, Sendable {
        var model: String?
        var requests: Int?
        var finished: Int?
        var successPct: Double?
        var reports: Int?
    }

    struct Mac: Decodable, Equatable, Sendable {
        var chip: String?
        var memoryGb: Double?
        var reports: Int?
    }

    struct Version: Decodable, Equatable, Sendable {
        var appVersion: String?
        var reports: Int?
    }

    struct Issue: Decodable, Equatable, Sendable {
        var kind: String?
        var errorType: String?
        var message: String?
        var times: Int?
        var firstDay: String?
        var lastDay: String?
        var versions: String?
        var models: String?
        var latestTrace: String?
    }

    static func decode(_ data: Data) throws -> AdminSummary {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return try d.decode(AdminSummary.self, from: data)
    }
}

// MARK: - Token storage

/// Where the admin token lives. The real one is the Keychain; tests use a fake.
protocol AdminTokenStore: Sendable {
    func read() -> String?
    func save(_ token: String) throws
    func delete() throws
}

struct AdminKeychainError: Error, Equatable { let status: OSStatus }

/// Keychain generic password, service "shuttrr-admin-token", account "admin".
struct KeychainAdminToken: AdminTokenStore {
    static let service = "shuttrr-admin-token"
    static let account = "admin"

    private var base: [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: Self.service,
         kSecAttrAccount as String: Self.account]
    }

    func read() -> String? {
        var q = base
        q[kSecReturnData as String] = true
        q[kSecMatchLimit as String] = kSecMatchLimitOne
        var out: CFTypeRef?
        guard SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess, let data = out as? Data,
              let s = String(data: data, encoding: .utf8) else { return nil }
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        return t.isEmpty ? nil : t
    }

    func save(_ token: String) throws {
        let data = Data(token.utf8)
        let status = SecItemUpdate(base as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if status == errSecItemNotFound {
            var add = base
            add[kSecValueData as String] = data
            let s = SecItemAdd(add as CFDictionary, nil)
            if s != errSecSuccess { throw AdminKeychainError(status: s) }
        } else if status != errSecSuccess {
            throw AdminKeychainError(status: status)
        }
    }

    func delete() throws {
        let s = SecItemDelete(base as CFDictionary)
        if s != errSecSuccess && s != errSecItemNotFound { throw AdminKeychainError(status: s) }
    }
}

// MARK: - Worker read API

enum AdminFetchResult: Equatable, Sendable {
    case ok(AdminSummary)
    /// The Worker refused the token (it answers 404 "not found" for a wrong or missing token).
    case rejected
    /// No answer, or one that could not be read. Says nothing about the token.
    case failed(String)
}

typealias AdminFetcher = @Sendable (_ token: String, _ days: Int) async -> AdminFetchResult

enum AdminAPI {
    static let defaultBase = "https://telemetry.shuttrr.sarthakrawool.com"

    /// SHUTTER_TELEMETRY_ADMIN_URL overrides the Worker (local testing against `wrangler dev`).
    static var base: String {
        var b = ProcessInfo.processInfo.environment["SHUTTER_TELEMETRY_ADMIN_URL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if b.isEmpty { b = defaultBase }
        while b.hasSuffix("/") { b.removeLast() }
        return b
    }

    static func classify(status: Int, data: Data) -> AdminFetchResult {
        switch status {
        case 200:
            guard let s = try? AdminSummary.decode(data) else { return .failed("The server's answer could not be read.") }
            return .ok(s)
        case 401, 403, 404:
            return .rejected
        default:
            return .failed("The server answered \(status).")
        }
    }

    static let fetch: AdminFetcher = { token, days in
        guard let url = URL(string: "\(base)/v1/summary?days=\(days)") else { return .failed("Bad server address.") }
        var req = URLRequest(url: url, timeoutInterval: 20)
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        // ephemeral: nothing about the authorised request is cached on disk
        let session = URLSession(configuration: .ephemeral)
        defer { session.finishTasksAndInvalidate() }
        do {
            let (data, resp) = try await session.data(for: req)
            guard let http = resp as? HTTPURLResponse else { return .failed("No answer from the server.") }
            return classify(status: http.statusCode, data: data)
        } catch {
            return .failed("Could not reach the server.")
        }
    }
}

// MARK: - Store

@MainActor
final class AdminStore: ObservableObject {
    static let shared = AdminStore(tokens: KeychainAdminToken(), fetch: AdminAPI.fetch)

    /// True only after the Worker accepted the token in this session. Everything admin is hidden until then.
    @Published private(set) var verified = false
    @Published private(set) var summary: AdminSummary?
    @Published private(set) var days = 7
    @Published private(set) var loading = false
    /// Why the last refresh did not work (network), shown on the Admin page only.
    @Published private(set) var lastError: String?

    private let tokens: AdminTokenStore
    private let fetch: AdminFetcher
    private var started = false

    init(tokens: AdminTokenStore, fetch: @escaping AdminFetcher) {
        self.tokens = tokens
        self.fetch = fetch
    }

    /// At app start: checks the token once, if there is one. No token = no network call.
    func start() {
        guard !started else { return }
        started = true
        Task { await refresh() }
    }

    func setDays(_ d: Int) {
        guard d != days else { return }
        days = d
        Task { await refresh() }
    }

    /// Reads the token (off the main thread: the Keychain may ask the user once) and asks the Worker.
    @discardableResult
    func refresh() async -> AdminFetchResult? {
        let store = tokens
        let token = await Task.detached { store.read() }.value
        guard let token else {
            verified = false
            summary = nil
            lastError = nil
            return nil
        }
        loading = true
        let wanted = days
        let r = await fetch(token, wanted)
        loading = false
        apply(r, days: wanted)
        return r
    }

    private func apply(_ r: AdminFetchResult, days wanted: Int) {
        switch r {
        case .ok(let s):
            verified = true
            if wanted == days { summary = s }
            lastError = nil
        case .rejected:
            verified = false
            summary = nil
            lastError = nil
        case .failed(let why):
            // keep what we had: hidden unless it was verified earlier in this session
            lastError = why
        }
    }

    enum SaveOutcome: Equatable { case accepted, notAccepted, unreachable, couldNotSave }

    /// The hidden Admin Token… sheet: store the token, then check it.
    func saveToken(_ token: String) async -> SaveOutcome {
        let t = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return .notAccepted }
        let store = tokens
        let saved = await Task.detached { (try? store.save(t)) != nil }.value
        guard saved else { return .couldNotSave }
        switch await refresh() {
        case .ok?: return .accepted
        case .rejected?: return .notAccepted
        default: return .unreachable
        }
    }

    func removeToken() {
        try? tokens.delete()
        verified = false
        summary = nil
        lastError = nil
    }
}
