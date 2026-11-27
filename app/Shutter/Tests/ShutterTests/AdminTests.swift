import XCTest
@testable import Shutter

/// In-memory token store: the real Keychain is never touched in tests.
final class FakeAdminTokens: AdminTokenStore, @unchecked Sendable {
    private let lock = NSLock()
    private var value: String?
    init(_ v: String?) { value = v }
    func read() -> String? { lock.lock(); defer { lock.unlock() }; return value }
    func save(_ token: String) throws { lock.lock(); value = token; lock.unlock() }
    func delete() throws { lock.lock(); value = nil; lock.unlock() }
}

/// Counts calls and answers from a list of canned results (the last one repeats).
final class FakeFetch: @unchecked Sendable {
    private let lock = NSLock()
    private var results: [AdminFetchResult]
    private(set) var calls: [(token: String, days: Int)] = []
    init(_ r: AdminFetchResult...) { results = r }
    var fetcher: AdminFetcher {
        { token, days in
            self.lock.lock(); defer { self.lock.unlock() }
            self.calls.append((token, days))
            return self.results.count > 1 ? self.results.removeFirst() : self.results[0]
        }
    }
}

final class AdminTests: XCTestCase {
    static let full = """
    {"days":7,"since":"2026-11-20","usage_reports_this_week":2,
     "models":[{"model":"z-image-turbo-8bit","requests":6,"finished":4,"success_pct":66.7,"reports":2}],
     "macs":[{"chip":"M5","memory_gb":16,"reports":2}],
     "versions":[{"app_version":"0.3.3","reports":2}],
     "issues":[{"kind":"app_crash","error_type":"EXC","message":"m","times":2,"first_day":"2026-11-27","last_day":"2026-11-27","versions":"0.3.3","models":null,"latest_trace":"t"}]}
    """

    func testDecodeFullSummary() throws {
        let s = try AdminSummary.decode(Data(Self.full.utf8))
        XCTAssertEqual(s.days, 7)
        XCTAssertEqual(s.since, "2026-11-20")
        XCTAssertEqual(s.usageReportsThisWeek, 2)
        XCTAssertEqual(s.models?.first?.model, "z-image-turbo-8bit")
        XCTAssertEqual(s.models?.first?.requests, 6)
        XCTAssertEqual(s.models?.first?.finished, 4)
        XCTAssertEqual(s.models?.first?.successPct, 66.7)
        XCTAssertEqual(s.macs?.first?.chip, "M5")
        XCTAssertEqual(s.macs?.first?.memoryGb, 16)
        XCTAssertEqual(s.versions?.first?.appVersion, "0.3.3")
        let i = try XCTUnwrap(s.issues?.first)
        XCTAssertEqual(i.kind, "app_crash")
        XCTAssertEqual(i.errorType, "EXC")
        XCTAssertEqual(i.times, 2)
        XCTAssertEqual(i.firstDay, "2026-11-27")
        XCTAssertEqual(i.lastDay, "2026-11-27")
        XCTAssertEqual(i.versions, "0.3.3")
        XCTAssertNil(i.models)
        XCTAssertEqual(i.latestTrace, "t")
    }

    func testDecodeWithNulls() throws {
        let json = """
        {"days":null,"since":null,"usage_reports_this_week":null,"models":null,"macs":[{"chip":null,"memory_gb":null,"reports":null}],
         "versions":[],"issues":[{"kind":null,"error_type":null,"message":null,"times":null,"first_day":null,"last_day":null,
         "versions":null,"models":null,"latest_trace":null}]}
        """
        let s = try AdminSummary.decode(Data(json.utf8))
        XCTAssertNil(s.days)
        XCTAssertNil(s.usageReportsThisWeek)
        XCTAssertNil(s.models)
        XCTAssertEqual(s.macs?.count, 1)
        XCTAssertNil(s.macs?.first?.chip)
        XCTAssertEqual(s.versions, [])
        XCTAssertNil(s.issues?.first?.latestTrace)
        // missing keys entirely
        XCTAssertNoThrow(try AdminSummary.decode(Data("{}".utf8)))
    }

    func testClassifyStatus() {
        if case .ok = AdminAPI.classify(status: 200, data: Data(Self.full.utf8)) {} else { XCTFail("200 should be ok") }
        XCTAssertEqual(AdminAPI.classify(status: 404, data: Data("not found".utf8)), .rejected)
        if case .failed = AdminAPI.classify(status: 200, data: Data("nope".utf8)) {} else { XCTFail("bad json") }
        if case .failed = AdminAPI.classify(status: 500, data: Data()) {} else { XCTFail("500 should fail") }
    }

    @MainActor
    func testNoTokenStaysHiddenAndMakesNoCall() async {
        let f = FakeFetch(.rejected)
        let s = AdminStore(tokens: FakeAdminTokens(nil), fetch: f.fetcher)
        await s.refresh()
        XCTAssertFalse(s.verified)
        XCTAssertTrue(f.calls.isEmpty)
    }

    @MainActor
    func testRejectedTokenStaysHidden() async {
        let s = AdminStore(tokens: FakeAdminTokens("fake-token"), fetch: FakeFetch(.rejected).fetcher)
        await s.refresh()
        XCTAssertFalse(s.verified)
        XCTAssertNil(s.summary)
    }

    @MainActor
    func testAcceptedTokenShowsTab() async throws {
        let summary = try AdminSummary.decode(Data(Self.full.utf8))
        let f = FakeFetch(.ok(summary))
        let s = AdminStore(tokens: FakeAdminTokens("fake-token"), fetch: f.fetcher)
        await s.refresh()
        XCTAssertTrue(s.verified)
        XCTAssertEqual(s.summary, summary)
        XCTAssertEqual(f.calls.first?.days, 7)
        XCTAssertEqual(f.calls.first?.token, "fake-token")
    }

    @MainActor
    func testNetworkErrorBeforeSuccessStaysHidden() async {
        let s = AdminStore(tokens: FakeAdminTokens("fake-token"), fetch: FakeFetch(.failed("offline")).fetcher)
        await s.refresh()
        XCTAssertFalse(s.verified)
        XCTAssertEqual(s.lastError, "offline")
    }

    @MainActor
    func testNetworkErrorAfterSuccessKeepsTab_thenRejectHides() async throws {
        let summary = try AdminSummary.decode(Data(Self.full.utf8))
        let s = AdminStore(tokens: FakeAdminTokens("fake-token"),
                           fetch: FakeFetch(.ok(summary), .failed("offline"), .rejected).fetcher)
        await s.refresh()
        XCTAssertTrue(s.verified)
        await s.refresh()
        XCTAssertTrue(s.verified)
        XCTAssertEqual(s.summary, summary)
        await s.refresh()
        XCTAssertFalse(s.verified)
        XCTAssertNil(s.summary)
    }

    @MainActor
    func testSaveAndRemoveToken() async throws {
        let summary = try AdminSummary.decode(Data(Self.full.utf8))
        let tokens = FakeAdminTokens(nil)
        let s = AdminStore(tokens: tokens, fetch: FakeFetch(.rejected, .ok(summary)).fetcher)
        let first = await s.saveToken("wrong")
        XCTAssertEqual(first, .notAccepted)
        XCTAssertFalse(s.verified)
        let second = await s.saveToken("  right \n")
        XCTAssertEqual(second, .accepted)
        XCTAssertTrue(s.verified)
        XCTAssertEqual(tokens.read(), "right")
        s.removeToken()
        XCTAssertFalse(s.verified)
        XCTAssertNil(tokens.read())
    }
}
