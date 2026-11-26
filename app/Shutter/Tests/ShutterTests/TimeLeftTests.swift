import XCTest
@testable import Shutter

final class TimeLeftTests: XCTestCase {
    private func job(_ json: String) throws -> Job {
        try JSONDecoder().decode(Job.self, from: Data(json.utf8))
    }

    func testRunningWithoutMeasurementSaysEstimating() throws {
        let j = try job(#"{"id":"a","state":"running","stage":"encoding the prompt","eta_s":null}"#)
        XCTAssertEqual(j.timeLeftText, "Estimating…")
        XCTAssertNil(j.remainingS)      // never extrapolated from progress
    }

    func testRunningWithEta() throws {
        let j = try job(#"{"id":"a","state":"running","progress":0.4,"eta_s":185,"eta_decode_known":true}"#)
        XCTAssertEqual(j.timeLeftText, "About 3 min left")
        let k = try job(#"{"id":"a","state":"running","eta_s":40,"eta_decode_known":false}"#)
        XCTAssertEqual(k.timeLeftText, "About 40 s left + decode")
    }

    func testQueuedUsesExpectedTime() throws {
        let j = try job(#"{"id":"a","state":"queued","expected_s":130}"#)
        XCTAssertEqual(j.timeLeftText, "About 2 min once it starts")
        let k = try job(#"{"id":"a","state":"queued"}"#)
        XCTAssertNil(k.timeLeftText)
    }

    func testCancellingHidesTimeAndShowsCancelling() throws {
        let j = try job(#"{"id":"a","state":"running","eta_s":90,"cancel_requested":true}"#)
        XCTAssertNil(j.timeLeftText)
        XCTAssertEqual(j.detailLine, "Cancelling…")
        let old = try job(#"{"id":"a","state":"running","message":"cancelling"}"#)   // engines before 0.2.11
        XCTAssertTrue(old.cancelRequested)
    }

    func testCancelledIsNotAFailure() throws {
        let j = try job(#"{"id":"a","state":"cancelled","message":"cancelled while running"}"#)
        XCTAssertNotEqual(j.state, .failed)
    }
}
