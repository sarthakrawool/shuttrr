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

final class VoiceModelTests: XCTestCase {
    func testAudioOutputsListsEveryTake() throws {
        let j = try JSONDecoder().decode(Job.self, from: Data(#"{"id":"a","state":"done","result":{"output":"/o/a.wav","outputs":["/o/a.wav","/o/a.s43.wav"],"kind":"audio"}}"#.utf8))
        XCTAssertEqual(j.audioOutputs, ["/o/a.wav", "/o/a.s43.wav"])
        let v = try JSONDecoder().decode(Job.self, from: Data(#"{"id":"b","state":"done","result":{"output":"/o/b.mp4"}}"#.utf8))
        XCTAssertTrue(v.audioOutputs.isEmpty)
    }

    func testCatalogLicenceDecodes() throws {
        let m = try JSONDecoder().decode(CatalogModel.self, from: Data(#"{"id":"fish-drama3-5bit","name":"Fish Audio Drama 3","kind":"audio","licence":{"name":"Fish Audio Community License v1.0","url":"https://x","points":["a","b"]},"licence_accepted":false}"#.utf8))
        XCTAssertTrue(m.needsLicence)
        XCTAssertEqual(m.licence?.points.count, 2)
        let w = try JSONDecoder().decode(CatalogModel.self, from: Data(#"{"id":"w","name":"Wan"}"#.utf8))
        XCTAssertFalse(w.needsLicence)
        XCTAssertTrue(w.available)       // older engines send no "available": downloadable as before
        let up = try JSONDecoder().decode(CatalogModel.self, from: Data(#"{"id":"f","available":false,"availability_note":"Coming soon","beta":true}"#.utf8))
        XCTAssertFalse(up.available)
        XCTAssertTrue(up.beta)
    }
}
