import XCTest
@testable import Shutter

final class PlainErrorTests: XCTestCase {
    func testOldEngineLoadErrorBecomesASentence() {
        // The text a 0.2.3 engine sends for Load on Wan.
        let raw = "RuntimeError: wan2.1-t2v-1.3b-8bit cannot be loaded ahead of a request; it loads when a request needs it"
        let f = LoadFailure.other(modelID: "wan2.1-t2v-1.3b-8bit", name: "Wan 2.1 Text to Video 1.3B", reason: raw)
        XCTAssertFalse(f.body.contains("RuntimeError"))
        XCTAssertFalse(f.body.contains("wan2.1-t2v-1.3b-8bit"))
        XCTAssertEqual(f.body, "Wan 2.1 Text to Video 1.3B cannot be loaded ahead of a request; it loads when a request needs it.")
    }

    func testTracebackShowsTheFallback() {
        let raw = "Traceback (most recent call last):\n  File \"x.py\", line 1, in <module>\nValueError: boom"
        XCTAssertEqual(PlainError.sentence(raw, fallback: "F."), "F.")
    }

    func testDottedExceptionNameIsDropped() {
        XCTAssertEqual(PlainError.sentence("shutter.types.MemoryRefused: not enough memory", fallback: "F."),
                       "Not enough memory.")
    }

    func testCodeLikeTextShowsTheFallback() {
        XCTAssertEqual(PlainError.sentence("KeyError: <object at 0x10a2b3c4>", fallback: "F."), "F.")
        XCTAssertEqual(PlainError.sentence("", fallback: "F."), "F.")
    }

    func testPlainSentencesPassThrough() {
        let s = "Wan 2.1 T2V 1.3B needs 6.2 GB and macOS can give 4.0 GB right now."
        XCTAssertEqual(PlainError.sentence(s, fallback: "F."), s)
        XCTAssertEqual(PlainError.ifTechnical("loading model", fallback: "F."), "loading model")
        XCTAssertFalse(PlainError.isTechnical("loading model"))
        XCTAssertTrue(PlainError.isTechnical("RuntimeError: x"))
    }
}

final class EngineVersionCheckTests: XCTestCase {
    func testMismatch() {
        XCTAssertTrue(EngineVersionCheck.mismatch(running: "0.2.3", bundled: "0.2.7"))
        XCTAssertFalse(EngineVersionCheck.mismatch(running: "0.2.7", bundled: "0.2.7"))
        XCTAssertFalse(EngineVersionCheck.mismatch(running: "v0.2.7\n", bundled: "0.2.7"))
        XCTAssertFalse(EngineVersionCheck.mismatch(running: "", bundled: "0.2.7"))   // unknown: leave alone
        XCTAssertFalse(EngineVersionCheck.mismatch(running: "0.2.3", bundled: ""))
    }

    func testIsShuttrrEngine() {
        XCTAssertTrue(EngineStopper.isShuttrrEngine("/opt/homebrew/Python /Users/a/.shutter/venv/bin/shutter serve"))
        XCTAssertFalse(EngineStopper.isShuttrrEngine("python -m http.server 8484"))
    }
}

final class UpdateIntervalTests: XCTestCase {
    func testChecksMoreThanOnceADay() {
        XCTAssertLessThanOrEqual(UpdateChecker.checkInterval, 4 * 3600)
        XCTAssertGreaterThanOrEqual(UpdateChecker.checkInterval, 3600)   // Sparkle's minimum
    }
}
