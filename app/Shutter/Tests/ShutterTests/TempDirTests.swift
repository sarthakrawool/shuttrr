import XCTest
@testable import Shutter

final class TempDirTests: XCTestCase {
    func testMissingInstallerSandboxFolderIsNotUsable() {
        XCTAssertFalse(TempDir.usable("/private/tmp/PKInstallSandbox.doesnotexist/tmp/"))
        XCTAssertTrue(TempDir.usable(NSTemporaryDirectory()))
    }

    func testRepairReplacesAMissingTMPDIR() {
        let saved = ProcessInfo.processInfo.environment["TMPDIR"]
        defer { if let s = saved { setenv("TMPDIR", s, 1) } }
        setenv("TMPDIR", "/private/tmp/PKInstallSandbox.doesnotexist/tmp/", 1)
        let fixed = TempDir.repair()
        XCTAssertTrue(TempDir.usable(fixed))
        XCTAssertEqual(ProcessInfo.processInfo.environment["TMPDIR"], fixed)
        XCTAssertEqual(TempDir.replaced, "/private/tmp/PKInstallSandbox.doesnotexist/tmp/")
    }
}
