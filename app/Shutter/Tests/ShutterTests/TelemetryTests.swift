import XCTest
@testable import Shutter

final class TelemetryTests: XCTestCase {
    // A trimmed macOS crash report: header line + JSON body, with a user's paths and file names in it.
    static let ips = """
    {"app_name":"Shutter","timestamp":"2026-11-27 03:29:04.00 +0530","app_version":"0.3.2","slice_uuid":"11111111-2222-3333-4444-555555555555","bug_type":"309","os_version":"macOS 27.2.1 (26B131)","name":"Shutter"}
    {
      "procName" : "Shutter",
      "procPath" : "/Users/alice/Applications/Shuttrr.app/Contents/MacOS/Shutter",
      "osVersion" : {"train" : "macOS 27.2.1", "build" : "26B131", "releaseType" : "User"},
      "bundleInfo" : {"CFBundleShortVersionString" : "0.3.2", "CFBundleIdentifier" : "com.shutter.app"},
      "exception" : {"type" : "EXC_BREAKPOINT", "signal" : "SIGTRAP", "subtype" : "crashed at /Users/alice/Desktop/secret-plan.txt"},
      "faultingThread" : 1,
      "threads" : [
        {"frames" : [{"imageIndex" : 0, "symbol" : "main", "imageOffset" : 100}]},
        {"triggered" : true, "frames" : [
          {"imageIndex" : 1, "symbol" : "_assertionFailure(_:_:file:line:flags:)", "imageOffset" : 5},
          {"imageIndex" : 0, "symbol" : "EngineStore.pollOnce() in /Users/alice/src/Shutter/EngineStore.swift", "imageOffset" : 200},
          {"imageIndex" : 0, "imageOffset" : 4096},
          {"imageIndex" : 0, "symbol" : "closure #1 in AppDelegate.load(alice_photo.png)", "imageOffset" : 300},
          {"imageIndex" : 2, "symbol" : "AGGraphGetValue", "imageOffset" : 7}
        ]}
      ],
      "usedImages" : [
        {"name" : "Shutter", "uuid" : "24B6D79A-606A-3BD3-8865-5861AB6337CE", "path" : "/Users/alice/Applications/Shuttrr.app/Contents/MacOS/Shutter"},
        {"name" : "libswiftCore.dylib", "path" : "/usr/lib/swift/libswiftCore.dylib"},
        {"name" : "AttributeGraph", "path" : "/System/Library/PrivateFrameworks/AttributeGraph.framework/AttributeGraph"}
      ]
    }
    """

    func testCrashReportIsScrubbed() throws {
        let r = try XCTUnwrap(CrashReportParser.parse(Self.ips, imageName: "Shutter",
                                                      userName: "alice", home: "/Users/alice"))
        let all = ([r.exception, r.signal, r.appVersion, r.macos, r.buildUUID ?? ""] + r.frames).joined(separator: "\n")
        // the Shutter image's UUID is carried, as a bare UUID (no path, no user name)
        XCTAssertEqual(r.buildUUID, "24b6d79a-606a-3bd3-8865-5861ab6337ce")
        let encoded = String(decoding: try JSONEncoder().encode(r), as: UTF8.self)
        XCTAssertTrue(encoded.contains("\"build_uuid\":\"24b6d79a-606a-3bd3-8865-5861ab6337ce\""), encoded)
        XCTAssertFalse(encoded.contains("alice"), encoded)
        XCTAssertFalse(encoded.contains("\\/") || encoded.contains("Users"), encoded)
        XCTAssertFalse(all.lowercased().contains("alice"), all)
        XCTAssertFalse(all.contains("/"), all)
        XCTAssertFalse(all.contains("secret-plan"), all)
        XCTAssertFalse(all.contains(".swift"), all)
        XCTAssertFalse(all.contains(".png"), all)

        XCTAssertTrue(r.exception.hasPrefix("EXC_BREAKPOINT"))
        XCTAssertEqual(r.signal, "SIGTRAP")
        XCTAssertEqual(r.appVersion, "0.3.2")
        XCTAssertEqual(r.macos, "macOS 27.2.1 (26B131)")
        // only frames inside the Shutter image, from the crashed thread, in order
        XCTAssertEqual(r.frames.count, 3)
        XCTAssertTrue(r.frames[0].hasPrefix("EngineStore.pollOnce() in"), r.frames[0])
        XCTAssertEqual(r.frames[1], "Shutter +0x1000")
        XCTAssertTrue(r.frames[2].hasPrefix("closure #1 in AppDelegate.load("), r.frames[2])
        XCTAssertFalse(r.frames.contains { $0.contains("AGGraphGetValue") || $0.contains("_assertionFailure") })
    }

    func testCrashReportEncodesWithEngineKeys() throws {
        let r = AppCrashReport(exception: "EXC_BAD_ACCESS", signal: "SIGSEGV", appVersion: "0.3.2",
                               macos: "macOS 27.2.1", frames: ["a()"])
        let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(r)) as? [String: Any])
        XCTAssertEqual(Set(obj.keys), ["exception", "signal", "app_version", "macos", "frames"])   // no UUID: left out
    }

    func testBuildUUIDFallsBackToHeaderAndRejectsNonUUIDs() throws {
        let noImageUUID = Self.ips.replacingOccurrences(of: #""uuid" : "24B6D79A-606A-3BD3-8865-5861AB6337CE", "#, with: "")
        let r = try XCTUnwrap(CrashReportParser.parse(noImageUUID, userName: "alice", home: "/Users/alice"))
        XCTAssertEqual(r.buildUUID, "11111111-2222-3333-4444-555555555555")
        XCTAssertNil(AppCrashReport.cleanUUID("/Users/alice/24b6d79a-606a-3bd3-8865-5861ab6337ce"))
        XCTAssertNil(AppCrashReport.cleanUUID("not-a-uuid"))
        XCTAssertNil(AppCrashReport.cleanUUID(nil))
    }

    func testGarbageIsNotAReport() {
        XCTAssertNil(CrashReportParser.parse("not json"))
        XCTAssertNil(CrashReportParser.parse("{}\nnot json"))
    }

    func testScrubberKeepsSymbolsAndPlainText() {
        XCTAssertEqual(Scrubber.clean("Trace/BPT trap: 5", userName: "bob", home: "/Users/bob"), "Trace/BPT trap: 5")
        XCTAssertEqual(Scrubber.clean("specialized EnvironmentObject.error()", userName: "bob", home: "/Users/bob"),
                       "specialized EnvironmentObject.error()")
        XCTAssertEqual(Scrubber.clean("opened ~/Desktop/a.mp4 for bob", userName: "bob", home: "/Users/bob"),
                       "opened for")
    }

    func testDecodeTelemetry() throws {
        let json = """
        {"enabled": true, "asked": true, "endpoint": "https://example.test/t",
         "what": ["Which model you use most."], "never": ["Your prompts."],
         "last_payload": {"top_model": "wan2.1-t2v-1.3b-8bit", "count": 12, "app": {"version": "0.3.2"}},
         "last_sent_at": 1764200000.5, "queued_issues": 2}
        """
        let t = try TelemetryInfo.decode(Data(json.utf8))
        XCTAssertTrue(t.enabled)
        XCTAssertTrue(t.asked)
        XCTAssertEqual(t.endpoint, "https://example.test/t")
        XCTAssertEqual(t.what, ["Which model you use most."])
        XCTAssertEqual(t.never, ["Your prompts."])
        XCTAssertEqual(t.lastSentAt, 1764200000.5)
        XCTAssertEqual(t.queuedIssues, 2)
        let text = try XCTUnwrap(t.lastPayloadText)
        XCTAssertTrue(text.contains("\"count\" : 12"), text)          // whole numbers stay whole
        XCTAssertTrue(text.contains("\"top_model\" : \"wan2.1-t2v-1.3b-8bit\""), text)
        XCTAssertTrue(text.contains("\n"))                             // pretty-printed
    }

    func testDecodeTelemetryWithNothingSent() throws {
        let json = #"{"enabled": false, "asked": false, "endpoint": "", "what": [], "never": [], "last_payload": null, "last_sent_at": null, "queued_issues": 0}"#
        let t = try TelemetryInfo.decode(Data(json.utf8))
        XCTAssertFalse(t.enabled)
        XCTAssertFalse(t.asked)
        XCTAssertNil(t.lastPayloadText)
        XCTAssertNil(t.lastSentAt)
        XCTAssertEqual(t.whatOrFallback, TelemetryInfo.fallbackWhat)   // empty lists fall back to the built-in text
        XCTAssertEqual(t.neverOrFallback, TelemetryInfo.fallbackNever)
    }

    func testUnsupportedEngineIsRecognised() {
        XCTAssertTrue(EngineStore.isUnsupported(APIError.http(404, "")))
        XCTAssertTrue(EngineStore.isUnsupported(APIError.http(501, "")))
        XCTAssertFalse(EngineStore.isUnsupported(APIError.http(500, "")))
        XCTAssertFalse(EngineStore.isUnsupported(URLError(.cannotConnectToHost)))
    }

    func testNewReportsOnlyTheAppsOwnAndNewer() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("shutter-ips-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let old = dir.appendingPathComponent("Shutter-2026-11-01-000000.ips")
        let new = dir.appendingPathComponent("Shutter-2026-11-27-000000.ips")
        let other = dir.appendingPathComponent("ShutterHelper-2026-11-27-000000.ips")
        let notIps = dir.appendingPathComponent("Shutter-2026-11-27-000000.diag")
        for u in [old, new, other, notIps] { try Data("x".utf8).write(to: u) }
        try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: 1000)], ofItemAtPath: old.path)
        try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: 3000)], ofItemAtPath: new.path)
        let found = AppCrashReporter.newReports(in: dir, executable: "Shutter", after: 2000)
        XCTAssertEqual(found.map { $0.url.lastPathComponent }, [new.lastPathComponent])
    }
}
