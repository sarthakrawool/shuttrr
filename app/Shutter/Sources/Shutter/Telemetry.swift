import Foundation

// "Help The Dev Improve": opt-in, anonymous usage (which model is used most) and crash reports.
// The engine owns the setting and does the sending (GET/POST /v1/telemetry); the app only asks, shows what is
// sent, and hands the engine scrubbed crash reports of the app itself. Engines without the endpoint (the Rust
// daemon, older Python engines) answer 404/501 and the whole feature stays hidden.

/// GET /v1/telemetry.
struct TelemetryInfo: Equatable, Sendable {
    var enabled = false
    var asked = false
    var endpoint = ""
    var what: [String] = []
    var never: [String] = []
    /// The last payload the engine sent, pretty-printed JSON (keys sorted). Nil when nothing was sent yet.
    var lastPayloadText: String?
    var lastSentAt: Double?
    var queuedIssues = 0

    /// Built-in wording, used when the engine gives no lists (or does not answer during set-up).
    static let fallbackWhat = [
        "Which model you use the most, as a model name and a count.",
        "When Shuttrr or its engine crashes or a request fails: the kind of error and the names of the code functions where it happened.",
        "The Shuttrr version and the macOS version.",
    ]
    static let fallbackNever = [
        "Your prompts, pictures, videos or audio.",
        "File names, folder paths or your user name.",
        "Your name, email or anything else that says who you are.",
    ]

    var whatOrFallback: [String] { what.isEmpty ? Self.fallbackWhat : what }
    var neverOrFallback: [String] { never.isEmpty ? Self.fallbackNever : never }

    /// Reads the engine's JSON. Done by hand (JSONSerialization) so `last_payload` keeps its own numbers and
    /// keys when it is shown, instead of going through a Swift type.
    static func decode(_ data: Data) throws -> TelemetryInfo {
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw APIError.badResponse
        }
        var t = TelemetryInfo()
        t.enabled = obj["enabled"] as? Bool ?? false
        t.asked = obj["asked"] as? Bool ?? false
        t.endpoint = obj["endpoint"] as? String ?? ""
        t.what = (obj["what"] as? [Any])?.compactMap { $0 as? String } ?? []
        t.never = (obj["never"] as? [Any])?.compactMap { $0 as? String } ?? []
        if let p = obj["last_payload"], !(p is NSNull) {
            if JSONSerialization.isValidJSONObject(p),
               let d = try? JSONSerialization.data(withJSONObject: p,
                                                   options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]) {
                t.lastPayloadText = String(decoding: d, as: UTF8.self)
            } else {
                t.lastPayloadText = String(describing: p)
            }
        }
        t.lastSentAt = (obj["last_sent_at"] as? NSNumber)?.doubleValue
        t.queuedIssues = (obj["queued_issues"] as? NSNumber)?.intValue ?? 0
        return t
    }
}

/// Body of POST /v1/telemetry/app-crash.
struct AppCrashReport: Codable, Equatable, Sendable {
    var exception: String
    var signal: String
    var appVersion: String
    var macos: String
    var frames: [String]
    /// The Shutter image's build UUID (slice_uuid), so frame offsets of a stripped build can be looked up later.
    /// A bare UUID only; left out of the JSON when unknown.
    var buildUUID: String? = nil

    enum CodingKeys: String, CodingKey {
        case exception, signal, appVersion = "app_version", macos, frames, buildUUID = "build_uuid"
    }

    /// Keeps only a canonical UUID ("24b6d79a-606a-3bd3-8865-5861ab6337ce"), lowercased; anything else is nil.
    static func cleanUUID(_ raw: String?) -> String? {
        guard let raw, let u = UUID(uuidString: raw.trimmingCharacters(in: .whitespaces)) else { return nil }
        return u.uuidString.lowercased()
    }
}

/// Turns a macOS crash report (.ips) into an `AppCrashReport` with nothing personal in it: no paths, no user name,
/// no file names. Only the exception text, versions and function names from the app's own code.
enum CrashReportParser {
    static let maxFrames = 12

    /// `text` is the whole .ips file: line 1 is a JSON header, the rest is the JSON body.
    /// `imageName` is the app's executable ("Shutter"): only frames inside it are kept.
    static func parse(_ text: String, imageName: String = "Shutter", userName: String = NSUserName(),
                      home: String = NSHomeDirectory()) -> AppCrashReport? {
        guard let nl = text.firstIndex(of: "\n") else { return nil }
        let headerText = String(text[..<nl])
        let bodyText = String(text[text.index(after: nl)...])
        guard let header = json(headerText), let body = json(bodyText) else { return nil }
        let scrub = { (s: String) in Scrubber.clean(s, userName: userName, home: home) }

        // exception: "EXC_BREAKPOINT" (+ subtype), signal: "SIGTRAP"
        let exc = body["exception"] as? [String: Any] ?? [:]
        var exception = exc["type"] as? String ?? ""
        if let sub = exc["subtype"] as? String, !sub.isEmpty { exception += " (\(sub))" }
        if exception.isEmpty, let ind = (body["termination"] as? [String: Any])?["indicator"] as? String {
            exception = ind
        }
        let signal = exc["signal"] as? String ?? ""

        let bundle = body["bundleInfo"] as? [String: Any] ?? [:]
        let appVersion = bundle["CFBundleShortVersionString"] as? String ?? header["app_version"] as? String ?? ""
        var macos = header["os_version"] as? String ?? ""
        if let os = body["osVersion"] as? [String: Any], let train = os["train"] as? String {
            macos = train + ((os["build"] as? String).map { " (\($0))" } ?? "")
        }

        var frames: [String] = []
        let images = body["usedImages"] as? [[String: Any]] ?? []
        let threads = body["threads"] as? [[String: Any]] ?? []
        let crashed = (body["faultingThread"] as? Int).flatMap { threads.indices.contains($0) ? threads[$0] : nil }
            ?? threads.first { ($0["triggered"] as? Bool) == true }
        for f in (crashed?["frames"] as? [[String: Any]] ?? []) {
            guard let i = f["imageIndex"] as? Int, images.indices.contains(i),
                  (images[i]["name"] as? String) == imageName else { continue }
            if let sym = f["symbol"] as? String, !sym.isEmpty {
                let s = scrub(sym)
                if !s.isEmpty { frames.append(s) }
            } else if let off = f["imageOffset"] as? Int {
                frames.append("\(imageName) +0x\(String(off, radix: 16))")   // stripped build: offset only
            }
            if frames.count == maxFrames { break }
        }

        // the app image's UUID from usedImages, else the header's slice_uuid (the main executable's)
        let imageUUID = images.first { ($0["name"] as? String) == imageName }?["uuid"] as? String
        let buildUUID = AppCrashReport.cleanUUID(imageUUID) ?? AppCrashReport.cleanUUID(header["slice_uuid"] as? String)

        return AppCrashReport(exception: scrub(exception), signal: scrub(signal),
                              appVersion: scrub(appVersion), macos: scrub(macos), frames: frames,
                              buildUUID: buildUUID)
    }

    private static func json(_ s: String) -> [String: Any]? {
        guard let d = s.data(using: .utf8) else { return nil }
        return (try? JSONSerialization.jsonObject(with: d)) as? [String: Any]
    }
}

/// Removes paths, file names, the home folder and the user name from a line of crash text.
enum Scrubber {
    static func clean(_ input: String, userName: String = NSUserName(), home: String = NSHomeDirectory()) -> String {
        var s = input
        if !home.isEmpty && home != "/" { s = s.replacingOccurrences(of: home, with: "") }
        // absolute and home-relative paths: "/Users/x/a b.swift", "~/Desktop/x"
        s = s.replacingOccurrences(of: #"(?<![\w.])(~/|/)[^\s:'"(),\]\[]*"#, with: "", options: .regularExpression)
        // bare file names with a typical extension: "Foo.swift", "photo.png", "x.ips"
        s = s.replacingOccurrences(
            of: #"\b[\w\-.]*\w\.(swift|m|mm|c|cc|cpp|h|hpp|py|rs|txt|json|log|ips|crash|png|jpe?g|heic|gif|tiff?|mp4|mov|m4a|wav|mp3|aiff?|safetensors|gguf|bin|pt|pth|plist|app|dylib|framework)\b"#,
            with: "", options: [.regularExpression, .caseInsensitive])
        for name in [userName, NSFullUserName()] where name.count >= 2 {
            s = s.replacingOccurrences(of: name, with: "", options: .caseInsensitive)
        }
        s = s.replacingOccurrences(of: #"\s{2,}"#, with: " ", options: .regularExpression)
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// On launch, sends crash reports of this app made since the last check, but only while the person has switched
/// "Help The Dev Improve" on. Silent on every error and never in the way of the launch.
enum AppCrashReporter {
    static let lastCheckKey = "telemetryCrashScanAfter"

    static func scheduleLaunchCheck() {
        let defaults = UserDefaults.standard
        if defaults.object(forKey: lastCheckKey) == nil {
            defaults.set(Date().timeIntervalSince1970, forKey: lastCheckKey)   // older crashes are never sent
            return
        }
        Task.detached(priority: .utility) {
            try? await Task.sleep(nanoseconds: 6_000_000_000)
            // the engine may still be starting: ask it a few times
            var info: TelemetryInfo?
            for _ in 0..<10 {
                do {
                    info = try await APIClient.current().telemetry()
                    break
                } catch APIError.http {
                    return                       // no such endpoint (or it refused): nothing to do
                } catch {
                    try? await Task.sleep(nanoseconds: 10_000_000_000)
                }
            }
            guard let info else { return }
            guard info.enabled else {
                // switched off: crashes from now are not saved up to send after a later yes
                UserDefaults.standard.set(Date().timeIntervalSince1970, forKey: lastCheckKey)
                return
            }
            await sendNew()
        }
    }

    static func reportsFolder() -> URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/DiagnosticReports")
    }

    static var executableName: String {
        Bundle.main.executableURL?.lastPathComponent ?? "Shutter"
    }

    /// The app's own .ips files newer than `after`, oldest first.
    static func newReports(in folder: URL, executable: String, after: Double) -> [(url: URL, date: Double)] {
        let keys: [URLResourceKey] = [.contentModificationDateKey, .isRegularFileKey]
        guard let items = try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: keys)
        else { return [] }
        return items.compactMap { url -> (URL, Double)? in
            let name = url.lastPathComponent
            guard url.pathExtension == "ips", name.hasPrefix(executable + "-") || name.hasPrefix(executable + "_")
            else { return nil }
            guard let v = try? url.resourceValues(forKeys: Set(keys)), v.isRegularFile == true,
                  let d = v.contentModificationDate?.timeIntervalSince1970, d > after else { return nil }
            return (url, d)
        }
        .sorted { $0.1 < $1.1 }
    }

    private static func sendNew() async {
        let defaults = UserDefaults.standard
        let after = defaults.double(forKey: lastCheckKey)
        let exe = executableName
        for item in newReports(in: reportsFolder(), executable: exe, after: after) {
            if let text = try? String(contentsOf: item.url, encoding: .utf8),
               let report = CrashReportParser.parse(text, imageName: exe) {
                do {
                    try await APIClient.current().sendAppCrash(report)
                } catch {
                    return              // try again next launch (timestamp not moved past this one)
                }
            }
            defaults.set(item.date, forKey: lastCheckKey)
        }
    }
}
