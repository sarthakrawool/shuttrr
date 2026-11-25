import Foundation
import SwiftUI

/// Runs the engine installer that ships inside the app (Contents/Resources/engine/bootstrap.sh) and turns its
/// progress lines (`@@ <step> start|done|fail <sentence>`) into five steps the onboarding window shows.
@MainActor
final class SetupRunner: ObservableObject {
    enum StepState: Equatable { case pending, running, done, failed }

    struct Step: Identifiable, Equatable {
        let id: String
        let title: String
        var state: StepState = .pending
        var detail: String = ""
    }

    enum Phase: Equatable {
        case idle
        case running
        case finished
        case failed(String)
    }

    static let onboardingDoneKey = "onboardingDone"
    static let maxLogLines = 400
    static let stepTitles: [(String, String)] = [
        ("check", "Check this Mac"),
        ("python", "Get Python ready"),
        ("engine", "Install the Shuttrr engine"),
        ("packages", "Install what the models need"),
        ("finish", "Finish up"),
    ]

    @Published private(set) var steps: [Step] = SetupRunner.freshSteps()
    @Published private(set) var phase: Phase = .idle
    @Published private(set) var logLines: [String] = []

    private var process: Process?
    private var partial = ""

    nonisolated init() {}

    private static func freshSteps() -> [Step] {
        stepTitles.map { Step(id: $0.0, title: $0.1) }
    }

    /// 0…1: finished steps count fully, the running one counts half.
    var progress: Double {
        let total = Double(steps.count)
        guard total > 0 else { return 0 }
        let done = Double(steps.filter { $0.state == .done }.count)
        let running = steps.contains { $0.state == .running } ? 0.5 : 0
        return min(1, (done + running) / total)
    }

    var isRunning: Bool { phase == .running }

    // MARK: Static helpers

    static var shutterHome: URL {
        if let env = ProcessInfo.processInfo.environment["SHUTTER_HOME"], !env.isEmpty {
            return URL(fileURLWithPath: (env as NSString).expandingTildeInPath)
        }
        return URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".shutter")
    }

    /// The engine is installed when ~/.shutter/venv/bin/shutter exists and can be run.
    static var engineInstalled: Bool {
        FileManager.default.isExecutableFile(atPath: shutterHome.appendingPathComponent("venv/bin/shutter").path)
    }

    static var needsOnboarding: Bool {
        !UserDefaults.standard.bool(forKey: onboardingDoneKey) || !engineInstalled
    }

    /// The engine sources copied into the app bundle by build_app.sh.
    static var bundledEngine: URL? {
        guard let r = Bundle.main.resourceURL?.appendingPathComponent("engine") else { return nil }
        return FileManager.default.fileExists(atPath: r.appendingPathComponent("bootstrap.sh").path) ? r : nil
    }

    // MARK: Run

    func run() {
        guard process == nil else { return }
        steps = Self.freshSteps()
        logLines = []
        partial = ""
        guard let engineDir = Self.bundledEngine else {
            fail(step: "check", "The installer is missing from the app. Download Shuttrr again and try again.")
            return
        }
        phase = .running

        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/bash")
        p.arguments = [engineDir.appendingPathComponent("bootstrap.sh").path, "--source", engineDir.path, "--copy"]
        p.currentDirectoryURL = URL(fileURLWithPath: NSHomeDirectory())
        var env = ProcessInfo.processInfo.environment
        let path = env["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin"
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + path
        env["PYTHONUNBUFFERED"] = "1"
        p.environment = env
        p.standardInput = FileHandle.nullDevice

        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            let text = String(decoding: data, as: UTF8.self)
            DispatchQueue.main.async {
                MainActor.assumeIsolated { self?.consume(text) }
            }
        }
        p.terminationHandler = { [weak self] proc in
            let status = proc.terminationStatus
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    pipe.fileHandleForReading.readabilityHandler = nil
                    if let rest = try? pipe.fileHandleForReading.readToEnd(), !rest.isEmpty {
                        self?.consume(String(decoding: rest, as: UTF8.self))
                    }
                    self?.didExit(status: status)
                }
            }
        }
        appendLog("$ bash bootstrap.sh --source \(engineDir.path) --copy")
        do {
            try p.run()
            process = p
        } catch {
            pipe.fileHandleForReading.readabilityHandler = nil
            fail(step: "check", "Could not start the installer: \(error.localizedDescription)")
        }
    }

    private func consume(_ text: String) {
        let combined = partial + text
        var parts = combined.components(separatedBy: "\n")
        partial = parts.removeLast()
        for line in parts { handle(line) }
    }

    private func handle(_ raw: String) {
        let line = raw.trimmingCharacters(in: CharacterSet(charactersIn: "\r"))
        appendLog(line)
        guard line.hasPrefix("@@ ") else { return }
        let fields = line.dropFirst(3).split(separator: " ", maxSplits: 2, omittingEmptySubsequences: false)
        guard fields.count >= 2 else { return }
        let id = String(fields[0])
        let what = String(fields[1])
        let sentence = fields.count > 2 ? String(fields[2]) : ""
        guard let i = steps.firstIndex(where: { $0.id == id }) else { return }
        switch what {
        case "start":
            steps[i].state = .running
            steps[i].detail = sentence
        case "done":
            steps[i].state = .done
            steps[i].detail = sentence
        case "fail":
            steps[i].state = .failed
            steps[i].detail = sentence
            phase = .failed(sentence)
        default:
            break
        }
    }

    private func didExit(status: Int32) {
        if !partial.isEmpty { handle(partial); partial = "" }
        process = nil
        if case .failed = phase { return }
        if status == 0 && steps.allSatisfy({ $0.state == .done }) {
            phase = .finished
        } else {
            let reason = "The installer stopped early (exit \(status)). Show the details, then try again."
            if let i = steps.firstIndex(where: { $0.state == .running }) ?? steps.firstIndex(where: { $0.state == .pending }) {
                steps[i].state = .failed
                steps[i].detail = reason
            }
            phase = .failed(reason)
        }
    }

    private func fail(step id: String, _ reason: String) {
        if let i = steps.firstIndex(where: { $0.id == id }) {
            steps[i].state = .failed
            steps[i].detail = reason
        }
        appendLog(reason)
        phase = .failed(reason)
    }

    private func appendLog(_ line: String) {
        logLines.append(line)
        if logLines.count > Self.maxLogLines {
            logLines.removeFirst(logLines.count - Self.maxLogLines)
        }
    }
}
