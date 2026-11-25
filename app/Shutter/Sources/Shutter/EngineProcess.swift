import Foundation
import SwiftUI

/// Starts and stops the Python daemon as a child process.
///
/// - The command comes from Settings (default `~/.shutter/venv/bin/shutter serve`) and runs
///   through a login shell so `~` and PATH work: `/bin/zsh -l -c "exec <command>"`.
/// - Output goes to ~/Library/Logs/Shutter/engine.log (not a pipe), so the daemon can keep
///   running safely after the app quits. The app tails that file and keeps the last 200 lines.
/// - Only a daemon this object started is ever stopped. An engine that was already running
///   (started from Terminal, launchd, ...) is left alone.
@MainActor
final class EngineProcess: ObservableObject {
    static let commandKey = "daemonCommand"
    /// Start the engine when Shuttrr opens (if nothing answers on its port). On by default.
    static let autoStartKey = "autoStartEngine"
    static let defaultCommand = "~/.shutter/venv/bin/shutter serve"
    static let maxLines = 200

    @Published private(set) var isRunning = false
    @Published private(set) var logLines: [String] = []
    @Published private(set) var lastExit: String?

    private var process: Process?
    private var tailTimer: Timer?
    private var readOffset: UInt64 = 0
    private var partialLine = ""

    nonisolated init() {}

    static var logURL: URL {
        let lib = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library")
        return lib.appendingPathComponent("Logs/Shutter/engine.log")
    }

    var command: String {
        let s = UserDefaults.standard.string(forKey: Self.commandKey) ?? ""
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        // v0.1 runs only the Python engine (`shutter serve`). A command saved by an older build is kept only when it
        // starts that engine; anything else (such as the Rust daemon, shutterd) falls back to the default.
        let python = (t.contains("shutter serve") || t.contains("shutter.cli serve")) && !t.contains("shutterd")
        return python ? t : Self.defaultCommand
    }

    var pid: Int32? { process?.processIdentifier }

    // MARK: Start / stop

    func start() {
        guard process == nil else { return }
        let logURL = Self.logURL
        do {
            try FileManager.default.createDirectory(at: logURL.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            FileManager.default.createFile(atPath: logURL.path, contents: nil) // truncate
        } catch {
            append("Could not create log file: \(error.localizedDescription)\n")
            return
        }
        guard let out = try? FileHandle(forWritingTo: logURL) else {
            append("Could not open log file at \(logURL.path)\n")
            return
        }

        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/zsh")
        p.arguments = ["-l", "-c", "exec " + command]
        p.standardOutput = out
        p.standardError = out
        p.standardInput = FileHandle.nullDevice
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        p.environment = env
        p.terminationHandler = { [weak self] proc in
            let status = proc.terminationStatus
            let reason = proc.terminationReason
            guard let owner = self else { return }
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    owner.didExit(status: status, reason: reason)
                }
            }
        }

        logLines = []
        partialLine = ""
        readOffset = 0
        lastExit = nil
        append("$ \(command)\n")
        do {
            try p.run()
        } catch {
            append("Failed to start: \(error.localizedDescription)\n")
            try? out.close()
            return
        }
        try? out.close() // the child holds its own copy
        process = p
        isRunning = true
        startTailing()
    }

    /// Sends SIGTERM to the daemon we started; SIGKILL if it is still alive after 10 s.
    func stop() {
        guard let p = process, p.isRunning else { return }
        let pid = p.processIdentifier
        p.terminate()
        DispatchQueue.main.asyncAfter(deadline: .now() + 10) {
            if p.isRunning {
                kill(pid, SIGKILL)
            }
        }
    }

    /// Stop the engine this app started and wait for it (TERM, then KILL after `seconds`). Used on quit.
    func stopAndWait(seconds: Double) {
        guard let p = process, p.isRunning else { return }
        let pid = p.processIdentifier
        p.terminationHandler = nil
        p.terminate()
        let deadline = Date().addingTimeInterval(seconds)
        while Date() < deadline, p.isRunning { usleep(100_000) }
        if p.isRunning { kill(pid, SIGKILL) }
        process = nil
        isRunning = false
    }

    /// Forget the child without stopping it (used when the user keeps the engine
    /// running after quitting the app).
    func detach() {
        process?.terminationHandler = nil
        process = nil
        isRunning = false
        stopTailing()
    }

    private func didExit(status: Int32, reason: Process.TerminationReason) {
        readNewOutput()
        stopTailing()
        process = nil
        isRunning = false
        let how = reason == .uncaughtSignal ? "signal \(status)" : "status \(status)"
        lastExit = "Engine exited (\(how))"
        append("[\(lastExit ?? "")]\n")
    }

    // MARK: Log tail

    private func startTailing() {
        stopTailing()
        let t = Timer(timeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let owner = self else { return }
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    owner.readNewOutput()
                }
            }
        }
        RunLoop.main.add(t, forMode: .common)
        tailTimer = t
    }

    private func stopTailing() {
        tailTimer?.invalidate()
        tailTimer = nil
    }

    private func readNewOutput() {
        guard let h = try? FileHandle(forReadingFrom: Self.logURL) else { return }
        defer { try? h.close() }
        do {
            let end = try h.seekToEnd()
            if end < readOffset { readOffset = 0 } // file was truncated
            guard end > readOffset else { return }
            try h.seek(toOffset: readOffset)
            if let data = try h.readToEnd() {
                readOffset += UInt64(data.count)
                append(String(decoding: data, as: UTF8.self))
            }
        } catch {
            return
        }
    }

    private func append(_ text: String) {
        let combined = partialLine + text
        var parts = combined.components(separatedBy: "\n")
        partialLine = parts.removeLast() // incomplete trailing line (or "")
        guard !parts.isEmpty else { return }
        logLines.append(contentsOf: parts)
        if logLines.count > Self.maxLines {
            logLines.removeFirst(logLines.count - Self.maxLines)
        }
    }
}
