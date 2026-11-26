import AppKit
import Foundation
import SwiftUI

/// Keeps the engine in ~/.shutter in step with the app. After an app update the engine sources bundled in the app
/// (Contents/Resources/engine) are newer than the copy in ~/.shutter/engine; at launch this runs the bundled
/// bootstrap.sh again, which reuses the Python environment, swaps in the new engine files and installs any new
/// packages. Models, config.json, the job history and outputs are not touched, and nothing is downloaded again
/// apart from Python packages that changed.
///
/// If the upgrade fails, the previous engine folder is put back so the old engine keeps working, and the row in the
/// sidebar offers Copy Diagnostics. Output goes to ~/Library/Logs/Shutter/engine-upgrade.log.
@MainActor
final class EngineUpgrader: ObservableObject {
    static let shared = EngineUpgrader()

    enum Phase: Equatable {
        case idle
        case running(String)
        case done(String)
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    private var process: Process?
    private var partial = ""
    private var failSentence = ""

    nonisolated init() {}

    // MARK: Versions

    /// `__version__` from <dir>/shutter/__init__.py.
    nonisolated static func version(ofEngineAt dir: URL) -> String? {
        let file = dir.appendingPathComponent("shutter/__init__.py")
        guard let text = try? String(contentsOf: file, encoding: .utf8) else { return nil }
        for line in text.components(separatedBy: "\n") {
            let t = line.trimmingCharacters(in: .whitespaces)
            guard t.hasPrefix("__version__") else { continue }
            let parts = t.split(separator: "=", maxSplits: 1)
            guard parts.count == 2 else { continue }
            let v = parts[1].trimmingCharacters(in: .whitespaces).trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
            return v.isEmpty ? nil : v
        }
        return nil
    }

    static var installedEngineDir: URL { SetupRunner.shutterHome.appendingPathComponent("engine") }

    static var installedVersion: String? { version(ofEngineAt: installedEngineDir) }

    static var bundledVersion: String? { SetupRunner.bundledEngine.flatMap { version(ofEngineAt: $0) } }

    /// True when the app installed the engine (~/.shutter/engine exists) and the bundled engine has another version.
    /// An engine installed from a source checkout (scripts/install.sh, no engine folder) is left alone.
    static var needsUpgrade: Bool {
        guard SetupRunner.engineInstalled,
              let installed = installedVersion, let bundled = bundledVersion else { return false }
        return installed != bundled
    }

    nonisolated static var logURL: URL {
        let lib = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library")
        return lib.appendingPathComponent("Logs/Shutter/engine-upgrade.log")
    }

    var isRunning: Bool { if case .running = phase { return true } else { return false } }

    // MARK: Run

    /// Upgrades the engine, then calls `completion(true)` on success or `completion(false)` after restoring the old one.
    func run(completion: @escaping (Bool) -> Void) {
        guard process == nil, let source = SetupRunner.bundledEngine else { completion(false); return }
        let home = SetupRunner.shutterHome
        let engineDir = Self.installedEngineDir
        let backup = home.appendingPathComponent("engine.prev")
        let from = Self.installedVersion ?? "?"
        let to = Self.bundledVersion ?? "?"
        let fm = FileManager.default

        try? fm.createDirectory(at: Self.logURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        fm.createFile(atPath: Self.logURL.path, contents: nil)
        let log = try? FileHandle(forWritingTo: Self.logURL)
        func write(_ s: String) { log?.write(Data((s + "\n").utf8)) }
        write("Upgrading the engine from \(from) to \(to) (\(Date()))")

        // The engine files in the app must be readable before anything is touched.
        let python = home.appendingPathComponent("venv/bin/python").path
        let check = Diagnostics.run(python, ["-I", "-c",
            "import sys, tomllib; tomllib.load(open(sys.argv[1], 'rb')); open(sys.argv[2]).read()",
            source.appendingPathComponent("pyproject.toml").path,
            source.appendingPathComponent("shutter/__init__.py").path])
        if !check.isEmpty {
            write("Bundled engine files are damaged: \(check)")
            try? log?.close()
            phase = .failed("The engine files in this Shuttrr are damaged, so the engine was not updated. The previous engine still works. Download Shuttrr again.")
            completion(false)
            return
        }

        // Keep a copy of the working engine so a failed upgrade can be undone.
        try? fm.removeItem(at: backup)
        do {
            try fm.copyItem(at: engineDir, to: backup)
        } catch {
            write("Could not back up \(engineDir.path): \(error.localizedDescription)")
            phase = .failed("Could not update the engine: \(error.localizedDescription)")
            completion(false)
            return
        }

        failSentence = ""
        phase = .running("Updating the engine to \(to)…")
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/bash")
        p.arguments = [source.appendingPathComponent("bootstrap.sh").path, "--source", source.path, "--copy"]
        p.currentDirectoryURL = URL(fileURLWithPath: NSHomeDirectory())
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + (env["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin")
        env["PYTHONUNBUFFERED"] = "1"
        env["SHUTTER_HOME"] = home.path
        p.environment = env
        p.standardInput = FileHandle.nullDevice
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            guard !data.isEmpty else { return }
            log?.write(data)
            let text = String(decoding: data, as: UTF8.self)
            DispatchQueue.main.async { MainActor.assumeIsolated { self?.consume(text, to: to) } }
        }
        p.terminationHandler = { [weak self] proc in
            let status = proc.terminationStatus
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    pipe.fileHandleForReading.readabilityHandler = nil
                    if let rest = try? pipe.fileHandleForReading.readToEnd() { log?.write(rest) }
                    guard let self else { return }
                    self.process = nil
                    if status == 0, Self.installedVersion == to {
                        try? FileManager.default.removeItem(at: backup)
                        write("Done: engine \(to)")
                        try? log?.close()
                        self.phase = .done("Engine updated to \(to).")
                        completion(true)
                    } else {
                        // Put the old engine back. The Python environment and models were never removed.
                        try? FileManager.default.removeItem(at: engineDir)
                        do { try FileManager.default.moveItem(at: backup, to: engineDir) } catch {
                            write("Could not restore the old engine: \(error.localizedDescription)")
                        }
                        write("Upgrade failed (exit \(status)); restored engine \(Self.installedVersion ?? "?")")
                        try? log?.close()
                        let why = self.failSentence.isEmpty ? "" : " (\(self.failSentence))"
                        self.phase = .failed("The engine update to \(to) failed\(why). The previous engine still works.")
                        completion(false)
                    }
                }
            }
        }
        do {
            try p.run()
            process = p
        } catch {
            pipe.fileHandleForReading.readabilityHandler = nil
            try? FileManager.default.removeItem(at: backup)
            write("Could not start bootstrap.sh: \(error.localizedDescription)")
            phase = .failed("Could not start the engine update: \(error.localizedDescription)")
            completion(false)
        }
    }

    /// Progress lines from bootstrap.sh look like `@@ <step> start|done|fail <sentence>`.
    private func consume(_ text: String, to: String) {
        let combined = partial + text
        var lines = combined.components(separatedBy: "\n")
        partial = lines.removeLast()
        for line in lines where line.hasPrefix("@@ ") {
            let f = line.dropFirst(3).split(separator: " ", maxSplits: 2)
            if f.count == 3, f[1] == "fail" { failSentence = String(f[2]) }
            if f.count == 3, f[1] == "start" {
                phase = .running("Updating the engine to \(to): \(f[2])")
            }
        }
    }

    func dismiss() {
        if case .running = phase { return }
        phase = .idle
    }
}

/// A short line while the engine updates itself after an app update, and a hint when that failed.
struct EngineUpgradeRow: View {
    @ObservedObject var upgrader = EngineUpgrader.shared
    @EnvironmentObject private var store: EngineStore
    @State private var copied = false

    var body: some View {
        switch upgrader.phase {
        case .idle, .done:
            EmptyView()
        case .running(let text):
            row {
                ProgressView().controlSize(.small)
                Text(text)
                    .font(Theme.Fonts.meta)
                    .foregroundStyle(Theme.Palette.ink2)
                    .fixedSize(horizontal: false, vertical: true)
            }
        case .failed(let text):
            row {
                Image(systemName: "exclamationmark.triangle").foregroundStyle(Theme.Palette.danger)
                VStack(alignment: .leading, spacing: Theme.Space.s1) {
                    Text(text)
                        .font(Theme.Fonts.meta)
                        .foregroundStyle(Theme.Palette.ink)
                        .fixedSize(horizontal: false, vertical: true)
                    Button(copied ? "Copied" : "Copy Diagnostics") {
                        Task {
                            let s = await Diagnostics.collect(engineVersion: store.system?.version)
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(s, forType: .string)
                            copied = true
                        }
                    }
                    .buttonStyle(.kit(.line, small: true))
                }
            }
        }
    }

    private func row<C: View>(@ViewBuilder _ content: () -> C) -> some View {
        HStack(alignment: .top, spacing: Theme.Space.s2) { content() }
            .padding(Theme.Space.s2)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous).fill(Theme.Palette.fill))
    }
}
