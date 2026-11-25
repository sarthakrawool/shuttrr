import AppKit
import Foundation
import SwiftUI

// Release support: update check, diagnostics for bug reports, uninstall, and the brand-themed confirm sheet
// (used for quitting while a request runs and for uninstalling).

// MARK: - Releases

/// Where releases are published. One place to change when the repository moves.
enum Releases {
    static let owner = "OWNER"          // GitHub user or organisation that publishes Shuttrr
    static let repo = "shuttrr"
    static let latestAPI = URL(string: "https://api.github.com/repos/\(owner)/\(repo)/releases/latest")!
    static let latestPage = URL(string: "https://github.com/\(owner)/\(repo)/releases/latest")!
    static let newIssue = URL(string: "https://github.com/\(owner)/\(repo)/issues/new/choose")!

    static var appVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0.0.0"
    }

    /// True when `tag` (like "v0.2.0" or "0.2.0") is a newer version than `current`.
    static func isNewer(_ tag: String, than current: String) -> Bool {
        func parts(_ s: String) -> [Int] {
            let t = s.trimmingCharacters(in: .whitespaces).lowercased()
            let core = (t.hasPrefix("v") ? String(t.dropFirst()) : t).split(separator: "-").first.map(String.init) ?? ""
            return core.split(separator: ".").map { Int($0) ?? 0 }
        }
        let a = parts(tag), b = parts(current)
        for i in 0..<max(a.count, b.count) {
            let x = i < a.count ? a[i] : 0, y = i < b.count ? b[i] : 0
            if x != y { return x > y }
        }
        return false
    }
}

/// Checks GitHub's latest release at most once a day (when "Check for updates" is on) and remembers what it found.
/// It never downloads anything: the row it shows links to the release page.
@MainActor
final class UpdateChecker: ObservableObject {
    static let shared = UpdateChecker()
    static let enabledKey = "checkForUpdates"
    private static let lastCheckKey = "updateLastCheck"
    private static let latestTagKey = "updateLatestTag"

    @Published private(set) var latestTag: String?

    var updateAvailable: Bool {
        guard UserDefaults.standard.bool(forKey: Self.enabledKey), let t = latestTag else { return false }
        return Releases.isNewer(t, than: Releases.appVersion)
    }

    private init() {
        UserDefaults.standard.register(defaults: [Self.enabledKey: true])
        latestTag = UserDefaults.standard.string(forKey: Self.latestTagKey)
    }

    /// Called at launch: asks GitHub only when the switch is on and the last check is more than a day old.
    func checkIfDue() {
        let d = UserDefaults.standard
        guard d.bool(forKey: Self.enabledKey) else { return }
        let last = d.double(forKey: Self.lastCheckKey)
        guard Date().timeIntervalSince1970 - last > 24 * 3600 else { return }
        Task { await check() }
    }

    func check() async {
        var req = URLRequest(url: Releases.latestAPI, timeoutInterval: 15)
        req.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        req.setValue("Shuttrr/\(Releases.appVersion)", forHTTPHeaderField: "User-Agent")
        guard let (data, resp) = try? await URLSession.shared.data(for: req),
              (resp as? HTTPURLResponse)?.statusCode == 200,
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let tag = obj["tag_name"] as? String, !tag.isEmpty else { return }
        UserDefaults.standard.set(Date().timeIntervalSince1970, forKey: Self.lastCheckKey)
        UserDefaults.standard.set(tag, forKey: Self.latestTagKey)
        latestTag = tag
    }

    /// The switch changed: forget a stale result when it is turned off so no row shows.
    func enabledChanged(_ on: Bool) {
        objectWillChange.send()
        if on { checkIfDue() }
    }
}

/// The small "Update available" row with a link to the release page.
struct UpdateAvailableRow: View {
    @ObservedObject var updates = UpdateChecker.shared

    var body: some View {
        if updates.updateAvailable, let tag = updates.latestTag {
            Link(destination: Releases.latestPage) {
                HStack(spacing: Theme.Space.s2) {
                    Image(systemName: "arrow.down.circle")
                        .foregroundStyle(Theme.Palette.ink)
                    VStack(alignment: .leading, spacing: 0) {
                        Text("Update available")
                            .font(Theme.Fonts.row)
                            .foregroundStyle(Theme.Palette.ink)
                        Text("\(tag) · see what's new")
                            .font(Theme.Fonts.meta)
                            .foregroundStyle(Theme.Palette.muted)
                    }
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, Theme.Space.s2)
                .padding(.vertical, Theme.Space.s2)
                .background(RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous).fill(Theme.Palette.fill))
            }
            .buttonStyle(.plain)
            .help("Opens the release page in your browser. Shuttrr never downloads updates by itself.")
        }
    }
}

// MARK: - Diagnostics

/// The text "Copy Diagnostics" puts on the clipboard: doctor output, this Mac, versions, and the last 100 log lines.
/// The user's home folder is written as ~ so no user name ends up in a bug report.
enum Diagnostics {
    @MainActor
    static func collect(engineVersion: String?) async -> String {
        let shutter = SetupRunner.shutterHome.appendingPathComponent("venv/bin/shutter").path
        let logURL = EngineProcess.logURL
        return await Task.detached(priority: .userInitiated) { () -> String in
            let home = NSHomeDirectory()
            var out: [String] = []
            out.append("## Shuttrr diagnostics")
            out.append("App: \(Releases.appVersion)")
            out.append("Engine: \(engineVersion?.isEmpty == false ? engineVersion! : "not running")")
            out.append("Mac: \(sysctl("hw.model"))")
            out.append("Chip: \(sysctl("machdep.cpu.brand_string"))")
            out.append("Memory: \(ProcessInfo.processInfo.physicalMemory / (1 << 30)) GB")
            out.append("macOS: \(ProcessInfo.processInfo.operatingSystemVersionString)")
            out.append("")
            out.append("## shutter --json doctor")
            if FileManager.default.isExecutableFile(atPath: shutter) {
                out.append(run(shutter, ["--json", "doctor"]))
            } else {
                out.append("(engine not installed: \(shutter) not found)")
            }
            out.append("")
            out.append("## Engine log (last 100 lines)")
            let log = (try? String(contentsOf: logURL, encoding: .utf8)) ?? ""
            var lines = log.components(separatedBy: "\n")
            while lines.last == "" { lines.removeLast() }
            out.append(lines.suffix(100).joined(separator: "\n"))
            return out.joined(separator: "\n").replacingOccurrences(of: home, with: "~")
        }.value
    }

    nonisolated static func sysctl(_ name: String) -> String {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "unknown" }
        var buf = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return "unknown" }
        return String(cString: buf)
    }

    nonisolated static func run(_ path: String, _ args: [String]) -> String {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: path)
        p.arguments = args
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        p.standardInput = FileHandle.nullDevice
        do { try p.run() } catch { return "(could not run: \(error.localizedDescription))" }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        return String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - Uninstall

/// Stops the engine and deletes the Shuttrr home folder (~/.shutter), optionally keeping downloaded models.
/// Uses `shutter uninstall --yes` when the engine is installed, so the app and the CLI do the same thing.
enum Uninstaller {
    @MainActor
    static func run(keepModels: Bool, engine: EngineProcess) async -> (ok: Bool, message: String) {
        engine.stopAndWait(seconds: 5)
        let base = APIClient.current().base
        let home = SetupRunner.shutterHome
        return await Task.detached(priority: .userInitiated) { () -> (Bool, String) in
            EngineStopper.stopShuttrrEngine(onPortOf: base, seconds: 5)
            let cli = home.appendingPathComponent("venv/bin/shutter").path
            if FileManager.default.isExecutableFile(atPath: cli) {
                var args = ["uninstall", "--yes"]
                if keepModels { args.append("--keep-models") }
                _ = Diagnostics.run(cli, args)
            }
            // Whatever the CLI left (or everything, when the engine was never installed).
            let fm = FileManager.default
            if let items = try? fm.contentsOfDirectory(at: home, includingPropertiesForKeys: nil) {
                for item in items where !(keepModels && item.lastPathComponent == "models") {
                    try? fm.removeItem(at: item)
                }
                if !keepModels { try? fm.removeItem(at: home) }
            }
            let left = (try? fm.contentsOfDirectory(atPath: home.path)) ?? []
            let clean = keepModels ? left.allSatisfy { $0 == "models" } : !fm.fileExists(atPath: home.path)
            let path = home.path.replacingOccurrences(of: NSHomeDirectory(), with: "~")
            if clean {
                return (true, keepModels
                    ? "Removed \(path), except your downloaded models in \(path)/models."
                    : "Removed \(path).")
            }
            return (false, "Some files in \(path) could not be removed: \(left.prefix(3).joined(separator: ", ")).")
        }.value
    }
}

// MARK: - Themed confirm sheet

/// The app's own confirm sheet (paper background, Fraunces title, kit buttons) in place of a system alert.
struct ConfirmSheet<Extra: View>: View {
    let title: String
    let message: String
    let confirmTitle: String
    var destructive = false
    var busy = false
    let onCancel: () -> Void
    let onConfirm: () -> Void
    @ViewBuilder var extra: () -> Extra

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s4) {
            Text(title)
                .font(Theme.Fonts.section)
                .foregroundStyle(Theme.Palette.ink)
                .fixedSize(horizontal: false, vertical: true)
            Text(message)
                .font(Theme.Fonts.row)
                .foregroundStyle(Theme.Palette.ink2)
                .fixedSize(horizontal: false, vertical: true)
            extra()
            HStack(spacing: Theme.Space.s3) {
                Spacer(minLength: 0)
                Button("Cancel", action: onCancel)
                    .buttonStyle(.kit(.line))
                    .keyboardShortcut(.cancelAction)
                    .disabled(busy)
                Button(action: onConfirm) {
                    HStack(spacing: Theme.Space.s2) {
                        if busy { ProgressView().controlSize(.small) }
                        Text(confirmTitle)
                    }
                }
                .buttonStyle(.kit(.ink))
                .tint(destructive ? Theme.Palette.danger : Theme.Palette.ink)
                .keyboardShortcut(.defaultAction)
                .disabled(busy)
            }
        }
        .padding(Theme.Space.s6)
        .frame(width: 440)
        .background(PaperBackground(glows: false))
    }
}

extension ConfirmSheet where Extra == EmptyView {
    init(title: String, message: String, confirmTitle: String, destructive: Bool = false, busy: Bool = false,
         onCancel: @escaping () -> Void, onConfirm: @escaping () -> Void) {
        self.init(title: title, message: message, confirmTitle: confirmTitle, destructive: destructive, busy: busy,
                  onCancel: onCancel, onConfirm: onConfirm, extra: { EmptyView() })
    }
}

/// Quitting while a request runs: the app delegate asks this to show the themed sheet in the main window and
/// waits (terminateLater) for the answer. The menu bar panel stays native.
@MainActor
final class QuitConfirm: ObservableObject {
    static let shared = QuitConfirm()
    @Published var asking = false
    /// Set before quitting on purpose (after Uninstall) so no question is asked.
    var skipNext = false
    /// Set by the app delegate: called with the user's answer.
    var answer: ((Bool) -> Void)?

    func reply(_ quit: Bool) {
        asking = false
        let a = answer
        answer = nil
        a?(quit)
    }
}
