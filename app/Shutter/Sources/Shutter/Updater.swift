import AppKit
import Foundation
import Sparkle
import SwiftUI

// In-app updates with Sparkle 2.
//
// The feed is appcast.xml on the latest GitHub release (SUFeedURL in Info.plist). Every update archive is signed
// with Shuttrr's EdDSA key (scripts/release.sh, private key in the release Mac's Keychain); Sparkle refuses an
// archive whose signature does not match SUPublicEDKey. Updates are downloaded in the background and installed when
// Shuttrr quits, or straight away from the "Restart to Update" button. Quitting stops the engine first
// (AppDelegate.applicationShouldTerminate), and after the new app starts, EngineUpgrader brings ~/.shutter/engine
// up to the bundled version.
//
// The "Check for updates" switch in Settings turns all of this off: no checks, no downloads.

@MainActor
final class UpdateChecker: NSObject, ObservableObject {
    static let shared = UpdateChecker()
    static let enabledKey = "checkForUpdates"
    /// The production bundle id. Test hooks below only work in builds with another bundle id.
    nonisolated static let productionBundleID = "com.shutter.app"

    /// The version downloaded and waiting for a restart (nil when there is none).
    @Published private(set) var readyVersion: String?
    @Published private(set) var lastError: String?

    private var controller: SPUStandardUpdaterController?
    private var installNowBlock: (() -> Void)?

    private override init() {
        super.init()
        UserDefaults.standard.register(defaults: [Self.enabledKey: true])
    }

    private var enabled: Bool { UserDefaults.standard.bool(forKey: Self.enabledKey) }

    private var isTestBuild: Bool { (Bundle.main.bundleIdentifier ?? "") != Self.productionBundleID }

    /// Sparkle can only update a real app bundle (not `swift run`).
    private var canUpdate: Bool {
        Bundle.main.bundleURL.pathExtension == "app"
            && Bundle.main.object(forInfoDictionaryKey: "SUPublicEDKey") != nil
    }

    var updater: SPUUpdater? { controller?.updater }

    /// Called once at launch: starts Sparkle and, when the switch is on, checks right away (and then daily).
    func start() {
        guard controller == nil, canUpdate else { return }
        let c = SPUStandardUpdaterController(startingUpdater: false, updaterDelegate: self, userDriverDelegate: nil)
        controller = c
        apply(enabled)
        do {
            try c.updater.start()
        } catch {
            lastError = error.localizedDescription
            return
        }
        if enabled { c.updater.checkForUpdatesInBackground() }
    }

    /// Kept for older call sites: the launch check.
    func checkIfDue() { start() }

    private func apply(_ on: Bool) {
        guard let u = controller?.updater else { return }
        u.automaticallyChecksForUpdates = on
        u.automaticallyDownloadsUpdates = on
        u.updateCheckInterval = 24 * 3600
    }

    /// The switch in Settings changed.
    func enabledChanged(_ on: Bool) {
        apply(on)
        if on { controller?.updater.checkForUpdatesInBackground() }
        objectWillChange.send()
    }

    var canCheckNow: Bool { controller?.updater.canCheckForUpdates ?? false }

    /// "Check for Updates…": Sparkle's own window shows the result.
    func checkNow() {
        guard let c = controller else {
            NSWorkspace.shared.open(Releases.latestPage)
            return
        }
        c.checkForUpdates(nil)
    }

    /// "Restart to Update": quits Shuttrr (stopping the engine), installs, and opens the new version.
    func installNow() {
        if let block = installNowBlock {
            installNowBlock = nil
            block()
        } else {
            checkNow()
        }
    }
}

extension UpdateChecker: SPUUpdaterDelegate {
    nonisolated func feedURLString(for updater: SPUUpdater) -> String? {
        // Test builds (another bundle id) may point at a local feed; the shipped app always uses Info.plist.
        guard (Bundle.main.bundleIdentifier ?? "") != UpdateChecker.productionBundleID else { return nil }
        let s = UserDefaults.standard.string(forKey: "updateFeedURL") ?? ""
        return s.isEmpty ? nil : s
    }

    nonisolated func updater(_ updater: SPUUpdater, willInstallUpdateOnQuit item: SUAppcastItem,
                             immediateInstallationBlock immediateInstallHandler: @escaping () -> Void) -> Bool {
        let version = item.displayVersionString
        DispatchQueue.main.async {
            MainActor.assumeIsolated {
                let s = UpdateChecker.shared
                s.readyVersion = version
                s.installNowBlock = immediateInstallHandler
                if s.isTestBuild && UserDefaults.standard.bool(forKey: "updateInstallImmediately") {
                    s.installNow()
                }
            }
        }
        return true      // we show our own "Restart to Update" row instead of Sparkle's alert
    }

    nonisolated func updater(_ updater: SPUUpdater, didAbortWithError error: Error) {
        let ns = error as NSError
        // "No update found" also ends here; only real failures are remembered (shown in Copy Diagnostics).
        if ns.domain == SUSparkleErrorDomain && ns.code == Int(SUError.noUpdateError.rawValue) { return }
        let text = "\(ns.localizedDescription) (\(ns.domain) \(ns.code))"
            + ((ns.userInfo[NSUnderlyingErrorKey] as? NSError).map { ": \($0.localizedDescription)" } ?? "")
        UserDefaults.standard.set(text, forKey: "updateLastError")
        DispatchQueue.main.async { MainActor.assumeIsolated { UpdateChecker.shared.lastError = text } }
    }
}

/// "Shuttrr 0.2.2 is ready. Restart to update." Shown in the sidebar and in Settings once an update is downloaded.
struct UpdateAvailableRow: View {
    @ObservedObject var updates = UpdateChecker.shared

    var body: some View {
        if let v = updates.readyVersion {
            Button { updates.installNow() } label: {
                HStack(spacing: Theme.Space.s2) {
                    Image(systemName: "arrow.down.circle")
                        .foregroundStyle(Theme.Palette.ink)
                    VStack(alignment: .leading, spacing: 0) {
                        Text("Shuttrr \(v) is ready.")
                            .font(Theme.Fonts.row)
                            .foregroundStyle(Theme.Palette.ink)
                        Text("Restart to update")
                            .font(Theme.Fonts.meta)
                            .foregroundStyle(Theme.Palette.muted)
                    }
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, Theme.Space.s2)
                .padding(.vertical, Theme.Space.s2)
                .background(RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous).fill(Theme.Palette.fill))
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Quits Shuttrr (the engine stops), installs \(v) and opens it again. Your models and settings stay.")
        }
    }
}
