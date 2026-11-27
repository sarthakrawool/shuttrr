import AppKit
import Foundation
import Sparkle
import SwiftUI

// In-app updates with Sparkle 2.
//
// The feed is appcast.xml on the latest GitHub release (SUFeedURL in Info.plist). Every update archive is signed
// with Shuttrr's EdDSA key (scripts/release.sh, private key in the release Mac's Keychain); Sparkle refuses an
// archive whose signature does not match SUPublicEDKey.
//
// Ask first (since 0.2.2): Shuttrr checks at launch and every 3 hours (once a day before 0.2.7). When a newer version is on GitHub, Sparkle's
// window says "Shuttrr 0.2.x is now available, you have 0.2.y", shows the release notes and offers Install Update /
// Remind Me Later / Skip This Version. Nothing is downloaded until the person says yes; then Sparkle offers
// "Install and Relaunch". Quitting stops the engine first (AppDelegate.applicationShouldTerminate), and after the
// new app starts, EngineUpgrader brings ~/.shutter/engine up to the bundled version.
//
// 0.2.1 downloaded updates silently (Sparkle stored SUAutomaticallyUpdate = YES in the app's defaults). Every
// launch now forces automatic downloads off, which also migrates those installs. The "Check for updates
// automatically" switch in Settings only turns the launch/daily checks on or off; "Check for Updates…" always works.

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
        // 0.2.1 let Sparkle store "download updates by themselves" (SUAutomaticallyUpdate = YES). Ask first instead.
        let d = UserDefaults.standard
        if !d.bool(forKey: "updatesAskFirstMigrated") {
            d.removeObject(forKey: "SUAutomaticallyUpdate")
            d.set(true, forKey: "updatesAskFirstMigrated")
        }
    }

    /// Every 3 hours while Shuttrr is open (it usually stays open in the menu bar). Was once a day until 0.2.7,
    /// which left an app open since the morning on the old version all day.
    nonisolated static let checkInterval: TimeInterval = 3 * 3600

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
        u.automaticallyDownloadsUpdates = false      // ask first; also undoes 0.2.1's stored YES
        u.updateCheckInterval = Self.checkInterval
    }

    /// The switch in Settings changed.
    func enabledChanged(_ on: Bool) {
        apply(on)
        if on { controller?.updater.checkForUpdatesInBackground() }
        objectWillChange.send()
    }

    var canCheckNow: Bool { controller?.updater.canCheckForUpdates ?? false }

    /// "Check for Updates…": Sparkle's own window shows the result ("You're up to date!" with the version, or the
    /// update offer).
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
            QuitConfirm.shared.quitEverything = true      // installing quits completely (engine too)
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
        // Only reached for silent downloads, which 0.2.2 no longer does (a 0.2.1 install updating itself does).
        return true      // our own "Restart to Update" row instead of Sparkle's alert
    }

    /// Sparkle is about to quit Shuttrr to install an update and open the new version: quit completely.
    nonisolated func updaterWillRelaunchApplication(_ updater: SPUUpdater) {
        if Thread.isMainThread {
            MainActor.assumeIsolated { QuitConfirm.shared.quitEverything = true }
        } else {
            DispatchQueue.main.sync { MainActor.assumeIsolated { QuitConfirm.shared.quitEverything = true } }
        }
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

/// Sidebar footer: the version and a Check for Updates button.
struct SidebarVersionFooter: View {
    @ObservedObject var updates = UpdateChecker.shared

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s2) {
            Text("Shuttrr \(Releases.appVersion)")
                .font(Theme.Fonts.meta)
                .foregroundStyle(Theme.Palette.muted)
                .monospacedDigit()
                .lineLimit(1)
                .fixedSize()
            Button("Check for Updates…") { updates.checkNow() }
                .buttonStyle(.kit(.line, small: true))
                .help("Ask GitHub for a newer Shuttrr. Nothing downloads unless you say yes.")
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, Theme.Space.s2)
        .padding(.top, Theme.Space.s2)
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
