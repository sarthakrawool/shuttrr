import AppKit
import SwiftUI

@main
struct ShutterApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        MenuBarExtra {
            MenuPanelView()
                .environmentObject(appDelegate.store)
                .environmentObject(appDelegate.engine)
        } label: {
            MenuBarIcon(store: appDelegate.store)
        }
        .menuBarExtraStyle(.window)

        Window("Shuttrr", id: AppDelegate.dashboardID) {
            DashboardView()
                .environmentObject(appDelegate.store)
                .environmentObject(appDelegate.engine)
                .environmentObject(appDelegate.flow)
        }
        .defaultSize(width: 1000, height: 640)
        .windowStyle(.hiddenTitleBar)
        .commands {
            CommandGroup(after: .appInfo) {
                Button("Check for Updates…") { UpdateChecker.shared.checkNow() }
            }
        }

        Window("Welcome to Shuttrr", id: OnboardingFlow.windowID) {
            OnboardingView()
                .environmentObject(appDelegate.store)
                .environmentObject(appDelegate.engine)
                .environmentObject(appDelegate.setup)
                .environmentObject(appDelegate.flow)
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentMinSize)
        .defaultSize(width: OnboardingView.width, height: OnboardingView.height)
        .defaultPosition(.center)
    }
}

/// Finds the process listening on the engine's port and stops it, but only when it is a Shuttrr engine
/// (`shutter serve` or `shutterd`); an unrelated process on the port is never touched.
enum EngineStopper {
    static func port(of base: String) -> Int {
        URLComponents(string: base)?.port ?? 8484
    }

    private static func run(_ path: String, _ args: [String]) -> String {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: path)
        p.arguments = args
        let out = Pipe()
        p.standardOutput = out
        p.standardError = FileHandle.nullDevice
        do { try p.run() } catch { return "" }
        p.waitUntilExit()
        return String(decoding: out.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
    }

    static func isShuttrrEngine(_ command: String) -> Bool {
        let c = command.lowercased()
        return (c.contains("shutter serve") || c.contains("shutter.cli serve") || c.contains("shutterd"))
            && (c.contains("/.shutter/") || c.contains("shuttrr.app") || c.contains("shutterd") || c.contains("/shutter/"))
    }

    static func listeners(port: Int) -> [pid_t] {
        run("/usr/sbin/lsof", ["-nP", "-iTCP:\(port)", "-sTCP:LISTEN", "-t"])
            .split(whereSeparator: \.isNewline).compactMap { pid_t(String($0).trimmingCharacters(in: .whitespaces)) }
    }

    static func command(of pid: pid_t) -> String {
        run("/bin/ps", ["-o", "command=", "-p", String(pid)]).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func stopShuttrrEngine(onPortOf base: String, seconds: Double) {
        for pid in listeners(port: port(of: base)) where pid != getpid() && isShuttrrEngine(command(of: pid)) {
            kill(pid, SIGTERM)
            let deadline = Date().addingTimeInterval(seconds)
            while Date() < deadline, kill(pid, 0) == 0 { usleep(100_000) }
            if kill(pid, 0) == 0 { kill(pid, SIGKILL) }
        }
    }
}

/// Opens the first-run window from places that have no SwiftUI environment (the app delegate). A view that
/// can open windows registers its `openWindow` action here when it appears.
@MainActor
final class OnboardingOpener {
    static let shared = OnboardingOpener()
    var openWindow: ((String) -> Void)?
    private var pending: [String] = []

    func open(_ id: String = OnboardingFlow.windowID) {
        if let openWindow {
            WindowFront.show(id, open: openWindow)
        } else if !pending.contains(id) {
            pending.append(id)
        }
    }

    func register(_ action: @escaping (String) -> Void) {
        openWindow = action
        let ids = pending
        pending = []
        ids.forEach { open($0) }
    }
}

/// Brings one of the app's windows to the front from the menu bar panel, and dismisses the panel.
/// The panel is SwiftUI's MenuBarExtra in window style: an NSPanel that only goes away by itself when it loses key
/// status, which an LSUIElement app opening a window does not reliably cause. So it is closed explicitly.
@MainActor
enum WindowFront {
    /// True for the MenuBarExtra panel (SwiftUI's private window class) and not for the app's own windows.
    static func isMenuBarPanel(_ w: NSWindow) -> Bool {
        let cls = String(describing: type(of: w))
        return cls.contains("MenuBarExtra") || (w is NSPanel && (w.identifier?.rawValue ?? "").isEmpty
            && w.styleMask.contains(.nonactivatingPanel))
    }

    static func closeMenuBarPanel() {
        for w in NSApp.windows where w.isVisible && isMenuBarPanel(w) {
            w.orderOut(nil)
            w.close()
        }
    }

    static func window(_ id: String) -> NSWindow? {
        NSApp.windows.first { ($0.identifier?.rawValue ?? "").hasPrefix(id) && !isMenuBarPanel($0) }
    }

    /// Close the panel, open (or reuse) the window `id`, un-minimise it and put the app above other apps.
    static func show(_ id: String, open: @escaping (String) -> Void) {
        closeMenuBarPanel()
        open(id)
        NSApp.activate(ignoringOtherApps: true)
        func front() {
            guard let w = window(id) else { return }
            if w.isMiniaturized { w.deminiaturize(nil) }
            w.makeKeyAndOrderFront(nil)
            w.orderFrontRegardless()
        }
        front()
        // openWindow creates the window on the next run loop pass; make sure it ends up in front and the panel stays gone.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
            closeMenuBarPanel()
            front()
            NSApp.activate(ignoringOtherApps: true)
        }
    }
}

/// The menu bar icon. Template symbol, so it follows the menu bar's appearance.
struct MenuBarIcon: View {
    @ObservedObject var store: EngineStore
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Image(systemName: store.menuBarSymbol)
            .accessibilityLabel("Shuttrr")
            .onAppear {
                let action = openWindow
                OnboardingOpener.shared.register { action(id: $0) }
            }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, ObservableObject {
    static let dashboardID = "dashboard"

    let store = EngineStore()
    let engine = EngineProcess()
    let setup = SetupRunner()
    let flow = OnboardingFlow()

    func applicationDidFinishLaunching(_ notification: Notification) {
        // A Shuttrr home set in the app's settings (instead of the SHUTTER_HOME variable) is passed on to every
        // engine command the app runs.
        if ProcessInfo.processInfo.environment["SHUTTER_HOME"] == nil,
           let d = UserDefaults.standard.string(forKey: "shutterHome"), !d.isEmpty {
            setenv("SHUTTER_HOME", (d as NSString).expandingTildeInPath, 1)
        }
        // A regular app: Dock icon and app switcher, as well as the menu bar icon.
        NSApp.setActivationPolicy(.regular)
        store.start()
        UpdateChecker.shared.start()      // Sparkle: checks now and daily, only when "Check for updates" is on
        let needsOnboarding = SetupRunner.needsOnboarding
            || ProcessInfo.processInfo.environment["SHUTTER_ONBOARDING_PAGE"] != nil
        // SwiftUI may open the Window scenes at launch; Shutter lives in the menu bar, so close the dashboard
        // until the user asks for it. The first-run window stays (and comes to the front) only when needed.
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 200_000_000)
            var shown = false
            var dashboard = false
            for window in NSApplication.shared.windows {
                guard let id = window.identifier?.rawValue else { continue }
                if id.hasPrefix(AppDelegate.dashboardID) {
                    if needsOnboarding { window.close() } else { window.makeKeyAndOrderFront(nil); dashboard = true }
                } else if id.hasPrefix(OnboardingFlow.windowID) {
                    if needsOnboarding {
                        window.center()
                        window.makeKeyAndOrderFront(nil)
                        window.orderFrontRegardless()
                        shown = true
                    } else {
                        window.close()
                    }
                }
            }
            if needsOnboarding {
                if !shown { OnboardingOpener.shared.open() }
                NSApp.activate(ignoringOtherApps: true)
            } else if !dashboard {
                OnboardingOpener.shared.open(AppDelegate.dashboardID)      // a real app opens on its main window
            }
        }
        // Already set up: make sure an engine is running (unless something already answers on its port, or the user
        // switched "Start with Shuttrr" off on the Server page).
        UserDefaults.standard.register(defaults: [EngineProcess.autoStartKey: true])
        let autoStart = !needsOnboarding && UserDefaults.standard.bool(forKey: EngineProcess.autoStartKey)
        // After an app update the bundled engine is newer than ~/.shutter/engine: upgrade it first (models,
        // settings and outputs stay), then start it as usual. A failed upgrade keeps the old engine.
        if !needsOnboarding && EngineUpgrader.needsUpgrade {
            Task { @MainActor in
                // An engine from the previous version may still be on the port (started by hand): stop it first.
                EngineStopper.stopShuttrrEngine(onPortOf: APIClient.current().base, seconds: 5)
                EngineUpgrader.shared.run { [weak self] _ in
                    guard let self, autoStart, !self.engine.isRunning else { return }
                    self.engine.start()
                    self.store.refreshNow()
                }
            }
        } else if autoStart {
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: 3_000_000_000)
                if !store.connection.isConnected, !(await EngineProbe.answers()), !engine.isRunning {
                    engine.start()
                    store.refreshNow()
                }
            }
        }
    }

    /// Closing the last window keeps Shuttrr running (menu bar icon, engine).
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    /// A click on the Dock icon brings the main window forward, or opens it when none is visible.
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        let visible = NSApp.windows.contains { $0.isVisible && !WindowFront.isMenuBarPanel($0)
            && ($0.identifier?.rawValue ?? "").isEmpty == false }
        if !visible {
            if let open = OnboardingOpener.shared.openWindow {
                WindowFront.show(AppDelegate.dashboardID, open: open)
            }
            return false
        }
        return true
    }

    /// Quitting Shuttrr stops its engine too: the one this app started, and a Shuttrr engine left on the app's port
    /// by a previous run or started by hand. A running request asks first.
    /// The question is the app's own sheet in the main window (QuitConfirm), answered with terminateLater.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        let confirm = QuitConfirm.shared
        if store.hasActiveJob && !confirm.skipNext {
            guard !confirm.asking else { return .terminateCancel }
            confirm.answer = { [weak self] quit in
                if quit { self?.stopEngines() }
                NSApp.reply(toApplicationShouldTerminate: quit)
            }
            confirm.asking = true
            NSApp.activate(ignoringOtherApps: true)
            OnboardingOpener.shared.open(AppDelegate.dashboardID)
            return .terminateLater
        }
        stopEngines()
        return .terminateNow
    }

    private func stopEngines() {
        engine.stopAndWait(seconds: 5)
        EngineStopper.stopShuttrrEngine(onPortOf: APIClient.current().base, seconds: 5)
    }

    func applicationWillTerminate(_ notification: Notification) {
        store.stop()
    }
}
