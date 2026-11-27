import AppKit
import ServiceManagement
import SwiftUI

/// Settings: how the app behaves (launch at login, keyboard). Everything about the engine server is on ServerView.
struct SettingsView: View {
    @EnvironmentObject private var store: EngineStore
    @EnvironmentObject private var engine: EngineProcess

    @State private var launchAtLogin = false
    @State private var loginError: String?
    @AppStorage(UpdateChecker.enabledKey) private var checkUpdates = true
    @State private var showUninstall = false
    @State private var keepModels = true
    @State private var uninstalling = false
    @State private var uninstallResult: (ok: Bool, message: String)?
    @State private var showTelemetry = false
    /// The owner's hidden Admin Token… button: only while Option is held (or by Option-clicking the version).
    @State private var optionHeld = false
    @State private var showAdminToken = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.s5) {
                PageTitle(title: "Settings",
                          subtitle: "How the Shuttrr app behaves. The engine itself is on the Server page; memory mode is on the Memory page.")

                section("General") {
                    KVGroup {
                        KVRow(label: "Launch at login", first: true) {
                            Toggle("", isOn: Binding(get: { launchAtLogin }, set: { setLaunchAtLogin($0) }))
                                .labelsHidden()
                                .toggleStyle(.switch)
                                .tint(Theme.Palette.violet)
                        }
                    }
                    if let loginError {
                        Text(loginError).font(Theme.Fonts.meta).foregroundStyle(Theme.Palette.danger)
                    }
                }

                section("Updates") {
                    KVGroup {
                        KVRow(label: "Check for updates automatically", first: true) {
                            HStack {
                                Text("At launch and once a day, ask GitHub for a newer Shuttrr. When there is one, Shuttrr tells you and asks before downloading. Your models and settings stay.")
                                    .foregroundStyle(Theme.Palette.ink2)
                                    .fixedSize(horizontal: false, vertical: true)
                                Spacer(minLength: Theme.Space.s3)
                                Toggle("", isOn: $checkUpdates)
                                    .labelsHidden()
                                    .toggleStyle(.switch)
                                    .tint(Theme.Palette.violet)
                                    .onChange(of: checkUpdates) { _, on in UpdateChecker.shared.enabledChanged(on) }
                            }
                        }
                        KVRow(label: "Version") {
                            HStack {
                                Text(Releases.appVersion).foregroundStyle(Theme.Palette.ink2)
                                    .onTapGesture {
                                        if NSEvent.modifierFlags.contains(.option) { showAdminToken = true }
                                    }
                                Spacer(minLength: Theme.Space.s3)
                                if optionHeld {
                                    Button("Admin Token…") { showAdminToken = true }
                                        .buttonStyle(.kit(.line, small: true))
                                }
                                Button("Check for Updates…") { UpdateChecker.shared.checkNow() }
                                    .buttonStyle(.kit(.line, small: true))
                            }
                        }
                    }
                    UpdateAvailableRow()
                }

                if let t = store.telemetry {
                    section("Privacy") {
                        KVGroup {
                            KVRow(label: "Help The Dev Improve", first: true) {
                                HStack {
                                    Text("Send anonymous usage and crash reports: which model you use the most, and what went wrong when something crashes. Never your prompts, files or name.")
                                        .foregroundStyle(Theme.Palette.ink2)
                                        .fixedSize(horizontal: false, vertical: true)
                                    Spacer(minLength: Theme.Space.s3)
                                    Toggle("", isOn: Binding(get: { t.enabled },
                                                             set: { on in Task { await store.setTelemetry(on) } }))
                                        .labelsHidden()
                                        .toggleStyle(.switch)
                                        .tint(Theme.Palette.violet)
                                        .disabled(store.telemetryBusy)
                                }
                            }
                            KVRow(label: "What is sent") {
                                HStack {
                                    Text(t.lastSentAt.map { "Last sent \(TelemetryDetailsSheet.relative($0))." }
                                         ?? "Nothing has been sent yet.")
                                        .foregroundStyle(Theme.Palette.ink2)
                                    Spacer(minLength: Theme.Space.s3)
                                    Button("See What's Sent") { showTelemetry = true }
                                        .buttonStyle(.kit(.line, small: true))
                                }
                            }
                            KVRow(label: "Update checks") {
                                Text("Not tracking: when \"Check for updates automatically\" is on, Shuttrr asks GitHub every few hours whether a newer version exists. The request carries no data about you or how you use Shuttrr, and you can switch it off above.")
                                    .foregroundStyle(Theme.Palette.ink2)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }
                    }
                }

                section("Keyboard") {
                    KVGroup {
                        KVRow(label: "New request", first: true) { keys(["⌘", "N"]) }
                        KVRow(label: "Search requests") { keys(["⌘", "F"]) }
                        KVRow(label: "Models, Requests, Memory, Server, Settings") { keys(["⌘", "1–5"]) }
                        KVRow(label: "Send the new request") { keys(["⌘", "↩"]) }
                        KVRow(label: "Close the sheet, clear search") { keys(["esc"]) }
                    }
                }

                section("Uninstall") {
                    KVGroup {
                        KVRow(label: "Remove Shuttrr's files", first: true) {
                            HStack {
                                Text("Stops the engine and deletes ~/.shutter: the engine, settings, job history and, if you choose, downloaded models.")
                                    .foregroundStyle(Theme.Palette.ink2)
                                    .fixedSize(horizontal: false, vertical: true)
                                Spacer(minLength: Theme.Space.s3)
                                Button("Uninstall Shuttrr…") { uninstallResult = nil; showUninstall = true }
                                    .buttonStyle(.kit(.line, small: true))
                            }
                        }
                    }
                }

            }
            .padding(.horizontal, Theme.Space.s6)
            .padding(.top, Theme.Space.s5)
            .padding(.bottom, Theme.Space.s6)
        }
        .onAppear {
            launchAtLogin = SMAppService.mainApp.status == .enabled
        }
        .sheet(isPresented: $showUninstall) { uninstallSheet }
        .background(EmptyView().sheet(isPresented: $showTelemetry) {
            TelemetryDetailsSheet().environmentObject(store)
        })
        .background(EmptyView().sheet(isPresented: $showAdminToken) { AdminTokenSheet() })
        .modifier(OptionKeyWatcher(held: $optionHeld))
    }

    @ViewBuilder private var uninstallSheet: some View {
        if let result = uninstallResult {
            ConfirmSheet(title: result.ok ? "Shuttrr's files are gone" : "Not everything was removed",
                         message: result.message + " To finish, quit Shuttrr and drag Shuttrr.app to the Trash.",
                         confirmTitle: "Quit Shuttrr",
                         onCancel: { showUninstall = false },
                         onConfirm: {
                             QuitConfirm.shared.skipNext = true
                             QuitConfirm.shared.quitEverything = true
                             NSApp.terminate(nil)
                         })
        } else {
            ConfirmSheet(title: "Uninstall Shuttrr?",
                         message: "This stops the engine and deletes ~/.shutter. Requests that are running stop. It cannot be undone.",
                         confirmTitle: uninstalling ? "Removing…" : "Uninstall", destructive: true, busy: uninstalling,
                         onCancel: { showUninstall = false },
                         onConfirm: { uninstall() }) {
                Toggle(isOn: $keepModels) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Keep downloaded models").font(Theme.Fonts.row).foregroundStyle(Theme.Palette.ink)
                        Text("They stay in ~/.shutter/models, so a reinstall does not download them again.")
                            .font(Theme.Fonts.meta).foregroundStyle(Theme.Palette.muted)
                    }
                }
                .toggleStyle(.checkbox)
                .disabled(uninstalling)
            }
        }
    }

    private func uninstall() {
        uninstalling = true
        Task { @MainActor in
            let r = await Uninstaller.run(keepModels: keepModels, engine: engine)
            try? await SMAppService.mainApp.unregister()
            UserDefaults.standard.removeObject(forKey: SetupRunner.onboardingDoneKey)   // a reinstall starts at the welcome window
            uninstalling = false
            uninstallResult = r
        }
    }

    private func section<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.s3) {
            SectionHeader(title: title)
            content()
        }
    }

    private func hint(_ text: String) -> some View {
        Text(text)
            .font(Theme.Fonts.meta)
            .foregroundStyle(Theme.Palette.muted)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func keys(_ k: [String]) -> some View {
        HStack(spacing: Theme.Space.s1) {
            ForEach(k, id: \.self) { KeyCap(key: $0) }
        }
    }

    private func setLaunchAtLogin(_ enabled: Bool) {
        loginError = nil
        do {
            if enabled {
                try SMAppService.mainApp.register()
            } else {
                try SMAppService.mainApp.unregister()
            }
        } catch {
            loginError = "Could not change login item: \(error.localizedDescription)"
        }
        launchAtLogin = SMAppService.mainApp.status == .enabled
    }
}

/// Last lines of the daemon's output, newest at the bottom.
struct EngineLogView: View {
    let lines: [String]

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: Theme.Space.s0) {
                    if lines.isEmpty {
                        Text("No output yet. The log fills when Shuttrr starts the engine.")
                            .font(Theme.Fonts.row)
                            .foregroundStyle(Theme.Palette.muted)
                    }
                    ForEach(Array(lines.enumerated()), id: \.offset) { item in
                        Text(item.element.isEmpty ? " " : item.element)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(item.offset)
                    }
                }
                .font(Theme.Fonts.mono)
                .foregroundStyle(Theme.Palette.ink2)
                .textSelection(.enabled)
                .padding(Theme.Space.s3)
            }
            .background(Theme.Palette.chrome)
            .onChange(of: lines) { _, newLines in
                if !newLines.isEmpty {
                    proxy.scrollTo(newLines.count - 1, anchor: .bottom)
                }
            }
        }
    }
}
