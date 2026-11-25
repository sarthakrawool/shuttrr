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
                        KVRow(label: "Check for updates", first: true) {
                            HStack {
                                Text("Once a day, ask GitHub whether a newer Shuttrr is out. Nothing is downloaded.")
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
                        KVRow("Version", Releases.appVersion)
                    }
                    UpdateAvailableRow()
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
    }

    @ViewBuilder private var uninstallSheet: some View {
        if let result = uninstallResult {
            ConfirmSheet(title: result.ok ? "Shuttrr's files are gone" : "Not everything was removed",
                         message: result.message + " To finish, quit Shuttrr and drag Shuttrr.app to the Trash.",
                         confirmTitle: "Quit Shuttrr",
                         onCancel: { showUninstall = false },
                         onConfirm: {
                             QuitConfirm.shared.skipNext = true
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
