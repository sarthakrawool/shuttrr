import AppKit
import SwiftUI

/// Server: everything about the engine Shuttrr talks to (the local inference server): whether it runs, its address,
/// the command that starts it, starting it with the app, idle unload, the addresses other tools use, and its log.
struct ServerView: View {
    @EnvironmentObject private var store: EngineStore
    @EnvironmentObject private var engine: EngineProcess

    @AppStorage(APIClient.baseURLKey) private var engineURL = APIClient.defaultBaseURL
    @AppStorage(EngineProcess.autoStartKey) private var autoStart = true
    @State private var copied: String?
    @State private var restarting = false
    @State private var showLog = false
    @State private var diagnostics: DiagState = .idle

    enum DiagState { case idle, working, copied }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.s5) {
                PageTitle(title: "Server",
                          subtitle: "The engine that runs the models on this Mac. Other apps send it requests at the address below.")

                section("Status") {
                    KVGroup {
                        KVRow(label: "Engine", first: true) {
                            HStack(spacing: Theme.Space.s3) {
                                Circle().fill(statusColor).frame(width: 8, height: 8)
                                Text(engineState)
                                    .foregroundStyle(Theme.Palette.ink)
                                    .lineLimit(2)
                                    .fixedSize(horizontal: false, vertical: true)
                                Spacer(minLength: Theme.Space.s3)
                                controls
                            }
                        }
                        KVRow(label: "Start with Shuttrr") {
                            HStack {
                                Text("Start the engine when Shuttrr opens, if none is running")
                                    .foregroundStyle(Theme.Palette.ink2)
                                    .fixedSize(horizontal: false, vertical: true)
                                Spacer(minLength: Theme.Space.s3)
                                Toggle("", isOn: $autoStart)
                                    .labelsHidden()
                                    .toggleStyle(.switch)
                                    .tint(Theme.Palette.violet)
                            }
                        }
                        KVRow(label: "Idle unload") {
                            Text(idleText)
                                .foregroundStyle(Theme.Palette.ink2)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                    hint("Shuttrr only stops or restarts an engine it started itself. Models you load with Load stay in memory until you unload them.")
                }

                section("Help") {
                    KVGroup {
                        KVRow(label: "Diagnostics", first: true) {
                            HStack {
                                Text("Copies the engine check, this Mac's model, chip, memory, macOS, versions and the last 100 log lines, ready to paste into a bug report. Your home folder shows as ~.")
                                    .foregroundStyle(Theme.Palette.ink2)
                                    .fixedSize(horizontal: false, vertical: true)
                                Spacer(minLength: Theme.Space.s3)
                                Button(diagnosticsTitle) { copyDiagnostics() }
                                    .buttonStyle(.kit(.line, small: true))
                                    .disabled(diagnostics == .working)
                            }
                        }
                        KVRow(label: "Report a problem") {
                            HStack {
                                Text("Opens the issue form on GitHub. Paste the diagnostics there.")
                                    .foregroundStyle(Theme.Palette.ink2)
                                    .fixedSize(horizontal: false, vertical: true)
                                Spacer(minLength: Theme.Space.s3)
                                Link(destination: Releases.newIssue) {
                                    Text("Open GitHub Issues")
                                }
                                .buttonStyle(.kit(.line, small: true))
                            }
                        }
                    }
                }

                section("Address and command") {
                    KVGroup {
                        KVRow(label: "Address", first: true) {
                            TextField("", text: $engineURL, prompt: Text(APIClient.defaultBaseURL))
                                .kitField()
                                .onSubmit { store.refreshNow() }
                        }
                        KVRow(label: "Engine command") {
                            Text(engine.command)
                                .font(Theme.Fonts.mono)
                                .foregroundStyle(Theme.Palette.ink2)
                                .lineLimit(1)
                                .truncationMode(.middle)
                                .textSelection(.enabled)
                        }
                        KVRow(label: "Default address") {
                            HStack {
                                Text(APIClient.defaultBaseURL)
                                    .font(Theme.Fonts.mono)
                                    .foregroundStyle(Theme.Palette.muted)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                                Spacer(minLength: Theme.Space.s3)
                                Button("Reset to Default") {
                                    engineURL = APIClient.defaultBaseURL
                                    store.refreshNow()
                                }
                                .buttonStyle(.kit(.line, small: true))
                            }
                        }
                    }
                    hint("Shuttrr runs the Python engine (shutter serve) in a login shell when it starts the engine.")
                }

                section("For other apps") {
                    KVGroup {
                        copyRow("Shuttrr API", APIClient.current().base, first: true)
                        copyRow("OpenAI-compatible images", APIClient.current().base + "/v1")
                        copyRow("ComfyUI nodes", APIClient.current().base)
                    }
                    hint("ComfyUI, scripts and other tools on this Mac use these addresses. Nothing leaves your computer.")
                }

                if let sys = store.system {
                    section("Running engine") {
                        KVGroup {
                            KVRow("Version", sys.version.isEmpty ? "–" : sys.version, first: true)
                            KVRow("Memory reserve", "\(Fmt.gb(sys.guardInfo.headroomBytes)) kept free" + (sys.guardInfo.headroomAuto ? " (automatic)" : ""))
                            KVRow("Protected services", sys.guardInfo.protected.isEmpty ? "None" : sys.guardInfo.protected.map(\.name).joined(separator: ", "))
                            KVRow("GPU lock", gpuText(sys.gpuLock))
                        }
                        hint("Engine limits come from config.json in the Shuttrr home folder; restart the engine after changing it.")
                    }
                }

                section("Engine log (recent)") {
                    RecentLogBox(fallback: engine.logLines)
                        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous))
                        .overlay(RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous)
                            .strokeBorder(Theme.Palette.line, lineWidth: Theme.Metric.hairline))
                    HStack(spacing: Theme.Space.s3) {
                        Text(EngineProcess.logURL.path)
                            .font(Theme.Fonts.mono)
                            .foregroundStyle(Theme.Palette.muted)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .textSelection(.enabled)
                        Spacer(minLength: Theme.Space.s3)
                        Button("View All") { showLog = true }
                            .buttonStyle(.kit(.line, small: true))
                            .disabled(!FileManager.default.fileExists(atPath: EngineProcess.logURL.path))
                        Button("View in Finder") {
                            NSWorkspace.shared.activateFileViewerSelecting([EngineProcess.logURL])
                        }
                        .buttonStyle(.kit(.line, small: true))
                        .disabled(!FileManager.default.fileExists(atPath: EngineProcess.logURL.path))
                    }
                }
            }
            .padding(.horizontal, Theme.Space.s6)
            .padding(.top, Theme.Space.s5)
            .padding(.bottom, Theme.Space.s6)
        }
        .sheet(isPresented: $showLog) { FullLogSheet() }
    }

    private var diagnosticsTitle: String {
        switch diagnostics {
        case .idle: return "Copy Diagnostics"
        case .working: return "Collecting…"
        case .copied: return "Copied"
        }
    }

    private func copyDiagnostics() {
        diagnostics = .working
        Task { @MainActor in
            let text = await Diagnostics.collect(engineVersion: store.system?.version)
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
            diagnostics = .copied
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            if diagnostics == .copied { diagnostics = .idle }
        }
    }

    @ViewBuilder private var controls: some View {
        if engine.isRunning {
            Button(restarting ? "Restarting…" : "Restart") { restart() }
                .buttonStyle(.kit(.line, small: true))
                .disabled(restarting || store.hasActiveJob)
                .help(store.hasActiveJob ? "Wait until the running request is done" : "Stop and start the engine")
            Button("Stop") { engine.stop() }
                .buttonStyle(.kit(.line, small: true))
                .disabled(restarting)
        } else {
            Button("Start") {
                engine.start()
                store.refreshNow()
            }
            .buttonStyle(.kit(.ink, small: true))
            .disabled(store.connection.isConnected)
            .help(store.connection.isConnected ? "An engine started outside Shuttrr is already running" : "Start the engine")
        }
    }

    private var statusColor: Color {
        switch store.connection {
        case .connected: return Theme.Palette.good
        case .connecting: return Theme.Palette.warn
        case .disconnected: return Theme.Palette.danger
        }
    }

    private var engineState: String {
        switch store.connection {
        case .connected: return engine.isRunning ? "Running (started by Shuttrr)" : "Running (started outside Shuttrr)"
        case .connecting: return "Connecting…"
        case .disconnected(let why): return "Stopped. \(why)"
        }
    }

    /// idle_unload_s from config.json (default 15 minutes); the engine reads it when it starts.
    private var idleText: String {
        var secs = 900.0
        let url = SetupRunner.shutterHome.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: url),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let v = obj["idle_unload_s"] as? Double {
            secs = v
        }
        if secs <= 0 { return "Models unload as soon as a request is done (idle_unload_s in config.json)." }
        let mins = Int((secs / 60).rounded())
        return "Models a request loaded unload after \(mins) minute\(mins == 1 ? "" : "s") without use (idle_unload_s in config.json)."
    }

    private func copyRow(_ label: String, _ value: String, first: Bool = false) -> some View {
        KVRow(label: label, first: first) {
            HStack {
                Text(value)
                    .font(Theme.Fonts.mono)
                    .foregroundStyle(Theme.Palette.ink)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                Spacer(minLength: Theme.Space.s3)
                Button(copied == label ? "Copied" : "Copy") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(value, forType: .string)
                    copied = label
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { if copied == label { copied = nil } }
                }
                .buttonStyle(.kit(.line, small: true))
            }
        }
    }

    private func restart() {
        restarting = true
        engine.stop()
        Task { @MainActor in
            for _ in 0..<60 where engine.isRunning {
                try? await Task.sleep(nanoseconds: 250_000_000)
            }
            engine.start()
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            store.refreshNow()
            restarting = false
        }
    }

    private func gpuText(_ g: GpuLockInfo) -> String {
        guard g.held else { return "Free" }
        if let h = g.holder, !h.label.isEmpty { return "Held by \(h.label)" }
        return "Held"
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
}
