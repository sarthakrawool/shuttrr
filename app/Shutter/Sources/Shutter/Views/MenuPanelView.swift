import AppKit
import SwiftUI

/// The panel under the menu bar icon. It is the one place that stays native glass: the system's
/// own panel material, system fonts, system colours (so it follows light and dark), laid out like
/// the Wi-Fi and Sound panels: a title with a switch, glass modules, then plain rows.
/// The main window and the web dashboard carry the brand theme; this panel deliberately does not,
/// so nothing here uses Theme's fonts or colours. Spacing and motion are shared.
struct MenuPanelView: View {
    @EnvironmentObject private var store: EngineStore
    @EnvironmentObject private var engine: EngineProcess
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s3) {
            header.rise(0)

            if let sys = store.system {
                blockingLine
                GlassMemory(system: sys).glassModule().rise(1)
                if store.settings != nil {
                    GlassMode().glassModule().rise(2)
                }
                GlassLoadedModels(pool: sys.pool).glassModule().rise(3)
                GlassRequests(queue: sys.queue).glassModule().rise(4)
            } else {
                disconnectedBody.glassModule().rise(1)
            }

            if let err = store.lastError {
                HStack(alignment: .top, spacing: Theme.Space.s2) {
                    Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
                    Text(err)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: Theme.Space.s1)
                    Button { store.lastError = nil } label: {
                        Image(systemName: "xmark").foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Dismiss")
                }
                .font(.system(size: 12))
                .padding(.horizontal, Theme.Space.s1)
                .sectionTransition()
            }

            Divider()
            actions.rise(5)
        }
        .frame(width: Theme.Metric.panelWidth - 2 * Theme.Space.s3, alignment: .leading)
        .padding(Theme.Space.s3)
        .frame(width: Theme.Metric.panelWidth, alignment: .leading)
        .animation(Motion.smooth, value: store.system != nil)
        .animation(Motion.smooth, value: store.lastError != nil)
        // No background and no forced appearance: the system's menu bar panel glass shows through.
    }

    // MARK: Header: name, state, on/off switch

    private var header: some View {
        HStack(spacing: Theme.Space.s2) {
            VStack(alignment: .leading, spacing: 0) {
                Text("Shuttrr")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.primary)
                HStack(spacing: Theme.Space.s1) {
                    PulseDot(color: statusColor, active: store.runningJob != nil)
                    Text(statusText)
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .contentTransition(.opacity)
                }
                .animation(Motion.gentle, value: statusText)
            }
            Spacer(minLength: Theme.Space.s2)
            Toggle("Engine", isOn: Binding(
                get: { engine.isRunning || store.connection.isConnected },
                set: { on in
                    if on {
                        engine.start()
                        store.refreshNow()
                    } else {
                        engine.stop()
                    }
                }))
                .toggleStyle(.switch)
                .labelsHidden()
                .disabled(store.connection.isConnected && !engine.isRunning)
                .help(store.connection.isConnected && !engine.isRunning
                      ? "This engine was started outside Shuttrr, so the app leaves it alone."
                      : "Start or stop the engine")
        }
        .padding(.horizontal, Theme.Space.s1)
        .padding(.top, Theme.Space.s0)
    }

    private var statusText: String {
        switch store.connection {
        case .connected:
            let v = store.system?.version ?? ""
            let state = store.queuePaused ? "Paused" : (store.runningJob != nil ? "Working" : "Ready")
            return v.isEmpty ? state : "\(state) · v\(v)"
        case .connecting: return "Connecting…"
        case .disconnected: return engine.isRunning ? "Starting…" : "Off"
        }
    }

    private var statusColor: Color {
        switch store.connection {
        case .connected: return store.queuePaused ? .orange : .green
        case .connecting: return Color.secondary
        case .disconnected: return engine.isRunning ? .orange : Color.secondary
        }
    }

    // MARK: Only when something holds the engine back

    @ViewBuilder
    private var blockingLine: some View {
        if let svc = store.busyProtected.first {
            panelNote(symbol: "pause.circle.fill", title: "Waiting for \(svc.name)", detail: svc.busy ?? "It is using memory Shuttrr will not compete for.")
        } else if store.queuePaused {
            panelNote(symbol: "pause.circle.fill", title: "Requests are paused", detail: "Nothing new starts until you resume.")
        } else if let sys = store.system, sys.gpuLock.held, store.runningJob == nil, let holder = sys.gpuLock.holder {
            panelNote(symbol: "lock.fill", title: "GPU in use",
                      detail: "\(holder.label.isEmpty ? "Another process" : holder.label) holds the GPU.")
        }
    }

    private func panelNote(symbol: String, title: String, detail: String) -> some View {
        HStack(alignment: .top, spacing: Theme.Space.s2) {
            Image(systemName: symbol)
                .font(.system(size: 15))
                .foregroundStyle(.orange)
            VStack(alignment: .leading, spacing: 0) {
                Text(title).font(.system(size: 13, weight: .medium)).foregroundStyle(.primary)
                Text(detail)
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(Theme.Space.s3)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Color.orange.opacity(0.14)))
        .sectionTransition()
    }

    private var disconnectedBody: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s1) {
            Text(engine.isRunning ? "Starting the engine…" : "The engine is off")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.primary)
            Text(engine.lastExit ?? "Turn it on to serve models to ComfyUI and other apps.")
                .font(.system(size: 12))
                .foregroundStyle(engine.lastExit == nil ? Color.secondary : Color.red)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: Rows

    private var actions: some View {
        VStack(spacing: 0) {
            GlassEndpointRow()
            if store.queuePauseSupported {
                GlassRow(title: store.queuePaused ? "Resume Requests" : "Pause Requests",
                         symbol: store.queuePaused ? "play" : "pause") {
                    store.setQueuePaused(!store.queuePaused)
                }
            }
            GlassRow(title: "Unload Models", symbol: "eject") { store.release(key: nil) }
                .disabled(store.system?.pool.entries.isEmpty ?? true)
            Divider().padding(.vertical, Theme.Space.s1)
            GlassRow(title: "Open Shuttrr", symbol: "macwindow", shortcut: "⌘D") {
                WindowFront.show(AppDelegate.dashboardID, open: { openWindow(id: $0) })
            }
            .keyboardShortcut("d")
            GlassRow(title: "Quit Shuttrr", symbol: "power", shortcut: "⌘Q") { NSApp.terminate(nil) }
                .keyboardShortcut("q")
        }
    }
}

// MARK: - Glass building blocks (system fonts and colours only)

private enum Glass {
    static let label = Font.system(size: 12, weight: .semibold)
    static let meta = Font.system(size: 12)
    static let row = Font.system(size: 13)
    static let rowMedium = Font.system(size: 13, weight: .medium)
    static let big = Font.system(size: 22, weight: .semibold, design: .rounded)
    static let track = Color.primary.opacity(0.08)
    /// A light veil over the panel's material, brighter in light mode, faint in dark.
    static let veil = Color.dynamic(light: 0xFFFFFF, dark: 0xFFFFFF, lightAlpha: 0.55, darkAlpha: 0.07)
    static let edge = Color.dynamic(light: 0xFFFFFF, dark: 0xFFFFFF, lightAlpha: 0.9, darkAlpha: 0.16)
}

private struct GlassModule: ViewModifier {
    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: 12, style: .continuous)
        content
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(Theme.Space.s3)
            .background(shape.fill(Glass.veil))
            .overlay(shape.strokeBorder(
                LinearGradient(colors: [Glass.edge, Color.primary.opacity(0.10)], startPoint: .top, endPoint: .bottom),
                lineWidth: 1))
            .shadow(color: Color.black.opacity(0.10), radius: 6, x: 0, y: 2)
    }
}

private extension View {
    /// A glass card: translucent veil, bright top edge, soft shadow.
    func glassModule() -> some View { modifier(GlassModule()) }
}

private struct GlassLabel: View {
    let text: String
    var body: some View {
        Text(text).font(Glass.label).foregroundStyle(.secondary)
    }
}

private struct GlassLegend: View {
    let color: Color
    let label: String
    let value: String
    var body: some View {
        HStack(spacing: Theme.Space.s1) {
            RoundedRectangle(cornerRadius: 2).fill(color).frame(width: 8, height: 8)
            Text(label).foregroundStyle(.secondary)
            Text(value).fontWeight(.medium).foregroundStyle(.primary).monospacedDigit()
        }
        .font(Glass.meta)
        .lineLimit(1)
        .fixedSize()
    }
}

/// A row like the ones in the system's Wi-Fi or Sound panels: highlights on hover.
private struct GlassRow: View {
    let title: String
    var symbol: String?
    var shortcut: String?
    let action: () -> Void

    @State private var hovering = false
    @Environment(\.isEnabled) private var isEnabled

    var body: some View {
        Button(action: action) {
            HStack(spacing: Theme.Space.s2) {
                if let symbol {
                    Image(systemName: symbol)
                        .font(Glass.row)
                        .foregroundStyle(.secondary)
                        .frame(width: 20)
                }
                Text(title).font(Glass.row).foregroundStyle(.primary)
                Spacer(minLength: Theme.Space.s3)
                if let shortcut {
                    Text(shortcut).font(Glass.row).foregroundStyle(.tertiary)
                }
            }
            .padding(.horizontal, Theme.Space.s2)
            .frame(height: 26)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(hovering && isEnabled ? Glass.track : Color.clear))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .opacity(isEnabled ? 1 : 0.4)
        .animation(Motion.quick, value: hovering)
        .onHover { hovering = $0 }
    }
}

/// The API address other apps connect to; click to copy.
private struct GlassEndpointRow: View {
    @State private var copied = false

    var body: some View {
        let address = APIClient.current().base.replacingOccurrences(of: "http://", with: "")
        GlassRow(title: copied ? "Copied" : address, symbol: "network", shortcut: copied ? nil : "Copy") {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(APIClient.current().base, forType: .string)
            copied = true
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: 1_400_000_000)
                copied = false
            }
        }
        .help("The address ComfyUI and other apps use to reach Shuttrr")
    }
}

// MARK: - Modules

/// The one big number (free for models), the used / reserve bar and its legend.
private struct GlassMemory: View {
    let system: SystemInfo

    var body: some View {
        let parts = MemoryParts(system)
        let total = CGFloat(max(parts.total, 1))
        VStack(alignment: .leading, spacing: Theme.Space.s2) {
            HStack(alignment: .firstTextBaseline, spacing: Theme.Space.s1) {
                GlassLabel(text: "Memory")
                Spacer(minLength: Theme.Space.s2)
                (Text("Swap ").foregroundColor(.secondary)
                    + Text(Fmt.gb(parts.swap)).fontWeight(.medium).foregroundColor(parts.swap > 0 ? .orange : .primary))
                    .font(Glass.meta)
                    .monospacedDigit()
                    .lineLimit(1)
            }
            HStack(alignment: .firstTextBaseline, spacing: Theme.Space.s2) {
                Text(Fmt.gb(parts.free))
                    .font(Glass.big)
                    .lineLimit(1)
                    .fixedSize()
                    .monospacedDigit()
                    .foregroundStyle(.primary)
                    .contentTransition(.numericText())
                Text("free for models")
                    .font(Glass.row)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Spacer(minLength: 0)
            }
            GeometryReader { geo in
                HStack(spacing: 0) {
                    Rectangle().fill(Color.accentColor)
                        .frame(width: geo.size.width * CGFloat(parts.used) / total)
                    Rectangle().fill(Color.accentColor.opacity(0.35))
                        .frame(width: geo.size.width * CGFloat(parts.headroom) / total)
                    Spacer(minLength: 0)
                }
                .background(Glass.track)
                .clipShape(Capsule())
                .animation(Motion.smooth, value: parts.used)
            }
            .frame(height: 8)
            .padding(.top, Theme.Space.s0)
            .accessibilityElement()
            .accessibilityLabel("Memory: used \(Fmt.gb(parts.used)), reserve \(Fmt.gb(parts.headroom)), free for models \(Fmt.gb(parts.free))")
            // The legend never decides the panel's width: it drops an item rather than overflow.
            ViewThatFits(in: .horizontal) {
                HStack(spacing: Theme.Space.s3) {
                    GlassLegend(color: .accentColor, label: "Used", value: Fmt.gb(parts.used))
                    GlassLegend(color: Color.accentColor.opacity(0.35), label: "Reserve", value: Fmt.gb(parts.headroom))
                }
                GlassLegend(color: .accentColor, label: "Used", value: Fmt.gb(parts.used))
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            if parts.pressure == "critical" {
                Label("macOS reports critical memory pressure.", systemImage: "exclamationmark.triangle.fill")
                    .font(Glass.meta)
                    .foregroundStyle(.red)
            }
        }
        .animation(Motion.smooth, value: parts.free)
    }
}

/// Fast / Low memory, with the slider that appears for Low memory.
private struct GlassMode: View {
    @EnvironmentObject private var store: EngineStore
    @State private var dragValue: Double?
    @Namespace private var thumb

    private static let modes = ["fast", "low_memory"]

    var body: some View {
        if let settings = store.settings {
            VStack(alignment: .leading, spacing: Theme.Space.s3) {
                GlassLabel(text: "Memory mode")
                HStack(spacing: 0) {
                    ForEach(Self.modes, id: \.self) { mode in
                        let on = settings.mode == mode
                        Button {
                            withAnimation(Motion.snappy) { store.setMode(mode) }
                        } label: {
                            Text(mode == "fast" ? "Fast" : "Low memory")
                                .font(on ? Glass.rowMedium : Glass.row)
                                .lineLimit(1)
                                .fixedSize()
                                .frame(maxWidth: .infinity)
                                .frame(height: 24)
                                .foregroundStyle(.primary)
                                .background {
                                    if on {
                                        RoundedRectangle(cornerRadius: 6, style: .continuous)
                                            .fill(Glass.edge)
                                            .shadow(color: Color.black.opacity(0.10), radius: 2, x: 0, y: 1)
                                            .matchedGeometryEffect(id: "thumb", in: thumb)
                                    }
                                }
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .accessibilityAddTraits(on ? .isSelected : [])
                    }
                }
                .padding(2)
                .background(RoundedRectangle(cornerRadius: 8, style: .continuous).fill(Glass.track))

                if settings.isLowMemory {
                    slider(settings).sectionTransition()
                } else {
                    Text("Weights stay loaded for speed. Steps down by itself if a model would not fit.")
                        .font(Glass.meta)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .transition(.opacity)
                }
            }
            .animation(Motion.smooth, value: settings.isLowMemory)
        }
    }

    @ViewBuilder
    private func slider(_ settings: EngineSettings) -> some View {
        let value = dragValue ?? settings.efficiency
        let rung = settings.ladder.isEmpty ? nil : settings.ladder[settings.index(for: value)]
        let n = settings.ladder.count
        VStack(alignment: .leading, spacing: Theme.Space.s2) {
            HStack(spacing: Theme.Space.s2) {
                Image(systemName: "hare").font(Glass.meta).foregroundStyle(.secondary).accessibilityHidden(true)
                Slider(value: Binding(get: { value },
                                      set: { dragValue = $0; store.setEfficiency($0, commit: false) }),
                       in: 0...1, step: n > 1 ? 1.0 / Double(n - 1) : 0.25) { editing in
                    if !editing, let v = dragValue {
                        store.setEfficiency(v)
                        dragValue = nil
                    }
                }
                .controlSize(.small)
                .accessibilityLabel("Efficiency")
                .accessibilityValue(rung?.label ?? "")
                Image(systemName: "memorychip").font(Glass.meta).foregroundStyle(.secondary).accessibilityHidden(true)
            }
            if let rung {
                HStack(alignment: .firstTextBaseline, spacing: Theme.Space.s2) {
                    Text(rung.label)
                        .font(Glass.rowMedium)
                        .foregroundStyle(.primary)
                        .contentTransition(.opacity)
                    Spacer(minLength: Theme.Space.s2)
                    if let peak = settings.peak(of: rung) {
                        Text((peak.estimate ? "about " : "") + Fmt.gb(peak.bytes))
                            .monospacedDigit()
                            .contentTransition(.numericText())
                    }
                    if let t = settings.timeRatio(of: rung) {
                        Text(String(format: "%.1f× time", t.ratio))
                            .monospacedDigit()
                            .contentTransition(.numericText())
                    }
                }
                .font(Glass.meta)
                .foregroundStyle(.secondary)
                .lineLimit(1)
            }
        }
        .animation(Motion.snappy, value: rung?.id)
    }
}

/// The models held in memory right now, each with its size.
private struct GlassLoadedModels: View {
    let pool: PoolInfo

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s2) {
            HStack(spacing: Theme.Space.s2) {
                GlassLabel(text: "In memory")
                Spacer(minLength: Theme.Space.s2)
                if !pool.entries.isEmpty {
                    Text(Fmt.gb(pool.totalBytes)).font(Glass.meta).monospacedDigit().foregroundStyle(.secondary)
                }
            }
            if pool.entries.isEmpty {
                Text("No model loaded").font(Glass.row).foregroundStyle(.secondary)
            } else {
                ForEach(pool.entries.prefix(4)) { entry in
                    HStack(spacing: Theme.Space.s2) {
                        Image(systemName: "cube").font(Glass.meta).foregroundStyle(.secondary).frame(width: 16)
                        Text(entry.key)
                            .font(Glass.row)
                            .foregroundStyle(.primary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        if entry.isStreamed {
                            Text("Runs from disk")
                                .font(.system(size: 11, weight: .medium))
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 6)
                                .frame(height: 18)
                                .background(Capsule().fill(Glass.track))
                                .fixedSize()
                                .help("Weights are read from the SSD as they are needed, not held in memory")
                        }
                        Spacer(minLength: Theme.Space.s2)
                        Text(Fmt.gb(entry.sizeBytes)).font(Glass.meta).monospacedDigit().foregroundStyle(.secondary)
                    }
                    .sectionTransition()
                }
                if pool.entries.count > 4 {
                    Text("and \(pool.entries.count - 4) more").font(Glass.meta).foregroundStyle(.secondary)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .animation(Motion.smooth, value: pool.entries.count)
    }
}

/// The request in flight and what is behind it.
private struct GlassRequests: View {
    @EnvironmentObject private var store: EngineStore
    let queue: QueueCounts

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s2) {
            HStack(spacing: Theme.Space.s2) {
                GlassLabel(text: "Requests")
                Spacer(minLength: Theme.Space.s2)
                Text(summary)
                    .font(Glass.meta)
                    .foregroundStyle(queue.failed > 0 ? Color.red : Color.secondary)
                    .lineLimit(1)
            }
            if let job = store.currentJob {
                current(job).sectionTransition()
            } else {
                Text("Idle").font(Glass.row).foregroundStyle(.secondary)
            }
            if !store.clients.isEmpty {
                Divider()
                ForEach(store.clients.prefix(3)) { client in
                    HStack(spacing: Theme.Space.s2) {
                        Image(systemName: "app.connected.to.app.below.fill")
                            .font(Glass.meta).foregroundStyle(.secondary).frame(width: 16)
                        Text(client.client)
                            .font(Glass.row)
                            .foregroundStyle(.primary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer(minLength: Theme.Space.s2)
                        Text("\(client.requests) request\(client.requests == 1 ? "" : "s")")
                            .font(Glass.meta)
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .animation(Motion.smooth, value: store.currentJob != nil)
    }

    private func current(_ job: Job) -> some View {
        let running = job.state == .running
        return VStack(alignment: .leading, spacing: Theme.Space.s2) {
            HStack(alignment: .center, spacing: Theme.Space.s2) {
                Image(systemName: running ? "bolt.fill" : "hourglass")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(running ? Color.green : Color.orange)
                    .frame(width: 16)
                VStack(alignment: .leading, spacing: 0) {
                    Text(job.request.model)
                        .font(Glass.rowMedium)
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text(subtitle(job))
                        .font(Glass.meta)
                        .foregroundStyle(.secondary)
                        .lineLimit(running ? 1 : 2)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: Theme.Space.s2)
                if running {
                    Text(Fmt.percent(job.progress))
                        .font(Glass.rowMedium)
                        .monospacedDigit()
                        .foregroundStyle(.primary)
                        .contentTransition(.numericText())
                }
                Button { store.cancel(job) } label: {
                    Image(systemName: "xmark.circle.fill").font(.system(size: 15)).foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
                .help("Cancel this request")
                .accessibilityLabel("Cancel request")
            }
            if running {
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Glass.track)
                        Capsule().fill(Color.accentColor)
                            .frame(width: geo.size.width * CGFloat(min(max(job.progress, 0), 1)))
                            .animation(Motion.smooth, value: job.progress)
                    }
                }
                .frame(height: 4)
            }
        }
    }

    private func subtitle(_ job: Job) -> String {
        var parts: [String] = [job.state == .running ? job.detailLine : "Waiting until memory is free"]
        if job.state == .running, let speed = store.system?.throughput?.speedText {
            parts.append(speed)
        } else if let mode = store.modeText(for: job) {
            parts.append(mode)
        }
        return parts.joined(separator: " · ")
    }

    /// "2 queued · 1 failed": only the counts that are not zero.
    private var summary: String {
        var parts: [String] = []
        // from the request list, which leaves out Loads from the Models page
        let queued = store.jobs.filter { $0.state == .queued }.count
        let waiting = store.jobs.filter { $0.state == .waitingMemory }.count
        if queued > 0 { parts.append("\(queued) queued") }
        if waiting > 0 { parts.append("\(waiting) waiting") }
        if queue.failed > 0 { parts.append("\(queue.failed) failed") }
        if parts.isEmpty, let t = store.system?.throughput, t.finished > 0 {
            parts.append("\(t.finished) done this hour")
        }
        return parts.joined(separator: " · ")
    }
}
