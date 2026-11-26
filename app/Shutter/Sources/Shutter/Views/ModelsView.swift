import AppKit
import SwiftUI

/// Models: what is loaded in memory (Unload, Unload all) and what the registry offers.
struct ModelsView: View {
    @EnvironmentObject private var store: EngineStore
    @EnvironmentObject private var flow: OnboardingFlow
    @Environment(\.openWindow) private var openWindow
    let onNewJob: (String?) -> Void

    init(onNewJob: @escaping (String?) -> Void) {
        self.onNewJob = onNewJob
    }

    var body: some View {
        let entries = store.system?.pool.entries ?? []
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.s5) {
                HStack(alignment: .center) {
                    PageTitle(title: "Models", subtitle: subtitle(entries))
                    Spacer()
                    Button("Set-up…") {
                        flow.reset()
                        openWindow(id: OnboardingFlow.windowID)
                        NSApp.activate(ignoringOtherApps: true)
                    }
                    .buttonStyle(.kit(.line))
                    .help("Open the first-run window: install or repair the engine and choose a model")
                    Button("Reload") { store.reloadModels() }
                        .buttonStyle(.kit(.line))
                        .disabled(!store.connection.isConnected)
                }

                if store.needsFirstModel {
                    SetupCard()
                        .sectionTransition()
                }

                VStack(alignment: .leading, spacing: Theme.Space.s3) {
                    SectionHeader(title: "In memory", count: entries.count) {
                        Button("Unload All") { store.release(key: nil) }
                            .buttonStyle(.kit(.line, small: true))
                            .disabled(entries.isEmpty)
                    }
                    if entries.isEmpty {
                        EmptyStateView(title: "Nothing loaded",
                                       message: "A model loads when a request needs it and unloads after it sits idle, so memory goes back to your other apps.")
                    } else {
                        VStack(spacing: 0) {
                            header(["Component", "Size", "Idle", "State"])
                            ForEach(entries) { entry in
                                LoadedRow(entry: entry) { store.release(key: entry.key) }
                            }
                        }
                    }
                    Text(gpuLine)
                        .font(Theme.Fonts.meta)
                        .foregroundStyle(Theme.Palette.muted)
                }

                if store.sampleModel != nil {
                    TrySampleCard()
                }

                VStack(alignment: .leading, spacing: Theme.Space.s3) {
                    SectionHeader(title: "Available", count: store.models.count)
                    if store.models.isEmpty {
                        EmptyStateView(title: store.connection.isConnected ? "No models configured" : "Engine not connected",
                                       message: "Download a model above, or add your own to config.json in the Shuttrr home folder and restart the engine.")
                    } else {
                        VStack(spacing: 0) {
                            AvailableHeader()
                            ForEach(store.models) { model in
                                AvailableRow(model: model) { onNewJob(model.id) }
                            }
                        }
                    }
                }

                if !store.needsFirstModel && !moreToDownload.isEmpty {
                    VStack(alignment: .leading, spacing: Theme.Space.s3) {
                        SectionHeader(title: "Download a model")
                        SectionedModelList(models: moreToDownload)
                    }
                }
            }
            .animation(Motion.smooth, value: store.needsFirstModel)
            .padding(.horizontal, Theme.Space.s6)
            .padding(.top, Theme.Space.s5)
            .padding(.bottom, Theme.Space.s6)
        }
    }

    /// Catalog models that are not installed yet, or that still wait for an engine restart.
    private var moreToDownload: [CatalogModel] {
        store.catalog.filter { !$0.installed || store.catalogRestartNeeded }
    }

    private func subtitle(_ entries: [PoolEntry]) -> String {
        guard let sys = store.system else { return "" }
        if entries.isEmpty { return "Nothing loaded" }
        return "\(entries.count) component\(entries.count == 1 ? "" : "s") loaded · \(Fmt.gb(sys.pool.totalBytes)) in memory"
    }

    private var gpuLine: String {
        guard let g = store.system?.gpuLock else { return " " }
        guard g.held else { return "GPU lock is free." }
        if let h = g.holder, !h.label.isEmpty { return "GPU lock held by \(h.label) (pid \(h.pid))." }
        return "GPU lock held."
    }

    private func header(_ titles: [String]) -> some View {
        HStack(spacing: 0) {
            Text(titles[0]).modelColumn(nil)
            Text(titles[1]).modelColumn(ModelColumns.size, .trailing)
            Text(titles[2]).modelColumn(ModelColumns.idle, .trailing)
            Text(titles[3]).modelColumn(ModelColumns.state)
            Color.clear.frame(height: 1).modelColumn(ModelColumns.action)
        }
        .font(Theme.Fonts.metaMedium)
        .foregroundStyle(Theme.Palette.muted)
        .frame(height: Theme.Metric.tableHeaderHeight)
        .overlay(alignment: .bottom) { Hairline() }
    }
}

private enum ModelColumns {
    static let size: CGFloat = Theme.Metric.colPeak
    static let idle: CGFloat = Theme.Metric.colTime
    static let state: CGFloat = Theme.Metric.colState
    static let family: CGFloat = Theme.Metric.colFamily
    static let kind: CGFloat = Theme.Metric.colTime
    static let action: CGFloat = Theme.Metric.colModelActions
    static let loadActions: CGFloat = 196
}

private extension View {
    func modelColumn(_ width: CGFloat?, _ alignment: Alignment = .leading) -> some View {
        Group {
            if let width {
                self.frame(width: width, alignment: alignment)
            } else {
                self.frame(maxWidth: .infinity, alignment: alignment)
            }
        }
        .padding(.horizontal, Theme.Space.s3)
    }

    func modelRow() -> some View {
        self
            .frame(height: Theme.Metric.rowHeight)
            .overlay(alignment: .bottom) { Hairline() }
    }
}

private struct LoadedRow: View {
    let entry: PoolEntry
    let onUnload: () -> Void

    var body: some View {
        HStack(spacing: 0) {
            Text(middleTruncated(entry.key, max: 70))
                .font(Theme.Fonts.rowMedium)
                .foregroundStyle(Theme.Palette.ink)
                .lineLimit(1)
                .textSelection(.enabled)
                .modelColumn(nil)
            Text(Fmt.gb(entry.sizeBytes))
                .monospacedDigit()
                .foregroundStyle(Theme.Palette.ink2)
                .modelColumn(ModelColumns.size, .trailing)
            Text(Clock.short(entry.idleS))
                .monospacedDigit()
                .foregroundStyle(Theme.Palette.ink2)
                .modelColumn(ModelColumns.idle, .trailing)
            KitBadge(text: entry.pinned ? "Pinned" : "Idle")
                .help(entry.pinned ? "Pinned: kept loaded" : "Unloads by itself after a while")
                .modelColumn(ModelColumns.state)
            Button("Unload", action: onUnload)
                .buttonStyle(.kit(.line, small: true))
                .modelColumn(ModelColumns.action, .trailing)
        }
        .font(Theme.Fonts.row)
        .modelRow()
    }
}

private struct AvailableHeader: View {
    var body: some View {
        HStack(spacing: 0) {
            Text("Model").modelColumn(nil)
            Text("Family").modelColumn(ModelColumns.family)
            Text("Kind").modelColumn(ModelColumns.kind)
            Color.clear.frame(height: 1).modelColumn(ModelColumns.loadActions)
        }
        .font(Theme.Fonts.metaMedium)
        .foregroundStyle(Theme.Palette.muted)
        .frame(height: Theme.Metric.tableHeaderHeight)
        .overlay(alignment: .bottom) { Hairline() }
    }
}

/// One model: name with its state under it, then one clear button (Load, Loading, Unload, or Cancel while it waits
/// for memory) and a ⋯ menu with View in Finder and Test.
private struct AvailableRow: View {
    @EnvironmentObject private var store: EngineStore
    let model: ModelInfo
    let onNewJob: () -> Void

    private var pending: Bool { store.modelPending.contains(model.id) }
    private var folderExists: Bool { !model.dir.isEmpty && FileManager.default.fileExists(atPath: model.dir) }

    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.displayName)
                    .font(Theme.Fonts.rowMedium)
                    .foregroundStyle(Theme.Palette.ink)
                    .lineLimit(1)
                    .truncationMode(.middle)
                status
            }
            .padding(.vertical, Theme.Space.s2)
            .modelColumn(nil)
            Text(model.family).foregroundStyle(Theme.Palette.ink).lineLimit(1).modelColumn(ModelColumns.family)
            Text(model.kind.isEmpty ? "–" : model.kind.capitalized).foregroundStyle(Theme.Palette.ink)
                .modelColumn(ModelColumns.kind)
            HStack(spacing: Theme.Space.s2) {
                primary
                more
            }
            .modelColumn(ModelColumns.loadActions, .trailing)
        }
        .font(Theme.Fonts.row)
        .frame(minHeight: Theme.Metric.rowHeight)
        .overlay(alignment: .bottom) { Hairline() }
        .animation(Motion.smooth, value: model.state)
    }

    @ViewBuilder private var status: some View {
        if !model.installed {
            statusLine("Not installed", color: Theme.Palette.muted)
        } else if model.isLoaded && pending {
            statusLine("Unloading…", color: Theme.Palette.ink2)
        } else if model.isLoaded {
            HStack(spacing: Theme.Space.s1) {
                Circle().fill(Theme.Palette.good).frame(width: 6, height: 6)
                statusLine(loadedText, color: Theme.Palette.ink2)
            }
        } else if model.isLoading {
            statusLine("Loading into memory…", color: Theme.Palette.ink2)
        } else if model.isWaiting {
            statusLine(model.waiting.isEmpty ? "Waiting for memory" : "Waiting for memory. \(model.waiting)",
                       color: Theme.Palette.warn)
                .fixedSize(horizontal: false, vertical: true)
                .lineLimit(3)
        } else if model.supportsLoad && !model.canLoad {
            statusLine(model.loadHint.isEmpty ? "Loads when a request needs it" : model.loadHint,
                       color: Theme.Palette.muted)
        } else if model.supportsLoad {
            statusLine("Not loaded", color: Theme.Palette.muted)
        } else {
            statusLine("Loads when a request needs it", color: Theme.Palette.muted)
        }
    }

    /// "Loaded (1.4 GB)", or "Loaded (streams from SSD) · 0.2 GB" when most of the model stays on the SSD.
    private var loadedText: String {
        let gb = model.loadedBytes > 0 ? Fmt.gb(model.loadedBytes) : ""
        if model.isStreamed { return gb.isEmpty ? "Loaded (streams from SSD)" : "Loaded (streams from SSD) · \(gb)" }
        return gb.isEmpty ? "Loaded" : "Loaded (\(gb))"
    }

    private func statusLine(_ text: String, color: Color) -> some View {
        Text(text)
            .font(Theme.Fonts.meta)
            .foregroundStyle(color)
            .lineLimit(1)
            .help(text)
    }

    @ViewBuilder private var primary: some View {
        if !model.installed || !model.supportsLoad {
            EmptyView()
        } else if model.isLoading {
            HStack(spacing: Theme.Space.s2) {
                ProgressView().controlSize(.small)
                Text("Loading")
                    .font(Theme.Fonts.button)
                    .foregroundStyle(Theme.Palette.ink2)
            }
            .frame(height: Theme.Metric.buttonHeight)
        } else if model.isWaiting {
            HStack(spacing: Theme.Space.s2) {
                ProgressView().controlSize(.small)
                Button("Cancel") { store.unloadModel(model.id) }
                    .buttonStyle(.kit(.line))
                    .disabled(pending)
                    .help("Stop waiting and do not load this model")
            }
        } else if model.isLoaded {
            Button(pending ? "Unloading…" : "Unload") { store.unloadModel(model.id) }
                .buttonStyle(.kit(.line))
                .disabled(pending)
                .help("Free the memory this model holds")
        } else if !model.canLoad {
            Button("Load") {}
                .buttonStyle(.kit(.line))
                .disabled(true)
                .help(model.loadHint.isEmpty ? "Loads when a request needs it" : model.loadHint)
        } else {
            Button(pending ? "Loading…" : "Load") { store.loadModel(model.id) }
                .buttonStyle(.kit(.ink))
                .disabled(pending || !store.connection.isConnected)
                .help("Load it now so the next request starts at once. Shuttrr waits if memory is short.")
        }
    }

    private var more: some View {
        Menu {
            Button("View in Finder") {
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: model.dir)])
            }
            .disabled(!folderExists)
            Button("Test", action: onNewJob)
                .disabled(!model.installed)
        } label: {
            Image(systemName: "ellipsis")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Theme.Palette.ink)
                .frame(width: Theme.Metric.buttonHeight, height: Theme.Metric.buttonHeight)
                .background(Circle().fill(Theme.Palette.white))
                .overlay(Circle().strokeBorder(Theme.Palette.lineStrong, lineWidth: Theme.Metric.hairline))
                .contentShape(Circle())
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .accessibilityLabel("More")
        .help("More")
    }
}
