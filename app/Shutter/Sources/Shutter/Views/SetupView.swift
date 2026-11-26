import AppKit
import SwiftUI

/// First-run set-up: shown on the Models page while the only thing to run is the test model.
struct SetupCard: View {
    @EnvironmentObject private var store: EngineStore

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s4) {
            VStack(alignment: .leading, spacing: Theme.Space.s1) {
                Text("Set up Shuttrr")
                    .font(Theme.Fonts.title)
                    .foregroundStyle(Theme.Palette.ink)
                Text("Download a model to make your first video. It is saved on this Mac and nothing is uploaded.")
                    .font(Theme.Fonts.row)
                    .foregroundStyle(Theme.Palette.ink2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            SectionedModelList(models: store.catalog)
        }
        .padding(Theme.Space.s5)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous).fill(Theme.Palette.white))
        .overlay(RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous)
            .strokeBorder(Theme.Palette.lineStrong, lineWidth: Theme.Metric.hairline))
    }
}

/// Catalog models in three headed sections: Recommended, Fits but barely, Doesn't fit this Mac. Empty sections are
/// hidden; Doesn't fit starts folded, with a count.
struct SectionedModelList: View {
    let models: [CatalogModel]
    /// Testing: SHUTTER_SHOW_NOFIT=1 opens the Doesn't fit section, for screenshots.
    @State private var showNoFit = ProcessInfo.processInfo.environment["SHUTTER_SHOW_NOFIT"] != nil

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s5) {
            ForEach(FitSection.allCases) { section in
                let rows = models.filter { FitSection.of($0) == section }
                if !rows.isEmpty {
                    VStack(alignment: .leading, spacing: Theme.Space.s2) {
                        if section == .noFit {
                            Button {
                                withAnimation(Motion.smooth) { showNoFit.toggle() }
                            } label: {
                                HStack(spacing: Theme.Space.s1) {
                                    Image(systemName: "chevron.right")
                                        .font(.system(size: 10, weight: .semibold))
                                        .rotationEffect(.degrees(showNoFit ? 90 : 0))
                                    Text("\(section.title) (\(rows.count))")
                                        .font(Theme.Fonts.metaSemibold)
                                }
                                .foregroundStyle(Theme.Palette.muted)
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                        } else {
                            Text(section.title)
                                .font(Theme.Fonts.metaSemibold)
                                .foregroundStyle(section == .recommended ? Theme.Palette.ink : Theme.Palette.ink2)
                        }
                        if section != .noFit || showNoFit {
                            ForEach(rows) { ModelDownloadRow(model: $0) }
                        }
                    }
                }
            }
        }
    }
}

/// One downloadable model, at most three short lines under the name: what it is, the download size, and (only when
/// it does not sit comfortably) why. Everything else is behind Details.
struct ModelDownloadRow: View {
    @EnvironmentObject private var store: EngineStore
    @EnvironmentObject private var engine: EngineProcess
    let model: CatalogModel
    @State private var showDetails = false

    private var pending: Bool { store.catalogPending.contains(model.id) }
    private var state: String { model.download?.state ?? "" }
    private var remaining: Int { max(0, model.sizeBytes - (model.download?.doneBytes ?? 0)) }
    private var diskShort: Bool {
        guard !model.installed && !model.isDownloading && !model.fit.isNoFit else { return false }
        if !model.disk.label.isEmpty { return !model.disk.ok }
        return model.freeBytes > 0 && model.freeBytes < remaining
    }
    private var noFit: Bool { model.fit.isNoFit }

    var body: some View {
        HStack(alignment: .top, spacing: Theme.Space.s4) {
            VStack(alignment: .leading, spacing: Theme.Space.s1) {
                HStack(spacing: Theme.Space.s2) {
                    Text(model.name)
                        .font(Theme.Fonts.rowMedium)
                        .foregroundStyle(Theme.Palette.ink)
                        .lineLimit(1)
                        .truncationMode(.tail)
                        .help(model.name)
                    detailsButton
                }
                Text(model.summary)
                    .font(Theme.Fonts.row)
                    .foregroundStyle(Theme.Palette.ink2)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                    .help(model.summary)
                meta
                status
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            action
        }
        .padding(.horizontal, Theme.Space.s5)
        .padding(.vertical, Theme.Space.s3)
        .background(RoundedRectangle(cornerRadius: Theme.Radius.control + 2, style: .continuous)
            .fill(Theme.Palette.white.opacity(noFit ? 0.35 : 0.8)))
        .overlay(RoundedRectangle(cornerRadius: Theme.Radius.control + 2, style: .continuous)
            .strokeBorder(Theme.Palette.line, lineWidth: Theme.Metric.hairline))
        .opacity(noFit && !model.installed ? 0.6 : 1)
        .animation(Motion.smooth, value: state)
    }

    /// "7.3 GB download", or the disk warning.
    @ViewBuilder private var meta: some View {
        if diskShort {
            line("Not enough free disk", danger: true)
                .help(model.disk.label)
        } else if !model.installed && !model.isDownloading {
            Text("\(Fmt.gb(model.sizeBytes)) download")
                .font(Theme.Fonts.meta)
                .foregroundStyle(Theme.Palette.muted)
                .lineLimit(1)
        }
    }

    @ViewBuilder private var status: some View {
        if model.installed && store.catalogRestartNeeded {
            line("Downloaded. Restart the engine to use it.")
        } else if model.installed {
            line("Installed and ready.")
        } else if let d = model.download, d.isRunning {
            ProgressLine(value: d.fraction, width: nil, height: Theme.Metric.progressHeightWide)
                .padding(.vertical, Theme.Space.s1)
            line("\(Fmt.gb(d.doneBytes)) of \(Fmt.gb(d.totalBytes > 0 ? d.totalBytes : model.sizeBytes)) · \(Fmt.percent(d.fraction))")
        } else if state == "failed" {
            line((model.download?.error ?? "").isEmpty ? "The download failed." : model.download?.error ?? "",
                 danger: true)
        } else if state == "stopped" || state == "cancelled" {
            line("Stopped at \(Fmt.gb(model.download?.doneBytes ?? 0)). Resume carries on.")
        } else if !model.fit.isRecommended && !model.fit.reason.isEmpty {
            line(model.fit.reason, danger: noFit)
        }
    }

    private var detailsButton: some View {
        Button { showDetails.toggle() } label: {
            Image(systemName: "info.circle")
                .font(.system(size: 12))
                .foregroundStyle(Theme.Palette.muted)
        }
        .buttonStyle(.plain)
        .help("Details")
        .popover(isPresented: $showDetails, arrowEdge: .bottom) {
            VStack(alignment: .leading, spacing: Theme.Space.s2) {
                Text(model.name).font(Theme.Fonts.rowMedium)
                ForEach(detailLines, id: \.self) { t in
                    Text(t)
                        .font(Theme.Fonts.meta)
                        .foregroundStyle(Theme.Palette.ink2)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(Theme.Space.s4)
            .frame(width: 340, alignment: .leading)
        }
    }

    private var detailLines: [String] {
        var out: [String] = []
        if !model.fit.reason.isEmpty { out.append(model.fit.reason) }
        if !model.makes.isEmpty { out.append(model.makes + ".") }
        if !model.memory.isEmpty { out.append(model.memory) }
        out.append("\(Fmt.gb(model.sizeBytes)) download from Hugging Face (disk space, not memory). "
                   + "\(Fmt.gb(model.freeBytes)) free on this Mac.")
        out.append(contentsOf: model.notes)
        return out
    }

    @ViewBuilder private var action: some View {
        if model.installed && store.catalogRestartNeeded {
            if engine.isRunning && !store.hasActiveJob {
                Button("Restart Engine") { restartEngine() }
                    .buttonStyle(.kit(.ink))
            }
        } else if model.installed {
            Button("View in Finder") {
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: model.dir)])
            }
            .buttonStyle(.kit(.line))
        } else if model.isDownloading {
            Button("Stop") { store.stopDownload(model.id) }
                .buttonStyle(.kit(.line))
                .disabled(pending)
        } else if !noFit {
            Button(pending ? "Starting…" : (state.isEmpty || state == "done" ? "Download" : "Resume")) {
                store.startDownload(model.id)
            }
            .buttonStyle(.kit(.ink))
            .disabled(pending || diskShort || !store.connection.isConnected)
        }
    }

    private func line(_ text: String, danger: Bool = false) -> some View {
        Text(text)
            .font(Theme.Fonts.meta)
            .foregroundStyle(danger ? Theme.Palette.danger : Theme.Palette.ink2)
            .lineLimit(2)
            .fixedSize(horizontal: false, vertical: true)
            .help(text)
    }

    private func restartEngine() {
        engine.stop()
        Task { @MainActor in
            for _ in 0..<60 where engine.isRunning {
                try? await Task.sleep(nanoseconds: 250_000_000)
            }
            engine.start()
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            store.refreshNow()
        }
    }
}


/// "Try a Sample": one short clip with the starter model, opened when it is done.
struct TrySampleCard: View {
    @EnvironmentObject private var store: EngineStore

    var body: some View {
        if let model = store.sampleModel, let sm = model.sample {
            HStack(alignment: .center, spacing: Theme.Space.s5) {
                VStack(alignment: .leading, spacing: Theme.Space.s1) {
                    Text("Try a sample")
                        .font(Theme.Fonts.rowMedium)
                        .foregroundStyle(Theme.Palette.ink)
                    Text(detail(model, sm))
                        .font(Theme.Fonts.meta)
                        .foregroundStyle(failed ? Theme.Palette.danger : Theme.Palette.ink2)
                        .fixedSize(horizontal: false, vertical: true)
                    if case .running(_, let p) = store.sample {
                        ProgressLine(value: p, width: nil, height: Theme.Metric.progressHeight)
                            .padding(.top, Theme.Space.s1)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                button
            }
            .padding(.horizontal, Theme.Space.s5)
            .padding(.vertical, Theme.Space.s4)
            .background(RoundedRectangle(cornerRadius: Theme.Radius.control + 2, style: .continuous)
                .fill(Theme.Palette.white))
            .overlay(RoundedRectangle(cornerRadius: Theme.Radius.control + 2, style: .continuous)
                .strokeBorder(Theme.Palette.line, lineWidth: Theme.Metric.hairline))
            .animation(Motion.smooth, value: store.sample)
        }
    }

    private var failed: Bool {
        if case .failed = store.sample { return true }
        return false
    }

    private func detail(_ model: CatalogModel, _ sm: CatalogSample) -> String {
        switch store.sample {
        case .idle:
            return "Makes a short clip with \(model.name) from a ready-made prompt, then opens it."
        case .running(_, let p):
            return p > 0 ? "Making the sample… \(Fmt.percent(p))" : "Starting the sample…"
        case .done:
            return "Done. The video opened in your video player."
        case .failed(let why):
            return why
        }
    }

    @ViewBuilder private var button: some View {
        switch store.sample {
        case .running:
            Button("Making…") {}
                .buttonStyle(.kit(.line))
                .disabled(true)
        case .done(let out) where !out.isEmpty:
            Button("Open Video") { NSWorkspace.shared.open(URL(fileURLWithPath: out)) }
                .buttonStyle(.kit(.line))
        default:
            Button("Try a Sample") { store.trySample() }
                .buttonStyle(.kit(.ink))
                .disabled(!store.connection.isConnected)
        }
    }
}
