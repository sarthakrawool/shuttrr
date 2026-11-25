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
            ForEach(store.catalog) { model in
                ModelDownloadRow(model: model)
            }
        }
        .padding(Theme.Space.s5)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous).fill(Theme.Palette.white))
        .overlay(RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous)
            .strokeBorder(Theme.Palette.lineStrong, lineWidth: Theme.Metric.hairline))
    }
}

/// One downloadable model: what it is, how big, progress, and the one button that fits its state.
struct ModelDownloadRow: View {
    @EnvironmentObject private var store: EngineStore
    @EnvironmentObject private var engine: EngineProcess
    let model: CatalogModel

    private var pending: Bool { store.catalogPending.contains(model.id) }
    private var state: String { model.download?.state ?? "" }
    private var remaining: Int { max(0, model.sizeBytes - (model.download?.doneBytes ?? 0)) }
    private var diskShort: Bool { !model.installed && !model.isDownloading && model.freeBytes > 0 && model.freeBytes < remaining }

    var body: some View {
        HStack(alignment: .top, spacing: Theme.Space.s5) {
            VStack(alignment: .leading, spacing: Theme.Space.s2) {
                Text(model.name)
                    .font(Theme.Fonts.rowMedium)
                    .foregroundStyle(Theme.Palette.ink)
                Text(model.summary)
                    .font(Theme.Fonts.row)
                    .foregroundStyle(Theme.Palette.ink2)
                    .fixedSize(horizontal: false, vertical: true)
                Text("\(Fmt.gb(model.sizeBytes)) download · from Hugging Face · \(Fmt.gb(model.freeBytes)) free on this Mac")
                    .font(Theme.Fonts.mono)
                    .foregroundStyle(Theme.Palette.muted)
                    .fixedSize(horizontal: false, vertical: true)
                status
                ForEach(model.notes, id: \.self) { note in
                    line(note)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            action
        }
        .padding(.horizontal, Theme.Space.s5)
        .padding(.vertical, Theme.Space.s4)
        .background(RoundedRectangle(cornerRadius: Theme.Radius.control + 2, style: .continuous)
            .strokeBorder(Theme.Palette.line, lineWidth: Theme.Metric.hairline))
        .animation(Motion.smooth, value: state)
    }

    @ViewBuilder private var status: some View {
        if model.installed && store.catalogRestartNeeded {
            line("Downloaded. Restart the engine to start using it.")
        } else if model.installed {
            line("Installed and ready.")
        } else if let d = model.download, d.isRunning {
            ProgressLine(value: d.fraction, width: nil, height: Theme.Metric.progressHeightWide)
            line("\(Fmt.gb(d.doneBytes)) of \(Fmt.gb(d.totalBytes > 0 ? d.totalBytes : model.sizeBytes)) · \(Fmt.percent(d.fraction)). You can keep using this Mac; the download carries on if you close this window.")
        } else {
            if state == "failed" {
                line((model.download?.error ?? "").isEmpty ? "The download failed." : model.download?.error ?? "", danger: true)
            } else if state == "stopped" || state == "cancelled" {
                line("Stopped at \(Fmt.gb(model.download?.doneBytes ?? 0)). It carries on from there.")
            }
            if !model.fitsThisMac {
                line("This model needs a Mac with at least \(Fmt.gb(model.minMemoryBytes)) of memory.", danger: true)
            } else if diskShort {
                line("Not enough free disk space for this download.", danger: true)
            }
        }
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
        } else {
            Button(pending ? "Starting…" : (state.isEmpty || state == "done" ? "Download" : "Resume Download")) {
                store.startDownload(model.id)
            }
            .buttonStyle(.kit(.ink))
            .disabled(pending || !model.fitsThisMac || !store.connection.isConnected)
        }
    }

    private func line(_ text: String, danger: Bool = false) -> some View {
        Text(text)
            .font(Theme.Fonts.meta)
            .foregroundStyle(danger ? Theme.Palette.danger : Theme.Palette.ink2)
            .fixedSize(horizontal: false, vertical: true)
            .textSelection(.enabled)
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
