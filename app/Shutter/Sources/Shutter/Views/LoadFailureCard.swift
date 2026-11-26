import AppKit
import SwiftUI

/// The red tinted popup for a model that could not load: memory short (with the numbers) or another failure.
struct LoadFailureCard: View {
    @EnvironmentObject private var store: EngineStore
    let failure: LoadFailure
    @State private var copied = false

    private func copyDiagnostics() {
        Task { @MainActor in
            let text = await Diagnostics.collect(engineVersion: store.system?.version)
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
            copied = true
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            copied = false
        }
    }

    static let tint = Color(hex: 0xF9E4DF)        // red tinted paper
    static let edge = Color(hex: 0xC2412D, alpha: 0.35)

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s3) {
            HStack(alignment: .top, spacing: Theme.Space.s3) {
                Image(systemName: failure.isMemory ? "memorychip" : "exclamationmark.triangle")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(Theme.Palette.danger)
                    .frame(width: 24, height: 24)
                VStack(alignment: .leading, spacing: Theme.Space.s2) {
                    Text(failure.title)
                        .font(Theme.Fonts.section)
                        .foregroundStyle(Theme.Palette.ink)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(failure.body)
                        .font(Theme.Fonts.row)
                        .foregroundStyle(Theme.Palette.ink)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                    Text(failure.help)
                        .font(Theme.Fonts.row)
                        .foregroundStyle(Theme.Palette.ink2)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
            }
            ViewThatFits(in: .horizontal) {
                HStack(spacing: Theme.Space.s2) { Spacer(minLength: 0); buttons }
                VStack(alignment: .trailing, spacing: Theme.Space.s2) { buttons }
                    .frame(maxWidth: .infinity, alignment: .trailing)
            }
        }
        .padding(Theme.Space.s5)
        .frame(maxWidth: 560, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous).fill(Self.tint))
        .overlay(RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous)
            .strokeBorder(Self.edge, lineWidth: Theme.Metric.hairline))
        .shadow(color: Theme.Palette.glassShadow, radius: 18, y: 8)
        .accessibilityElement(children: .contain)
    }

    @ViewBuilder private var buttons: some View {
        Button("Close") { store.loadFailure = nil }
            .buttonStyle(.kit(.line))
            .keyboardShortcut(.cancelAction)
        if failure.isMemory {
            Button("Load When Memory Frees Up") {
                let id = failure.modelID
                store.loadFailure = nil
                store.loadModel(id, wait: true)
            }
            .buttonStyle(.kit(.line))
        } else {
            Button(copied ? "Copied" : "Copy Diagnostics") { copyDiagnostics() }
                .buttonStyle(.kit(.line))
                .help("Copy the details (engine log included) to send with a bug report")
        }
        Button("Try Again") {
            let id = failure.modelID
            store.loadFailure = nil
            store.loadModel(id)
        }
        .buttonStyle(.kit(.ink))
        .keyboardShortcut(.defaultAction)
    }
}
