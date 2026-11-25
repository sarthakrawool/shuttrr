import SwiftUI

/// Fast / Low memory, with the efficiency slider that appears for Low memory. The slider
/// snaps to the ladder's rungs; the rung name, memory and time change with it.
struct ModeControl: View {
    @EnvironmentObject private var store: EngineStore
    var compact = false

    /// The step under the finger while dragging (the label and numbers follow it before the release commits it).
    @State private var previewIndex: Int?

    private static let modes = [KitOption(value: "fast", label: "Fast"),
                                KitOption(value: "low_memory", label: "Low memory")]

    var body: some View {
        if let settings = store.settings {
            VStack(alignment: .leading, spacing: Theme.Space.s3) {
                if !compact {
                    HStack(spacing: Theme.Space.s2) {
                        ModuleLabel(text: "Memory mode")
                        Spacer(minLength: Theme.Space.s2)
                        if !compact, let m = settings.measuredFor, !m.model.isEmpty {
                            Text("Measured on \(m.model)")
                                .font(Theme.Fonts.meta)
                                .foregroundStyle(Theme.Palette.muted)
                                .lineLimit(1)
                        }
                    }
                }
                KitSegmented(options: Self.modes,
                             selection: Binding(get: { settings.mode }, set: { store.setMode($0) }))

                if settings.isLowMemory {
                    slider(settings)
                        .transition(.opacity.combined(with: .move(edge: .top)))
                } else if !compact {
                    Text("Weights stay loaded for speed. Steps down by itself if a model would not fit.")
                        .font(Theme.Fonts.meta)
                        .foregroundStyle(Theme.Palette.muted)
                        .fixedSize(horizontal: false, vertical: true)
                        .transition(.opacity)
                }
            }
            .animation(Motion.smooth, value: settings.isLowMemory)
        }
    }

    @ViewBuilder
    private func slider(_ settings: EngineSettings) -> some View {
        let shown = previewIndex ?? settings.index(for: settings.efficiency)
        let rung = settings.ladder.indices.contains(shown) ? settings.ladder[shown] : nil
        VStack(alignment: .leading, spacing: Theme.Space.s2) {
            StepSlider(count: max(settings.ladder.count, 1),
                       index: Binding(get: { previewIndex ?? settings.index(for: settings.efficiency) },
                                      set: { i in
                                          previewIndex = nil
                                          store.setEfficiency(value(for: i, settings))
                                      }),
                       label: { i in settings.ladder.indices.contains(i) ? settings.ladder[i].label : "" },
                       onPreview: { i in previewIndex = i })
                .disabled(settings.ladder.isEmpty)
            if let rung {
                HStack(alignment: .firstTextBaseline, spacing: Theme.Space.s2) {
                    Text(rung.label)
                        .font(Theme.Fonts.rowMedium)
                        .foregroundStyle(Theme.Palette.ink)
                        .id(rung.id)
                        .transition(.opacity)
                    Spacer(minLength: Theme.Space.s2)
                    if let peak = settings.peak(of: rung) {
                        Text((peak.estimate ? "about " : "") + Fmt.gb(peak.bytes))
                            .font(Theme.Fonts.mono)
                            .contentTransition(.numericText())
                    }
                    if let t = settings.timeRatio(of: rung) {
                        Text(String(format: "%.1f× time", t.ratio))
                            .font(Theme.Fonts.mono)
                            .contentTransition(.numericText())
                    }
                }
                .font(Theme.Fonts.meta)
                .foregroundStyle(Theme.Palette.muted)
                .lineLimit(1)
                if !compact, !rung.detail.isEmpty {
                    Text(rung.detail)
                        .font(Theme.Fonts.meta)
                        .foregroundStyle(Theme.Palette.muted)
                        .fixedSize(horizontal: false, vertical: true)
                        .id("d" + rung.id)
                        .transition(.opacity)
                }
            }
        }
        .animation(Motion.gentle, value: rung?.id)
    }

    /// The efficiency value of step `i` (steps are evenly spaced on 0…1).
    private func value(for i: Int, _ settings: EngineSettings) -> Double {
        if settings.ladder.indices.contains(i) { return settings.ladder[i].at }
        let n = settings.ladder.count
        return n > 1 ? Double(i) / Double(n - 1) : 0
    }

    /// Distance between rungs (they are evenly spaced; falls back to quarters).
    private func step(_ settings: EngineSettings) -> Double {
        let n = settings.ladder.count
        return n > 1 ? 1.0 / Double(n - 1) : 0.25
    }
}
