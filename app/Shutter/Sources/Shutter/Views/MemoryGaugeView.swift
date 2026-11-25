import SwiftUI

/// Memory module for the menu bar panel: the one big number (free for models), the
/// used / headroom / free bar and its legend.
struct MemoryGaugeView: View {
    let system: SystemInfo

    var body: some View {
        let parts = MemoryParts(system)
        VStack(alignment: .leading, spacing: Theme.Space.s2) {
            HStack(alignment: .firstTextBaseline, spacing: Theme.Space.s1) {
                ModuleLabel(text: "Memory")
                Spacer(minLength: Theme.Space.s2)
                (Text("Swap ").foregroundColor(Theme.Palette.muted)
                    + Text(Fmt.gb(parts.swap)).font(Theme.Fonts.monoMedium).foregroundColor(parts.swap > 0 ? Theme.Palette.warn : Theme.Palette.ink))
                    .font(Theme.Fonts.meta)
                    .lineLimit(1)
            }
            HStack(alignment: .firstTextBaseline, spacing: Theme.Space.s2) {
                Text(Fmt.gb(parts.free))
                    .font(Theme.Fonts.big)
                    .lineLimit(1)
                    .fixedSize()
                    .monospacedDigit()
                    .foregroundStyle(Theme.Palette.ink)
                    .contentTransition(.numericText())
                Text("free for models")
                    .font(Theme.Fonts.row)
                    .foregroundStyle(Theme.Palette.muted)
                    .lineLimit(1)
                Spacer(minLength: 0)
            }
            MemoryBar(parts: parts, showLegend: false)
                .padding(.top, Theme.Space.s0)
            // The legend never decides the panel's width: it drops an item rather than overflow.
            ViewThatFits(in: .horizontal) {
                HStack(spacing: Theme.Space.s3) {
                    LegendItem(color: Theme.Palette.violet, label: "Used", value: Fmt.gb(parts.used))
                    LegendItem(color: Theme.Palette.amberSoft, label: "Reserve", value: Fmt.gb(parts.headroom))
                }
                LegendItem(color: Theme.Palette.violet, label: "Used", value: Fmt.gb(parts.used))
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            if parts.pressure == "critical" {
                Label("macOS reports critical memory pressure.", systemImage: "exclamationmark.triangle.fill")
                    .font(Theme.Fonts.meta)
                    .foregroundStyle(Theme.Palette.danger)
            }
        }
        .animation(.easeOut(duration: 0.3), value: parts.free)
    }
}
