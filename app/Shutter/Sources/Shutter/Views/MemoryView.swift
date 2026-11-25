import Charts
import SwiftUI

/// Memory: four hairline stat cards (one big number: Free for jobs), the used / headroom / free
/// bar, the last 10 minutes, and the protected services that make video jobs wait.
struct MemoryView: View {
    @EnvironmentObject private var store: EngineStore

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.s5) {
                PageTitle(title: "Memory", subtitle: subtitle)

                BlockingBanner()

                if let sys = store.system {
                    let parts = MemoryParts(sys)
                    HStack(alignment: .top, spacing: Theme.Space.s4) {
                        StatCard(label: "Free for models", value: Fmt.gb(parts.free), note: "Available minus the reserve", big: true)
                        StatCard(label: "Used", value: Fmt.gb(parts.used), note: "of \(Fmt.gb(parts.total))")
                        StatCard(label: "Reserve", value: Fmt.gb(parts.headroomSetting), note: "Kept for your other apps")
                        StatCard(label: "Swap", value: Fmt.gb(parts.swap), note: "In use now")
                    }
                    .fixedSize(horizontal: false, vertical: true)

                    MemoryBar(parts: parts)

                    if store.settings != nil {
                        ModeControl()
                            .kitCard()
                    }
                }

                VStack(alignment: .leading, spacing: Theme.Space.s3) {
                    SectionHeader(title: "Last 10 minutes")
                    MemoryChart(points: store.history)
                        .frame(height: Theme.Metric.chartHeight)
                        .kitCard()
                }

                VStack(alignment: .leading, spacing: Theme.Space.s3) {
                    SectionHeader(title: "Protected services")
                    KVGroup {
                        let services = store.system?.guardInfo.protected ?? []
                        if services.isEmpty {
                            KVRow("None configured", "Shuttrr starts requests whenever memory allows.", first: true)
                        }
                        ForEach(Array(services.enumerated()), id: \.offset) { item in
                            KVRow(label: item.element.name, first: item.offset == 0) {
                                if let busy = item.element.busy, !busy.isEmpty {
                                    (Text("Loaded: requests wait. ").fontWeight(.semibold) + Text(busy))
                                        .fixedSize(horizontal: false, vertical: true)
                                        .textSelection(.enabled)
                                } else {
                                    Text("Idle")
                                }
                            }
                        }
                        KVRow("GPU lock", gpuText)
                    }
                    Text("Requests never start while a protected service has a model loaded.")
                        .font(Theme.Fonts.meta)
                        .foregroundStyle(Theme.Palette.muted)
                }
            }
            .padding(.horizontal, Theme.Space.s6)
            .padding(.top, Theme.Space.s5)
            .padding(.bottom, Theme.Space.s6)
        }
    }

    private var subtitle: String {
        guard let m = store.system?.memory else { return "" }
        return "\(Fmt.gb(m.total)) total · pressure \(m.pressure)"
    }

    private var gpuText: String {
        guard let g = store.system?.gpuLock, g.held else { return "Free" }
        if let h = g.holder, !h.label.isEmpty { return "Held by \(h.label) (pid \(h.pid))" }
        return "Held"
    }
}

struct StatCard: View {
    let label: String
    let value: String
    let note: String
    var big = false

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s2) {
            Text(label)
                .font(Theme.Fonts.meta)
                .foregroundStyle(Theme.Palette.muted)
            Text(value)
                .font(big ? Theme.Fonts.big : Theme.Fonts.section)
                .foregroundStyle(Theme.Palette.ink)
                .monospacedDigit()
                .lineLimit(1)
            Spacer(minLength: 0)
            Text(note)
                .font(Theme.Fonts.meta)
                .foregroundStyle(Theme.Palette.muted)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .kitCard()
        .hoverLift()
        .accessibilityElement(children: .combine)
    }
}

/// Available memory (ink) and swap (grey) over the last 10 minutes. No accent colour.
struct MemoryChart: View {
    let points: [MemoryPoint]

    var body: some View {
        if points.count < 2 {
            Text("Collecting samples…")
                .font(Theme.Fonts.row)
                .foregroundStyle(Theme.Palette.muted)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            Chart {
                ForEach(points) { p in
                    LineMark(x: .value("Time", p.date),
                             y: .value("GB", p.availableGB),
                             series: .value("Series", "Available"))
                        .foregroundStyle(by: .value("Series", "Available"))
                    LineMark(x: .value("Time", p.date),
                             y: .value("GB", p.swapGB),
                             series: .value("Series", "Swap"))
                        .foregroundStyle(by: .value("Series", "Swap"))
                }
            }
            .chartForegroundStyleScale(["Available": Theme.Palette.violet, "Swap": Theme.Palette.amber])
            .chartYAxisLabel("GB")
            .font(Theme.Fonts.meta)
        }
    }
}
