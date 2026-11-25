import SwiftUI

/// The request the engine is working on (or holding until memory frees up): model,
/// what is happening now, progress, and a quiet cancel button.
struct CurrentJobView: View {
    @EnvironmentObject private var store: EngineStore
    let job: Job
    var onCancel: (() -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s2) {
            HStack(alignment: .center, spacing: Theme.Space.s2) {
                Image(systemName: job.state == .running ? "bolt.fill" : "hourglass")
                    .font(.system(size: Theme.Size.meta, weight: .semibold))
                    .foregroundStyle(job.state == .running ? Theme.Palette.good : Theme.Palette.warn)
                    .frame(width: Theme.Metric.icon)
                VStack(alignment: .leading, spacing: 0) {
                    Text(job.request.model)
                        .font(Theme.Fonts.rowMedium)
                        .foregroundStyle(Theme.Palette.ink)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text(subtitle)
                        .font(Theme.Fonts.meta)
                        .foregroundStyle(Theme.Palette.muted)
                        .lineLimit(job.state == .running ? 1 : 2)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: Theme.Space.s2)
                if job.state == .running {
                    Text(Fmt.percent(job.progress))
                        .font(Theme.Fonts.rowMedium)
                        .monospacedDigit()
                        .foregroundStyle(Theme.Palette.ink)
                        .contentTransition(.numericText())
                }
                if let onCancel {
                    Button(action: onCancel) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: Theme.Size.section))
                            .foregroundStyle(Theme.Palette.faint)
                    }
                    .buttonStyle(.plain)
                    .help("Cancel this request")
                    .accessibilityLabel("Cancel request")
                }
            }
            if job.state == .running {
                ProgressLine(value: job.progress, width: nil, height: Theme.Metric.progressHeight)
                    .animation(Motion.smooth, value: job.progress)
            }
        }
        .help("Request \(job.id)")
    }

    private var subtitle: String {
        var parts: [String] = []
        if job.state == .running {
            parts.append(job.detailLine)
        } else {
            parts.append("Waiting until memory is free")
        }
        if job.state == .running, let speed = store.system?.throughput?.speedText {
            parts.append(speed)
        } else if let mode = store.modeText(for: job) {
            parts.append(mode)
        }
        return parts.joined(separator: " · ")
    }
}
