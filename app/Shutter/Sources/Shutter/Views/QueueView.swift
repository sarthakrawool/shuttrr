import AppKit
import SwiftUI

/// Queue: title + actions, the blocking banner (only when something blocks), the memory bar,
/// then the jobs table (job, model + memory mode, status + progress, time, peak memory, cancel).
struct QueueView: View {
    @EnvironmentObject private var store: EngineStore

    let query: String
    let filter: QueueFilter
    @Binding var selectedJob: Job.ID?
    let onNewJob: () -> Void
    let onShowAll: () -> Void

    init(query: String, filter: QueueFilter, selectedJob: Binding<Job.ID?>,
         onNewJob: @escaping () -> Void, onShowAll: @escaping () -> Void) {
        self.query = query
        self.filter = filter
        self._selectedJob = selectedJob
        self.onNewJob = onNewJob
        self.onShowAll = onShowAll
    }

    static func matches(_ job: Job, _ query: String) -> Bool {
        let q = query.trimmingCharacters(in: .whitespaces).lowercased()
        guard !q.isEmpty else { return true }
        return [job.id, job.request.model, job.request.prompt, job.stateRaw, job.message]
            .joined(separator: " ").lowercased().contains(q)
    }

    /// Running first, then waiting, then queued (priority, then oldest), then finished (newest).
    private var sortedJobs: [Job] {
        func rank(_ s: JobState) -> Int {
            switch s {
            case .running: return 0
            case .waitingMemory: return 1
            case .queued: return 2
            default: return 3
            }
        }
        return store.jobs.sorted { a, b in
            let ra = rank(a.state), rb = rank(b.state)
            if ra != rb { return ra < rb }
            if ra < 3 {
                if a.request.priority != b.request.priority { return a.request.priority > b.request.priority }
                return a.createdAt < b.createdAt
            }
            return (a.finishedAt ?? a.createdAt) > (b.finishedAt ?? b.createdAt)
        }
    }

    private var visibleJobs: [Job] {
        sortedJobs.filter { filter.includes($0.state) && QueueView.matches($0, query) }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.s5) {
                HStack(alignment: .center, spacing: Theme.Space.s4) {
                    PageTitle(title: "Requests", subtitle: subtitle)
                    Spacer()
                    if store.queuePauseSupported {
                        Button(store.queuePaused ? "Resume" : "Pause") {
                            store.setQueuePaused(!store.queuePaused)
                        }
                        .buttonStyle(.kit(.line))
                        .disabled(!store.connection.isConnected)
                    }
                    Button("New Request", action: onNewJob)
                        .buttonStyle(.kit(.ink))
                        .disabled(!store.connection.isConnected)
                }

                BlockingBanner()

                if let sys = store.system {
                    MemoryBar(parts: MemoryParts(sys))
                }

                jobsTable
            }
            .padding(.horizontal, Theme.Space.s6)
            .padding(.top, Theme.Space.s5)
            .padding(.bottom, Theme.Space.s6)
        }
    }

    private var subtitle: String {
        guard let q = store.system?.queue else { return "" }
        var parts: [String] = []
        if q.running > 0 { parts.append("\(q.running) running") }
        if q.waitingMemory > 0 { parts.append("\(q.waitingMemory) waiting for memory") }
        if q.queued > 0 { parts.append("\(q.queued) queued") }
        parts.append("\(q.done) done")
        if q.failed > 0 { parts.append("\(q.failed) failed") }
        return parts.joined(separator: " · ")
    }

    @ViewBuilder
    private var jobsTable: some View {
        if store.jobs.isEmpty {
            if store.connection.isConnected {
                EmptyStateView(title: "No requests yet",
                               message: "Requests arrive from ComfyUI and other apps through the API. Press ⌘N to send one from here.") {
                    Button("New Request", action: onNewJob).buttonStyle(.kit(.ink))
                }
            }
        } else if visibleJobs.isEmpty {
            EmptyStateView(title: "No matching requests",
                           message: query.isEmpty ? "No \(filter.rawValue) requests right now." : "Nothing matches “\(query)”. Press esc in the search field to clear it.") {
                Button("Show All", action: onShowAll).buttonStyle(.kit(.line))
            }
        } else {
            VStack(spacing: 0) {
                QueueHeaderRow()
                ForEach(visibleJobs) { job in
                    JobRow(job: job, selected: selectedJob == job.id) {
                        store.cancel(job)
                    }
                    .onTapGesture {
                        selectedJob = (selectedJob == job.id) ? nil : job.id
                    }
                    .contextMenu { JobMenu(job: job) { store.cancel(job) } }
                }
            }
        }
    }
}

// MARK: - Columns (header and rows share these widths)

private extension View {
    func column(_ width: CGFloat?, _ alignment: Alignment = .leading) -> some View {
        Group {
            if let width {
                self.frame(width: width, alignment: alignment)
            } else {
                self.frame(maxWidth: .infinity, alignment: alignment)
            }
        }
        .padding(.horizontal, Theme.Space.s3)
    }
}

private struct QueueHeaderRow: View {
    var body: some View {
        HStack(spacing: 0) {
            Text("Request").column(nil)
            Text("Model").column(Theme.Metric.colModel)
            Text("Status").column(Theme.Metric.colStatus)
            Text("Time").column(Theme.Metric.colTime, .trailing)
            Text("Peak").column(Theme.Metric.colPeak, .trailing)
            Color.clear.frame(height: 1).column(Theme.Metric.colAction)
        }
        .font(Theme.Fonts.metaMedium)
        .foregroundStyle(Theme.Palette.muted)
        .frame(height: Theme.Metric.tableHeaderHeight)
        .overlay(alignment: .bottom) { Hairline() }
        .accessibilityHidden(true)
    }
}

private struct JobRow: View {
    @EnvironmentObject private var store: EngineStore
    let job: Job
    let selected: Bool
    let onCancel: () -> Void

    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 0) {
                Text(middleTruncated(job.displayTitle, max: 90))
                    .font(job.hasPrompt ? Theme.Fonts.rowMedium : Theme.Fonts.mono)
                    .foregroundStyle(Theme.rowTitle(selected: selected))
                Text(job.hasPrompt ? "\(job.id) · \(job.detailLine)" : job.detailLine)
                    .font(Theme.Fonts.meta)
                    .foregroundStyle(Theme.rowMeta(selected: selected))
                    .help(job.message)
            }
            .lineLimit(1)
            .truncationMode(.tail)
            .column(nil)

            VStack(alignment: .leading, spacing: 0) {
                Text(job.request.model)
                    .font(Theme.Fonts.row)
                    .foregroundStyle(Theme.rowTitle(selected: selected))
                Text(store.modeText(for: job) ?? job.request.kindLabel)
                    .font(Theme.Fonts.meta)
                    .foregroundStyle(Theme.rowMeta(selected: selected))
            }
            .lineLimit(1)
            .truncationMode(.middle)
            .column(Theme.Metric.colModel)

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: Theme.Space.s2) {
                    StateBadge(job: job)
                    if job.state == .running || (job.state.isLive && job.progress > 0) {
                        ProgressLine(value: job.progress,
                                     tint: selected ? Theme.Palette.onAccent : Theme.Palette.ink,
                                     track: selected ? Theme.Palette.onAccent.opacity(0.3) : Theme.Palette.fill)
                        Text(Fmt.percent(job.progress))
                            .font(Theme.Fonts.meta)
                            .monospacedDigit()
                            .foregroundStyle(Theme.rowMeta(selected: selected))
                    }
                }
                // "About 3 min left" / "Estimating…" / "About 2 min once it starts" (0.2.11)
                if job.state.isLive && !store.isCancelling(job), let t = job.timeLeftText {
                    Text(t)
                        .font(Theme.Fonts.meta)
                        .monospacedDigit()
                        .lineLimit(1)
                        .foregroundStyle(Theme.rowMeta(selected: selected))
                }
            }
            .column(Theme.Metric.colStatus)

            Group {
                if job.state.isLive {
                    TimelineView(.periodic(from: .now, by: 1)) { _ in clockText }
                } else {
                    clockText
                }
            }
            .column(Theme.Metric.colTime, .trailing)

            Text(Fmt.gb(job.peakBytes))
                .font(Theme.Fonts.row)
                .monospacedDigit()
                .foregroundStyle(job.peakBytes == nil ? Theme.rowMeta(selected: selected) : (selected ? Theme.Palette.onAccent : Theme.Palette.ink2))
                .column(Theme.Metric.colPeak, .trailing)

            Group {
                if job.state.isLive && store.isCancelling(job) {
                    HStack(spacing: Theme.Space.s1) {
                        ProgressView().controlSize(.mini)
                        Text("Cancelling…")
                            .font(Theme.Fonts.meta)
                            .lineLimit(1)
                            .fixedSize()
                            .foregroundStyle(Theme.rowMeta(selected: selected))
                    }
                    .help("Stopping at the next safe point, usually within a couple of seconds")
                } else if job.state.isLive {
                    Button("Cancel", action: onCancel)
                        .buttonStyle(.kit(.line, small: true))
                        .help("Cancel this request")
                } else {
                    Color.clear.frame(height: 1)
                }
            }
            .column(Theme.Metric.colAction, .trailing)
        }
        .frame(height: Theme.Metric.rowHeight)
        .background(selected ? Theme.Palette.accent : Color.clear)
        .overlay(alignment: .bottom) { Hairline() }
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }

    private var clockText: some View {
        let c = job.clockSeconds
        return Text(Clock.format(c.seconds))
            .font(Theme.Fonts.row)
            .monospacedDigit()
            .foregroundStyle(c.waiting ? Theme.rowMeta(selected: selected) : (selected ? Theme.Palette.onAccent : Theme.Palette.ink2))
    }
}

/// Right-click menu on a job row.
private struct JobMenu: View {
    let job: Job
    let onCancel: () -> Void

    var body: some View {
        if let path = job.outputPath {
            Button("View in Finder") {
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
            }
        }
        Button("Copy Request ID") {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(job.id, forType: .string)
        }
        if !job.message.isEmpty {
            Button("Copy Message") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(job.message, forType: .string)
            }
        }
        if job.state.isLive {
            Divider()
            Button("Cancel Request", action: onCancel)
        }
    }
}
