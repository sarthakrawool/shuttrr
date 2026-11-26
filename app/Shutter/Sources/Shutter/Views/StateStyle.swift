import Foundation
import SwiftUI

/// State wording and badges, the same as the dashboard: red and green only carry meaning
/// (failed, running); every other state is a quiet grey badge.
extension JobState {
    var badgeLabel: String {
        switch self {
        case .waitingMemory: return "Waiting"
        default: return label
        }
    }

    var badgeKind: KitBadge.Kind {
        switch self {
        case .running: return .good
        case .failed: return .danger
        case .waitingMemory: return .outline
        case .cancelled, .unknown: return .muted
        default: return .quiet
        }
    }

    var isLive: Bool { self == .queued || self == .waitingMemory || self == .running }
}

struct StateBadge: View {
    let job: Job

    var body: some View {
        KitBadge(text: job.state.badgeLabel, kind: job.state.badgeKind)
            .help(job.message.isEmpty ? job.state.label : job.message)
    }
}

extension Job {
    /// Title of a job row: the prompt, or the id when there is none.
    var displayTitle: String {
        let p = request.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        return p.isEmpty ? id : p.replacingOccurrences(of: "\n", with: " ")
    }

    var hasPrompt: Bool { !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }


    /// The second line of a job row (same rules as the dashboard).
    var detailLine: String {
        switch state {
        case .running:
            let stageText = stage.isEmpty ? "Starting" : stage
            if let left = remainingS, left > 0 { return "\(stageText) · \(Clock.short(left)) left" }
            return stageText
        case .queued:
            return request.priority != 0 ? "Priority \(request.priority)" : "Waiting for its turn"
        case .done:
            return outputPath.map { Fmt.fileName($0) } ?? "Finished"
        case .waitingMemory:
            return Job.memoryReason(message) ?? message
        default:
            return message
        }
    }

    /// "Needs 12.0 GB · 9.5 GB free right now" from the guard's sentence.
    static func memoryReason(_ message: String) -> String? {
        let pattern = "needs ([0-9.]+ GB), but macOS can give only ([0-9.]+ GB)"
        guard let re = try? NSRegularExpression(pattern: pattern),
              let m = re.firstMatch(in: message, range: NSRange(message.startIndex..., in: message)),
              m.numberOfRanges == 3 else { return nil }
        func group(_ i: Int) -> String {
            guard let r = Range(m.range(at: i), in: message) else { return "" }
            return String(message[r])
        }
        return "Needs \(group(1)) · \(group(2)) free right now"
    }

    /// Time column: run time once started, otherwise time spent waiting (shown muted).
    var clockSeconds: (seconds: Double, waiting: Bool) {
        let now = Date().timeIntervalSince1970
        if let start = startedAt, start > 0 { return ((finishedAt ?? now) - start, false) }
        return ((finishedAt ?? now) - createdAt, true)
    }
}

/// Memory pressure gets colour only when it is critical (meaning, not decoration).
func pressureColor(_ pressure: String) -> Color {
    pressure == "critical" ? Theme.Palette.danger : Theme.Palette.muted
}

extension JobRequestInfo {
    /// What the request asks for, in a word, when no mode is known yet.
    var kindLabel: String { durationS == nil ? "Request" : "Video request" }
}
