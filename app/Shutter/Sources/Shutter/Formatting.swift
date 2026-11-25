import Foundation

enum Fmt {
    static let gib: Double = 1024 * 1024 * 1024

    /// Bytes as GB with one decimal, e.g. "41.2 GB".
    static func gb(_ bytes: Int) -> String {
        String(format: "%.1f GB", Double(bytes) / gib)
    }

    static func gb(_ bytes: Int?) -> String {
        guard let bytes else { return "–" }
        return gb(bytes)
    }

    /// Seconds as "42s", "3m 05s" or "1h 02m".
    static func duration(_ seconds: Double?) -> String {
        guard let seconds, seconds.isFinite, seconds >= 0 else { return "–" }
        let s = Int(seconds.rounded())
        if s < 60 { return "\(s)s" }
        if s < 3600 { return String(format: "%dm %02ds", s / 60, s % 60) }
        return String(format: "%dh %02dm", s / 3600, (s % 3600) / 60)
    }

    static func percent(_ fraction: Double) -> String {
        "\(Int((min(max(fraction, 0), 1) * 100).rounded()))%"
    }

    /// Last path component, for compact table cells.
    static func fileName(_ path: String?) -> String {
        guard let path, !path.isEmpty else { return "–" }
        return (path as NSString).lastPathComponent
    }
}
