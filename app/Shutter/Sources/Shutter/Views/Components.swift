import SwiftUI

// Shared components in the personal-brand look (paper, ink, amber and violet). Every number comes from Theme.

// MARK: - Buttons: near-black primary, quiet partner

struct KitButtonStyle: ButtonStyle {
    /// ink = the primary action (system accent); line / fill = quiet glass; onBanner = quiet, on a banner.
    enum Kind { case ink, line, fill, onBanner }

    var kind: Kind
    var small: Bool
    @Environment(\.isEnabled) private var isEnabled

    init(kind: Kind = .line, small: Bool = false) {
        self.kind = kind
        self.small = small
    }

    func makeBody(configuration: Configuration) -> some View {
        let shape = Capsule()
        return configuration.label
            .font(small ? Theme.Fonts.metaMedium : Theme.Fonts.button)
            .lineLimit(1)
            .fixedSize(horizontal: true, vertical: false)      // a label is never truncated
            .padding(.horizontal, small ? Theme.Space.s3 : Theme.Space.s4)
            .frame(height: small ? Theme.Metric.smallButtonHeight : Theme.Metric.buttonHeight)
            .foregroundStyle(kind == .ink ? Theme.Palette.onAccent : Theme.Palette.ink)
            .background(shape.fill(kind == .ink ? AnyShapeStyle(Theme.Palette.ink) : AnyShapeStyle(Theme.Palette.white)))
            .overlay(shape.strokeBorder(kind == .ink ? Color.clear : Theme.Palette.lineStrong,
                                        lineWidth: Theme.Metric.hairline))
            .contentShape(shape)
            .scaleEffect(configuration.isPressed ? 0.96 : 1)
            .opacity(isEnabled ? (configuration.isPressed ? 0.85 : 1) : 0.4)
            .animation(Motion.lively, value: configuration.isPressed)
    }
}

extension ButtonStyle where Self == KitButtonStyle {
    static func kit(_ kind: KitButtonStyle.Kind = .line, small: Bool = false) -> KitButtonStyle {
        KitButtonStyle(kind: kind, small: small)
    }
}

// MARK: - Segmented control (.mac-seg): grey track, selected segment in ink

struct KitOption<Value: Hashable>: Identifiable {
    let value: Value
    let label: String
    var id: Value { value }
}

struct KitSegmented<Value: Hashable>: View {
    let options: [KitOption<Value>]
    @Binding var selection: Value
    @Namespace private var thumb

    var body: some View {
        HStack(spacing: 0) {
            ForEach(options) { option in
                let on = option.value == selection
                Button {
                    withAnimation(Motion.snappy) { selection = option.value }
                } label: {
                    Text(option.label)
                        .font(on ? Theme.Fonts.rowMedium : Theme.Fonts.row)
                        .lineLimit(1)
                        .fixedSize()
                        .padding(.horizontal, Theme.Space.s3)
                        .frame(maxWidth: .infinity)
                        .frame(height: Theme.Metric.segmentHeight - Theme.Space.s1)
                        .foregroundStyle(on ? Theme.Palette.onAccent : Theme.Palette.ink2)
                        .background {
                            if on {
                                Capsule()
                                    .fill(Theme.Palette.ink)
                                    .matchedGeometryEffect(id: "thumb", in: thumb)
                            }
                        }
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(on ? .isSelected : [])
            }
        }
        .padding(Theme.Space.s0)
        .background(Capsule().fill(Theme.Palette.white))
        .overlay(Capsule().strokeBorder(Theme.Palette.line, lineWidth: Theme.Metric.hairline))
    }
}

// MARK: - Key caps (.mac-kbd) and hints

struct KeyCap: View {
    let key: String
    var filled = false

    var body: some View {
        Text(key)
            .font(Theme.Fonts.monoMedium)
            .foregroundStyle(Theme.Palette.ink)
            .padding(.horizontal, Theme.Space.s1)
            .frame(minWidth: Theme.Metric.keyCap, minHeight: Theme.Metric.keyCap)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.key, style: .continuous)
                    .fill(Theme.Palette.fill)
            )
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.key, style: .continuous)
                    .strokeBorder(filled ? Color.clear : Theme.Palette.lineStrong, lineWidth: Theme.Metric.hairline)
            )
    }
}

struct KeyHint: View {
    let keys: [String]
    let label: String

    var body: some View {
        HStack(spacing: Theme.Space.s2) {
            HStack(spacing: Theme.Space.s1) {
                ForEach(keys, id: \.self) { KeyCap(key: $0, filled: true) }
            }
            Text(label).font(Theme.Fonts.meta).foregroundStyle(Theme.Palette.muted)
        }
    }
}

// MARK: - Badges, counts, dots

struct KitBadge: View {
    enum Kind { case good, danger, quiet, outline, muted }

    let text: String
    var kind: Kind = .quiet

    var body: some View {
        Text(text.uppercased())
            .font(Theme.Fonts.label)
            .tracking(0.8)
            .lineLimit(1)
            .foregroundStyle(foreground)
            .padding(.horizontal, Theme.Space.s2)
            .frame(height: Theme.Metric.badgeHeight)
            .background(Capsule().fill(background))
                        .fixedSize()
    }

    private var foreground: Color {
        switch kind {
        case .good: return Theme.Palette.good
        case .danger: return Theme.Palette.danger
        case .quiet: return Theme.Palette.ink2
        case .outline: return Theme.Palette.warn
        case .muted: return Theme.Palette.muted
        }
    }

    private var background: Color {
        switch kind {
        case .good: return Theme.Palette.good.opacity(0.16)
        case .danger: return Theme.Palette.danger.opacity(0.16)
        case .quiet, .muted: return Theme.Palette.fill
        case .outline: return Theme.Palette.amberSoft
        }
    }
}

/// Grey count pill next to a section header (.mac-count).
struct CountPill: View {
    let count: Int

    var body: some View {
        Text("\(count)")
            .font(Theme.Fonts.monoMedium)
            .foregroundStyle(Theme.Palette.ink)
            .padding(.horizontal, Theme.Space.s2)
            .frame(minWidth: Theme.Space.s5, minHeight: Theme.Metric.countHeight)
            .background(Capsule().fill(Theme.Palette.fill))
    }
}

struct StatusDot: View {
    let color: Color
    var body: some View {
        Circle().fill(color).frame(width: Theme.Metric.dot, height: Theme.Metric.dot)
    }
}

// MARK: - Headings

/// 24 pt light title + 12 pt subtitle (.mac-title / .mac-subtitle).
struct PageTitle: View {
    let title: String
    var subtitle: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(title).font(Theme.Fonts.title).foregroundStyle(Theme.Palette.ink)
            Text(subtitle.isEmpty ? " " : subtitle)
                .font(Theme.Fonts.meta)
                .foregroundStyle(Theme.Palette.muted)
                .lineLimit(2)
                .textSelection(.enabled)
        }
    }
}

/// 16 pt section header with an optional count pill and trailing control (.mac-h).
struct SectionHeader<Trailing: View>: View {
    let title: String
    var count: Int?
    @ViewBuilder var trailing: () -> Trailing

    var body: some View {
        HStack(spacing: Theme.Space.s2) {
            Text(title).font(Theme.Fonts.section).foregroundStyle(Theme.Palette.ink)
            if let count { CountPill(count: count) }
            Spacer(minLength: Theme.Space.s4)
            trailing()
        }
    }
}

extension SectionHeader where Trailing == EmptyView {
    init(title: String, count: Int? = nil) {
        self.title = title
        self.count = count
        self.trailing = { EmptyView() }
    }
}

// MARK: - Progress and memory bar

/// Accent on a quiet track.
struct ProgressLine: View {
    var value: Double
    var width: CGFloat? = Theme.Metric.progressWidth
    var height: CGFloat = Theme.Metric.progressHeight
    var tint: Color? = nil      // nil = the brand gradient
    var track: Color = Theme.Palette.fill

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(track)
                Capsule().fill(tint.map { AnyShapeStyle($0) } ?? AnyShapeStyle(Brand.gradient))
                    .frame(width: geo.size.width * CGFloat(min(max(value, 0), 1)))
                    .animation(Motion.smooth, value: value)
            }
        }
        .frame(width: width, height: height)
        .accessibilityElement()
        .accessibilityLabel("Progress")
        .accessibilityValue(Fmt.percent(value))
    }
}

/// Used / headroom / free, as on the dashboard: used in ink, headroom in grey, free is the track.
struct MemoryParts {
    let total: Int
    let used: Int
    let headroom: Int        // shown part of the headroom (never more than what is available)
    let headroomSetting: Int
    let free: Int            // free for jobs = available - headroom
    let swap: Int
    let pressure: String

    init(_ system: SystemInfo) {
        let m = system.memory
        total = m.total
        used = m.used
        headroomSetting = system.guardInfo.headroomBytes
        headroom = min(headroomSetting, m.available)
        free = max(0, m.available - headroomSetting)
        swap = m.swapUsed
        pressure = m.pressure
    }
}

struct MemoryBar: View {
    let parts: MemoryParts
    var showLegend = true

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s2) {
            GeometryReader { geo in
                let total = CGFloat(max(parts.total, 1))
                HStack(spacing: 0) {
                    Rectangle().fill(Brand.gradient)
                        .frame(width: geo.size.width * CGFloat(parts.used) / total)
                    Rectangle().fill(Theme.Palette.amberSoft)
                        .frame(width: geo.size.width * CGFloat(parts.headroom) / total)
                    Spacer(minLength: 0)
                }
                .background(Theme.Palette.fill)
                .clipShape(Capsule())
                .animation(Motion.smooth, value: parts.used)
                .animation(Motion.smooth, value: parts.headroom)
            }
            .frame(height: Theme.Metric.memoryBarHeight)
            .accessibilityElement()
            .accessibilityLabel("Memory: used \(Fmt.gb(parts.used)), free for models \(Fmt.gb(parts.free))")

            if showLegend {
                HStack(spacing: Theme.Space.s5) {
                    LegendItem(color: Theme.Palette.violet, label: "Used", value: Fmt.gb(parts.used))
                    if parts.headroomSetting > 0 {
                        LegendItem(color: Theme.Palette.amberSoft, label: "Headroom", value: Fmt.gb(parts.headroom))
                    }
                    LegendItem(color: Theme.Palette.fill, label: "Free for models", value: Fmt.gb(parts.free), outlined: true)
                    LegendItem(color: nil, label: "Swap", value: Fmt.gb(parts.swap))
                    Spacer(minLength: 0)
                }
            }
        }
    }
}

struct LegendItem: View {
    let color: Color?
    let label: String
    let value: String
    var outlined = false

    var body: some View {
        HStack(spacing: Theme.Space.s1) {
            if let color {
                RoundedRectangle(cornerRadius: Theme.Radius.legend)
                    .fill(color)
                    .overlay(RoundedRectangle(cornerRadius: Theme.Radius.legend).strokeBorder(outlined ? Theme.Palette.lineStrong : Color.clear, lineWidth: Theme.Metric.hairline))
                    .frame(width: Theme.Metric.legendKey, height: Theme.Metric.legendKey)
            }
            Text(label).foregroundStyle(Theme.Palette.muted)
            Text(value).font(Theme.Fonts.monoMedium).foregroundStyle(Theme.Palette.ink)
        }
        .font(Theme.Fonts.meta)
        .lineLimit(1)
        .fixedSize()
    }
}

// MARK: - Cards, groups, banners, empty states

extension View {
    /// Glass card (see Glass.swift).
    func kitCard(padding: CGFloat = Theme.Space.s4) -> some View {
        glassCard(padding: padding)
    }

    /// Kit text field: 32 pt, 8 pt radius, strong hairline.
    func kitField(height: CGFloat? = Theme.Metric.fieldHeight) -> some View {
        self
            .textFieldStyle(.plain)
            .font(Theme.Fonts.row)
            .foregroundStyle(Theme.Palette.ink)
            .padding(.horizontal, Theme.Space.s3)
            .padding(.vertical, height == nil ? Theme.Space.s2 : 0)
            .frame(height: height)
            .background(RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous).fill(Theme.Palette.white))
            .overlay(RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                .strokeBorder(Theme.Palette.lineStrong, lineWidth: Theme.Metric.hairline))
    }
}

/// Group of key/value rows separated by hairlines (.mac-group).
struct KVGroup<Content: View>: View {
    @ViewBuilder var content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            content()
        }
        .padding(.horizontal, Theme.Space.s4)
        .padding(.vertical, Theme.Space.s1)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassCard(padding: 0)
    }
}

struct KVRow<Value: View>: View {
    let label: String
    var first = false
    @ViewBuilder var value: () -> Value

    var body: some View {
        VStack(spacing: 0) {
            if !first { Rectangle().fill(Theme.Palette.line).frame(height: Theme.Metric.hairline) }
            HStack(alignment: .center, spacing: Theme.Space.s4) {
                Text(label)
                    .font(Theme.Fonts.row)
                    .foregroundStyle(Theme.Palette.muted)
                    .frame(width: Theme.Metric.kvLabelWidth, alignment: .leading)
                value()
                    .font(Theme.Fonts.row)
                    .foregroundStyle(Theme.Palette.ink)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(.vertical, Theme.Space.s2)
            .frame(minHeight: Theme.Metric.kvRowHeight)
        }
    }
}

extension KVRow where Value == Text {
    init(_ label: String, _ text: String, first: Bool = false) {
        self.label = label
        self.first = first
        self.value = { Text(text) }
    }
}

/// Dark banner (.mac-banner), shown only when something blocks the queue.
/// The line is 16 pt so the page keeps one 24 pt focal point.
struct KitBanner<Actions: View>: View {
    let symbol: String
    let title: String
    let detail: String
    @ViewBuilder var actions: () -> Actions

    var body: some View {
        HStack(spacing: Theme.Space.s4) {
            ZStack {
                Circle().fill(Theme.Palette.bannerIcon.opacity(0.18))
                Image(systemName: symbol)
                    .font(.system(size: Theme.Size.section, weight: .medium))
                    .foregroundStyle(Theme.Palette.bannerIcon)
            }
            .frame(width: Theme.Metric.bannerIcon - Theme.Space.s2, height: Theme.Metric.bannerIcon - Theme.Space.s2)

            VStack(alignment: .leading, spacing: Theme.Space.s0) {
                Text(title).font(Theme.Fonts.rowMedium).foregroundStyle(Theme.Palette.ink)
                Text(detail)
                    .font(Theme.Fonts.meta)
                    .foregroundStyle(Theme.Palette.bannerInk)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
            }
            Spacer(minLength: Theme.Space.s4)
            actions()
        }
        .padding(.vertical, Theme.Space.s3)
        .padding(.horizontal, Theme.Space.s4)
        .background(RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous).fill(Theme.Palette.banner))
        .overlay(RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous)
            .strokeBorder(Theme.Palette.bannerIcon.opacity(0.25), lineWidth: Theme.Metric.hairline))
        .accessibilityElement(children: .contain)
    }
}

extension KitBanner where Actions == EmptyView {
    init(symbol: String, title: String, detail: String) {
        self.symbol = symbol
        self.title = title
        self.detail = detail
        self.actions = { EmptyView() }
    }
}

/// Dashed empty state that says what to do.
struct EmptyStateView<Action: View>: View {
    let title: String
    let message: String
    @ViewBuilder var action: () -> Action

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s3) {
            Text(title).font(Theme.Fonts.section).foregroundStyle(Theme.Palette.ink)
            Text(message)
                .font(Theme.Fonts.row)
                .foregroundStyle(Theme.Palette.ink2)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
            action()
        }
        .padding(.vertical, Theme.Space.s6)
        .padding(.horizontal, Theme.Space.s5)
        .frame(maxWidth: .infinity, alignment: .leading)
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous)
                .strokeBorder(Theme.Palette.lineStrong, style: StrokeStyle(lineWidth: Theme.Metric.hairline, dash: Theme.Metric.dash))
        )
    }
}

extension EmptyStateView where Action == EmptyView {
    init(title: String, message: String) {
        self.title = title
        self.message = message
        self.action = { EmptyView() }
    }
}

/// 1 pt hairline separator.
struct Hairline: View {
    var vertical = false
    var body: some View {
        Rectangle()
            .fill(Theme.Palette.line)
            .frame(width: vertical ? Theme.Metric.hairline : nil, height: vertical ? nil : Theme.Metric.hairline)
    }
}

// MARK: - Formatting used by several views

enum Clock {
    /// "0:42", "12:05", "1:02:03": the dashboard's time column.
    static func format(_ seconds: Double?) -> String {
        guard let seconds, seconds.isFinite else { return "–" }
        let s = max(0, Int(seconds.rounded()))
        let h = s / 3600, m = (s % 3600) / 60, x = s % 60
        if h > 0 { return String(format: "%d:%02d:%02d", h, m, x) }
        return String(format: "%d:%02d", m, x)
    }

    /// "42 s" or "3 min", for "… left".
    static func short(_ seconds: Double?) -> String {
        guard let seconds, seconds.isFinite else { return "" }
        let s = Int(seconds.rounded())
        return s < 90 ? "\(s) s" : "\(Int((Double(s) / 60).rounded())) min"
    }
}

/// Shortens in the middle: "screencap…0-01-22.pdf".
func middleTruncated(_ s: String, max n: Int) -> String {
    guard s.count > n, n > 1 else { return s }
    let head = Int((Double(n - 1) * 0.6).rounded(.up))
    let tail = n - 1 - head
    return String(s.prefix(head)) + "…" + String(s.suffix(tail))
}
