import SwiftUI

/// The editable request values as the text in the fields (so a half-typed number is never lost).
struct RequestDraft: Equatable {
    var width = ""
    var height = ""
    var frames = ""
    var duration = ""
    var steps = ""
    var guidance = ""
    var shift = ""
    var vaeTile = true
    var fps: Double = 0
    /// An image model: no frames, duration, shift or tiled decode.
    var image = false

    init() {}

    init(_ v: RequestValues, image: Bool = false) {
        self.image = image
        width = v.width.map(String.init) ?? ""
        height = v.height.map(String.init) ?? ""
        frames = v.frames.map(String.init) ?? ""
        steps = v.steps.map(String.init) ?? ""
        guidance = v.guidance.map(RequestValues.text) ?? ""
        shift = v.shift.map(RequestValues.text) ?? ""
        vaeTile = v.vaeTile ?? true
        fps = v.fps ?? 0
        syncDuration()
    }

    static func int(_ s: String) -> Int? { Int(s.trimmingCharacters(in: .whitespaces)) }
    static func number(_ s: String) -> Double? {
        Double(s.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: ",", with: "."))
    }

    /// What the fields say, as numbers; a field that is empty or not a number is nil.
    var values: RequestValues {
        var v = RequestValues()
        v.width = Self.int(width)
        v.height = Self.int(height)
        v.steps = Self.int(steps)
        v.guidance = Self.number(guidance)
        if image { return v }
        v.frames = Self.int(frames)
        v.shift = Self.number(shift)
        v.vaeTile = vaeTile
        return v
    }

    /// The first field that is not a number, as a sentence; nil when every field reads fine.
    var problem: String? {
        let v = values
        if v.width == nil || v.height == nil { return "Width and height must be whole numbers." }
        if !image && v.frames == nil { return "Frames must be a whole number." }
        if v.steps == nil { return "Steps must be a whole number." }
        if !guidance.isEmpty, v.guidance == nil { return "Guidance must be a number." }
        if !shift.isEmpty, v.shift == nil { return "Shift must be a number." }
        return nil
    }

    /// Duration follows frames: (frames - 1) / fps.
    mutating func syncDuration() {
        guard fps > 0, let f = Self.int(frames), f >= 1 else { return }
        duration = RequestValues.text((Double(f - 1) / fps * 100).rounded() / 100)
    }

    /// Frames follow duration: duration * fps + 1, moved to the nearest count the model takes.
    mutating func syncFrames(_ limits: RequestLimits) {
        guard fps > 0, let d = Self.number(duration), d > 0 else { return }
        frames = String(limits.snapFrames(Int((d * fps).rounded()) + 1))
    }

    /// Puts the engine's suggested value into its field. False when the field is not one of ours.
    mutating func apply(_ issue: RequestIssue) -> Bool {
        guard let fix = issue.fix?.doubleValue else { return false }
        switch issue.field {
        case "width": width = String(Int(fix))
        case "height": height = String(Int(fix))
        case "frames":
            frames = String(Int(fix))
            syncDuration()
        case "steps": steps = String(Int(fix))
        case "guidance": guidance = RequestValues.text(fix)
        case "shift": shift = RequestValues.text(fix)
        default: return false
        }
        return true
    }

    static let fields = ["width", "height", "frames", "steps", "guidance", "shift"]
}

/// Size, length, steps, guidance, shift and tiled decode: the same fields in the new request sheet and in a
/// model's defaults. Issues from the estimate show under the row of their field, with "Use N" when there is a fix.
struct RequestValueFields: View {
    @Binding var draft: RequestDraft
    let options: RequestOptions
    let estimate: RequestEstimate?

    private enum Field { case frames, duration }
    @FocusState private var focus: Field?

    private static let custom = "custom"

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s4) {
            row(issues: ["width", "height", "size"]) {
                if !options.sizes.isEmpty {
                    LabeledField("Size") {
                        Picker("Size", selection: sizeChoice) {
                            ForEach(options.sizes) { s in Text(s.label).tag(s.id) }
                            Text("Custom").tag(Self.custom)
                        }
                        .labelsHidden()
                        .pickerStyle(.menu)
                        .frame(maxWidth: .infinity, minHeight: Theme.Metric.fieldHeight, alignment: .leading)
                    }
                }
                LabeledField("Width") { TextField("", text: $draft.width).kitField() }
                LabeledField("Height") { TextField("", text: $draft.height).kitField() }
            }
            if options.isImage {
                row(issues: ["steps", "guidance"]) {
                    LabeledField("Steps") { TextField("", text: $draft.steps).kitField() }
                    LabeledField("Guidance") {
                        TextField("", text: $draft.guidance).kitField()
                            .help("0 = off (Z-Image Turbo is made for 0). Above 1 also uses the negative prompt and "
                                  + "takes about twice as long.")
                    }
                }
            } else {
            row(issues: ["frames", "duration", "steps"]) {
                LabeledField("Frames") {
                    TextField("", text: $draft.frames).kitField().focused($focus, equals: .frames)
                }
                if draft.fps > 0 {
                    LabeledField("Duration (seconds)") {
                        TextField("", text: $draft.duration).kitField().focused($focus, equals: .duration)
                    }
                }
                LabeledField("Steps") { TextField("", text: $draft.steps).kitField() }
            }
            row(issues: ["guidance", "shift", "vae_tile"]) {
                LabeledField("Guidance") { TextField("", text: $draft.guidance).kitField() }
                LabeledField("Shift") { TextField("", text: $draft.shift).kitField() }
                LabeledField("Tiled decode") {
                    Toggle("", isOn: $draft.vaeTile)
                        .labelsHidden()
                        .toggleStyle(.switch)
                        .tint(Theme.Palette.violet)
                        .frame(height: Theme.Metric.fieldHeight)
                        .help("Decodes the video in tiles: less memory, a little slower.")
                }
            }
            }
            let rest = (estimate?.issues ?? []).filter { !Self.shown.contains($0.field) }
            if !rest.isEmpty { issueLines(rest) }
        }
        // Frames and duration stay in step: the one being typed in leads.
        .onChange(of: draft.frames) { _, _ in
            if focus != .duration { draft.syncDuration() }
        }
        .onChange(of: draft.duration) { _, _ in
            if focus == .duration { draft.syncFrames(options.limits) }
        }
    }

    private static let shown: Set<String> = ["width", "height", "size", "frames", "duration", "steps",
                                             "guidance", "shift", "vae_tile"]

    private var sizeChoice: Binding<String> {
        Binding(
            get: {
                let v = draft.values
                return options.sizes.first { $0.width == v.width && $0.height == v.height }?.id ?? Self.custom
            },
            set: { id in
                guard let s = options.sizes.first(where: { $0.id == id }) else { return }
                draft.width = String(s.width)
                draft.height = String(s.height)
            })
    }

    private func row<Content: View>(issues fields: [String], @ViewBuilder _ content: () -> Content) -> some View {
        let found = (estimate?.issues ?? []).filter { fields.contains($0.field) }
        return VStack(alignment: .leading, spacing: Theme.Space.s2) {
            HStack(alignment: .top, spacing: Theme.Space.s4) { content() }
            if !found.isEmpty { issueLines(found) }
        }
    }

    private func issueLines(_ issues: [RequestIssue]) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.s1) {
            ForEach(issues) { issue in
                HStack(alignment: .firstTextBaseline, spacing: Theme.Space.s2) {
                    Text(issue.message)
                        .font(Theme.Fonts.meta)
                        .foregroundStyle(Theme.Palette.danger)
                        .fixedSize(horizontal: false, vertical: true)
                    if let fix = issue.fix?.doubleValue, RequestDraft.fields.contains(issue.field) {
                        Button("Use \(RequestValues.text(fix))") { _ = draft.apply(issue) }
                            .buttonStyle(.kit(.line, small: true))
                    }
                }
            }
        }
    }
}

/// "About 10 min on this Mac", with the basis as a tooltip. Quiet while nothing is known.
struct EstimateLine: View {
    let estimate: RequestEstimate?

    var body: some View {
        if let estimate, !estimate.label.isEmpty {
            Text(estimate.label)
                .font(Theme.Fonts.meta)
                .foregroundStyle(estimate.fitLevel == "no_fit" ? Theme.Palette.warn : Theme.Palette.ink2)
                .lineLimit(1)
                .help(estimate.basis.isEmpty ? estimate.label : "\(estimate.label). Time \(estimate.basis).")
                .contentTransition(.opacity)
        }
    }
}

/// Warnings from the estimate: a quiet line each. They never block sending.
struct WarningLines: View {
    let estimate: RequestEstimate?

    var body: some View {
        if let warnings = estimate?.warnings, !warnings.isEmpty {
            VStack(alignment: .leading, spacing: Theme.Space.s1) {
                ForEach(warnings, id: \.self) { w in
                    Text(w)
                        .font(Theme.Fonts.meta)
                        .foregroundStyle(Theme.Palette.warn)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }
}
