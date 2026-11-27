import SwiftUI

/// A model's own defaults: the values every new request for it starts from. Saved by the engine
/// (PUT /v1/models/<id>/defaults), so they survive updates; "Reset to this Mac's defaults" removes them (DELETE).
struct ModelDefaultsSheet: View {
    let model: ModelInfo
    let onDone: () -> Void

    private enum Phase { case loading, ready, unsupported, failed(String) }

    @State private var phase: Phase = .loading
    @State private var options: RequestOptions?
    @State private var draft = RequestDraft()
    @State private var estimate: RequestEstimate?
    @State private var busy = false
    @State private var error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: Theme.Space.s1) {
                Text("Defaults for \(model.displayName)")
                    .font(Theme.Fonts.section)
                    .foregroundStyle(Theme.Palette.ink)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text("New requests start from these values. You can still change them for each request.")
                    .font(Theme.Fonts.meta)
                    .foregroundStyle(Theme.Palette.muted)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding([.top, .horizontal], Theme.Space.s5)

            VStack(alignment: .leading, spacing: Theme.Space.s4) { content }
                .padding(Theme.Space.s5)

            Hairline()

            HStack(spacing: Theme.Space.s3) {
                if let options, options.editable {
                    Button("Reset to This Mac's Defaults") { reset() }
                        .buttonStyle(.kit(.line))
                        .disabled(busy || options.userDefaults.isEmpty)
                        .help(options.userDefaults.isEmpty ? "Nothing saved: these are this Mac's defaults already."
                                                           : "Remove your saved values for this model.")
                }
                Spacer(minLength: Theme.Space.s3)
                Button(canSave ? "Cancel" : "Done", action: onDone)
                    .buttonStyle(.kit(.line))
                    .keyboardShortcut(.cancelAction)
                if canSave {
                    Button("Save") { save() }
                        .buttonStyle(.kit(.ink))
                        .keyboardShortcut(.return, modifiers: .command)
                        .disabled(busy || options.map { draft == RequestDraft($0.defaults, image: $0.isImage) } ?? true)
                }
            }
            .padding(.horizontal, Theme.Space.s5)
            .frame(height: Theme.Metric.toolbarHeight)
        }
        .frame(width: Theme.Metric.sheetWidth)
        .background(PaperBackground(glows: false))
        .tint(Theme.Palette.violet)
        .preferredColorScheme(.light)
        .task { await load() }
        .task(id: draft.values) { await refreshEstimate() }
    }

    private var canSave: Bool { options?.editable == true }

    @ViewBuilder
    private var content: some View {
        switch phase {
        case .loading:
            HStack(spacing: Theme.Space.s2) {
                ProgressView().controlSize(.small)
                Text("Reading the defaults…").font(Theme.Fonts.row).foregroundStyle(Theme.Palette.ink2)
            }
        case .unsupported:
            note("This engine version has no editable defaults. Update Shuttrr to change them.")
        case .failed(let message):
            note(message, color: Theme.Palette.danger)
        case .ready:
            if let options, options.editable {
                RequestValueFields(draft: $draft, options: options, estimate: estimate)
                Hairline()
                VStack(alignment: .leading, spacing: Theme.Space.s1) {
                    HStack(spacing: Theme.Space.s2) {
                        Text("This Mac's defaults").font(Theme.Fonts.metaMedium).foregroundStyle(Theme.Palette.ink2)
                        Spacer(minLength: Theme.Space.s3)
                        EstimateLine(estimate: estimate)
                    }
                    Text(options.macDefaults.summary)
                        .font(Theme.Fonts.meta)
                        .foregroundStyle(Theme.Palette.muted)
                        .fixedSize(horizontal: false, vertical: true)
                    WarningLines(estimate: estimate)
                }
                if let error { note(error, color: Theme.Palette.danger) }
            } else {
                note("This model has no editable settings. Seed and memory step can be set for each request.")
            }
        }
    }

    private func note(_ text: String, color: Color = Theme.Palette.ink2) -> some View {
        Text(text)
            .font(Theme.Fonts.row)
            .foregroundStyle(color)
            .fixedSize(horizontal: false, vertical: true)
            .textSelection(.enabled)
    }

    private func adopt(_ o: RequestOptions) {
        options = o
        draft = RequestDraft(o.defaults, image: o.isImage)
        estimate = o.estimate
        phase = .ready
    }

    private func load() async {
        do {
            if let o = try await APIClient.current().requestOptions(model: model.id) {
                adopt(o)
            } else {
                phase = .unsupported
            }
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    private func refreshEstimate() async {
        guard let options, options.editable else { return }
        if draft == RequestDraft(options.defaults, image: options.isImage), let e = options.estimate {
            estimate = e
            return
        }
        try? await Task.sleep(for: .milliseconds(300))
        guard !Task.isCancelled else { return }
        if let e = try? await APIClient.current().estimate(model: model.id, values: draft.values), !Task.isCancelled {
            estimate = e
        }
    }

    /// Saves the values that differ from this Mac's defaults, plus the ones already saved (so a value set back
    /// by hand is stored as typed).
    private func save() {
        guard let options else { return }
        if let problem = draft.problem {
            error = problem
            return
        }
        let v = draft.values, mac = options.macDefaults, mine = options.userDefaults
        var body = RequestValues()
        if v.width != mac.width || v.height != mac.height || mine.width != nil || mine.height != nil {
            body.width = v.width
            body.height = v.height
        }
        if v.frames != mac.frames || mine.frames != nil { body.frames = v.frames }
        if v.steps != mac.steps || mine.steps != nil { body.steps = v.steps }
        if v.guidance != mac.guidance || mine.guidance != nil { body.guidance = v.guidance }
        if v.shift != mac.shift || mine.shift != nil { body.shift = v.shift }
        if v.vaeTile != mac.vaeTile || mine.vaeTile != nil { body.vaeTile = v.vaeTile }
        guard !body.isEmpty else {
            onDone()
            return
        }
        run({ try await APIClient.current().saveDefaults(model: model.id, values: body) }, closes: true)
    }

    private func reset() {
        run({ try await APIClient.current().resetDefaults(model: model.id) }, closes: false)
    }

    private func run(_ call: @escaping () async throws -> RequestOptions, closes: Bool) {
        busy = true
        error = nil
        Task { @MainActor in
            defer { busy = false }
            do {
                let o = try await call()
                if closes { onDone() } else { adopt(o) }
            } catch {
                self.error = error.localizedDescription
            }
        }
    }
}
