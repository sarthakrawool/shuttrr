import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// Sheet for POST /v1/jobs: model, prompt, image path, audio path, seed, priority, and an Advanced section
/// (size, frames or duration, steps, guidance, shift, tiled decode, memory step) that starts from the model's
/// defaults and can be changed freely. An engine without request options shows the plain sheet.
struct NewJobSheet: View {
    @EnvironmentObject private var store: EngineStore
    @Environment(\.dismiss) private var dismiss

    let initialModel: String?

    init(initialModel: String? = nil) {
        self.initialModel = initialModel
    }

    @State private var model = ""
    @State private var prompt = ""
    @State private var imagePath = ""
    @State private var audioPath = ""
    @State private var framing = "auto"
    @State private var seed = 42
    @State private var priority = 0
    @State private var durationText = ""
    // Voice models (Fish Audio Drama 3): what the reference clip says, takes, splitting long text
    @State private var refText = ""
    @State private var takes = 1
    @State private var splitText = true
    // Image models (Z-Image): negative prompt, how many pictures, look preset, LoRA
    @State private var negativePrompt = ""
    @State private var imageCount = 1
    @State private var style = ""
    @State private var loraPath = ""
    @State private var loraStrength = 1.0
    /// Reference pictures for models that read them (MiniMax-H3-Image), referred to as <Picture 1>, <Picture 2>, …
    @State private var refPaths: [String] = ["", "", ""]
    @State private var cleanTexture = true
    @State private var submitting = false
    @State private var error: String?

    /// Request options of the chosen model; nil while loading and on an engine that has none.
    @State private var options: RequestOptions?
    @State private var draft = RequestDraft()
    @State private var estimate: RequestEstimate?
    /// Testing: SHUTTER_TEST_ADVANCED=1 opens the Advanced section.
    @State private var advancedOpen = ProcessInfo.processInfo.environment["SHUTTER_TEST_ADVANCED"] != nil
    @State private var formHeight: CGFloat = 380
    /// Memory step for this request: starts from the engine's setting.
    @State private var mode = "fast"
    @State private var rungIndex = 0
    @State private var modeTouched = false

    private var editable: Bool { options?.editable == true }
    private var showsMemoryStep: Bool { options != nil && store.settings != nil }
    private var showsAdvanced: Bool { editable || showsMemoryStep }

    private struct EstimateKey: Equatable {
        var model: String
        var values: RequestValues
    }

    /// The chosen model makes speech (kind "audio"): the sheet shows the voice fields instead of image/duration.
    private var isVoice: Bool {
        (store.models.first { $0.id == model }?.kind ?? "") == "audio"
    }

    /// The chosen model makes pictures (kind "image"): prompt, negative prompt, count and style instead of
    /// image / audio paths and duration.
    private var isImage: Bool {
        options?.isImage == true || (store.models.first { $0.id == model }?.kind ?? "") == "image"
    }

    private var modelChoices: [ModelInfo] {
        store.installedModels.isEmpty ? store.models : store.installedModels
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: Theme.Space.s1) {
                Text("New request").font(Theme.Fonts.section).foregroundStyle(Theme.Palette.ink)
                Text("Requests start when memory allows. Paths are files on this Mac.")
                    .font(Theme.Fonts.meta)
                    .foregroundStyle(Theme.Palette.muted)
            }
            .padding([.top, .horizontal], Theme.Space.s5)

            ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.s4) {
                LabeledField("Model") { modelField }

                if isVoice {
                    voiceFields
                } else if isImage {
                    imageFields
                } else {
                LabeledField("Prompt") {
                    TextField("", text: $prompt, prompt: Text("Optional"), axis: .vertical)
                        .lineLimit(3...6)
                        .kitField(height: nil)
                        .frame(minHeight: Theme.Metric.promptHeight, alignment: .top)
                }

                HStack(alignment: .top, spacing: Theme.Space.s4) {
                    LabeledField("Image path") {
                        PathField(path: $imagePath, placeholder: "~/Pictures/start.png", types: [.image], title: "image")
                    }
                    LabeledField("Audio path") {
                        PathField(path: $audioPath, placeholder: "~/Music/voice.wav", types: [.audio], title: "audio")
                    }
                }

                HStack(alignment: .top, spacing: Theme.Space.s4) {
                    LabeledField("Seed") {
                        TextField("", value: $seed, format: .number.grouping(.never)).kitField()
                    }
                    LabeledField("Priority") {
                        TextField("", value: $priority, format: .number.grouping(.never)).kitField()
                    }
                    if !editable {
                        LabeledField("Duration (seconds)") {
                            TextField("", text: $durationText, prompt: Text("Model default")).kitField()
                        }
                    }
                }
                }

                if showsAdvanced { advanced }
            }
            .padding(Theme.Space.s5)
            .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { formHeight = $0 }
            }
            // Testing: SHUTTER_TEST_SCROLL=bottom shows the end of the form (for screenshots).
            .defaultScrollAnchor(ProcessInfo.processInfo.environment["SHUTTER_TEST_SCROLL"] == "bottom" ? .bottom : .top)
            .frame(height: min(formHeight, 520))

            if let error {
                Text(error)
                    .font(Theme.Fonts.meta)
                    .foregroundStyle(Theme.Palette.danger)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
                    .padding(.horizontal, Theme.Space.s5)
                    .padding(.bottom, Theme.Space.s3)
            }

            Hairline()

            HStack(spacing: Theme.Space.s3) {
                KeyHint(keys: ["esc"], label: "Cancel")
                KeyHint(keys: ["⌘", "↩"], label: "Submit")
                Spacer(minLength: Theme.Space.s3)
                Button("Cancel") { dismiss() }
                    .buttonStyle(.kit(.line))
                    .keyboardShortcut(.cancelAction)
                Button("Send") { submit() }
                    .buttonStyle(.kit(.ink))
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(model.isEmpty || submitting)
            }
            .padding(.horizontal, Theme.Space.s5)
            .frame(height: Theme.Metric.toolbarHeight)
        }
        .frame(width: Theme.Metric.sheetWidth)
        .background(PaperBackground(glows: false))
        .tint(Theme.Palette.violet)
        .preferredColorScheme(.light)
        .onAppear {
            if let initialModel, !initialModel.isEmpty {
                model = initialModel
            } else if model.isEmpty, let first = modelChoices.first {
                model = first.id
            }
            adoptEngineMode()
        }
        .onChange(of: store.settings) { _, _ in
            if !modeTouched { adoptEngineMode() }
        }
        .task(id: model) { await loadOptions() }
        .task(id: EstimateKey(model: model, values: draft.values)) { await refreshEstimate() }
    }

    // MARK: Voice (Fish Audio Drama 3)

    private var voiceFields: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s4) {
            LabeledField("Text to speak") {
                TextField("", text: $prompt,
                          prompt: Text("What it should say. [Directions] in brackets set the tone, e.g. [calm, warm]."),
                          axis: .vertical)
                    .lineLimit(3...8)
                    .kitField(height: nil)
                    .frame(minHeight: Theme.Metric.promptHeight, alignment: .top)
            }
            HStack(alignment: .top, spacing: Theme.Space.s4) {
                LabeledField("Reference voice (optional)") {
                    PathField(path: $audioPath, placeholder: "~/Music/voice.wav", types: [.audio], title: "voice clip")
                }
                LabeledField("Takes") {
                    Stepper(value: $takes, in: 1...8) {
                        Text("\(takes)").monospacedDigit()
                    }
                    .help("Several takes of the same text with different seeds; pick the best one")
                }
                .frame(width: 120)
            }
            if !audioPath.isEmpty {
                LabeledField("What the clip says") {
                    TextField("", text: $refText, prompt: Text("The exact words spoken in the reference clip"),
                              axis: .vertical)
                        .lineLimit(1...3)
                        .kitField(height: nil)
                }
                Text("Only use the voice of someone who agreed to it.")
                    .font(Theme.Fonts.meta)
                    .foregroundStyle(Theme.Palette.muted)
            } else {
                Text("No clip: Drama 3 speaks in its own built-in voice.")
                    .font(Theme.Fonts.meta)
                    .foregroundStyle(Theme.Palette.muted)
            }
            Toggle("Split long text into short parts (steadier voice on long scripts)", isOn: $splitText)
                .toggleStyle(.checkbox)
                .font(Theme.Fonts.row)
            LabeledField("Seed") {
                TextField("", value: $seed, format: .number.grouping(.never)).kitField()
            }
            .frame(width: 160)
        }
    }

    // MARK: Image (Z-Image)

    private var imageFields: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s4) {
            LabeledField("Prompt") {
                TextField("", text: $prompt, prompt: Text("What the picture shows, e.g. an old man and his dog on a porch"),
                          axis: .vertical)
                    .lineLimit(3...6)
                    .kitField(height: nil)
                    .frame(minHeight: Theme.Metric.promptHeight, alignment: .top)
            }
            referenceFields
            HStack(alignment: .top, spacing: Theme.Space.s4) {
                LabeledField("Style") {
                    Picker("Style", selection: $style) {
                        Text("None").tag("")
                        ForEach(options?.styles ?? []) { s in Text(s.label).tag(s.id) }
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                    .frame(maxWidth: .infinity, minHeight: Theme.Metric.fieldHeight, alignment: .leading)
                    .help("Adds words for the look around your prompt")
                }
                LabeledField("Pictures") {
                    Stepper(value: $imageCount, in: 1...8) {
                        Text("\(imageCount)").monospacedDigit()
                    }
                    .help("Several pictures of the same prompt with seeds seed, seed + 1, …")
                    .frame(height: Theme.Metric.fieldHeight)
                }
                .frame(width: 120)
                LabeledField("Seed") {
                    TextField("", value: $seed, format: .number.grouping(.never)).kitField()
                }
                .frame(width: 140)
            }
            if options?.textureFix == true {
                VStack(alignment: .leading, spacing: Theme.Space.s1) {
                    Toggle("Cleaner texture (recommended)", isOn: $cleanTexture)
                        .disabled(!referencePaths.isEmpty)
                        .help("Fixes the grainy, crusty look skin, fur and stone can get. Same speed and memory.")
                    Text(referencePaths.isEmpty
                         ? "Smoother skin, fur and stone without losing detail. Turn off for the model's plain look."
                         : "Not used when you add reference pictures.")
                        .font(Theme.Fonts.meta)
                        .foregroundStyle(Theme.Palette.muted)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            LabeledField("Negative prompt (optional)") {
                TextField("", text: $negativePrompt, prompt: Text("Used only when guidance is above 1"))
                    .kitField()
            }
        }
    }

    /// Reference picture paths: the first always, each next one once the one before is filled.
    @ViewBuilder
    private var referenceFields: some View {
        let maxRefs = min(options?.references ?? 0, refPaths.count)
        if maxRefs > 0 {
            VStack(alignment: .leading, spacing: Theme.Space.s2) {
                ForEach(0..<maxRefs, id: \.self) { i in
                    if i == 0 || !refPaths[i - 1].isEmpty {
                        LabeledField(i == 0 ? "Reference pictures (optional)" : "Picture \(i + 1)") {
                            PathField(path: $refPaths[i], placeholder: "~/Pictures/picture-\(i + 1).jpg",
                                      types: [.image], title: "reference picture")
                        }
                    }
                }
                Text("Refer to them in the prompt as <Picture 1>, <Picture 2>, e.g. \"Put the woman from <Picture 1> in a rainy street\".")
                    .font(Theme.Fonts.meta)
                    .foregroundStyle(Theme.Palette.muted)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var referencePaths: [String] {
        let maxRefs = min(options?.references ?? 0, refPaths.count)
        return refPaths.prefix(maxRefs)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .map { ($0 as NSString).expandingTildeInPath }
    }

    @ViewBuilder
    private var loraFields: some View {
        if options?.lora == true {
            HStack(alignment: .top, spacing: Theme.Space.s4) {
                LabeledField("Style LoRA (optional)") {
                    PathField(path: $loraPath, placeholder: "~/Downloads/style-lora.safetensors",
                              types: [.data], title: "LoRA")
                }
                LabeledField("LoRA strength") {
                    HStack(spacing: Theme.Space.s2) {
                        Slider(value: $loraStrength, in: 0...2, step: 0.05)
                            .tint(Theme.Palette.violet)
                            .disabled(loraPath.isEmpty)
                        Text(RequestValues.text(loraStrength))
                            .font(Theme.Fonts.row)
                            .monospacedDigit()
                            .frame(width: 36, alignment: .trailing)
                    }
                    .frame(height: Theme.Metric.fieldHeight)
                }
                .frame(width: 200)
            }
        }
    }

    // MARK: Advanced

    private var advanced: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s3) {
            Hairline()
            HStack(spacing: Theme.Space.s2) {
                Button {
                    withAnimation(Motion.smooth) { advancedOpen.toggle() }
                } label: {
                    HStack(spacing: Theme.Space.s2) {
                        Image(systemName: "chevron.right")
                            .font(.system(size: 10, weight: .semibold))
                            .rotationEffect(.degrees(advancedOpen ? 90 : 0))
                        Text("Advanced").font(Theme.Fonts.rowMedium)
                    }
                    .foregroundStyle(Theme.Palette.ink)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(advancedOpen ? "Hide advanced settings" : "Show advanced settings")
                Spacer(minLength: Theme.Space.s3)
                if editable { EstimateLine(estimate: estimate) }
            }
            if editable { WarningLines(estimate: estimate) }

            if advancedOpen {
                VStack(alignment: .leading, spacing: Theme.Space.s4) {
                    if editable, let options {
                        RequestValueFields(draft: $draft, options: options, estimate: estimate)
                    }
                    loraFields
                    if showsMemoryStep { memoryStep }
                    if editable, let options {
                        HStack {
                            Spacer()
                            Button("Reset to defaults") { draft = RequestDraft(options.defaults, image: options.isImage) }
                                .buttonStyle(.kit(.line, small: true))
                                .disabled(draft == RequestDraft(options.defaults, image: options.isImage))
                                .help("Back to this model's defaults: \(options.defaults.summary)")
                        }
                    }
                }
                .transition(.opacity)
            }
        }
    }

    private static let modes = [KitOption(value: "fast", label: "Fast"),
                                KitOption(value: "low_memory", label: "Low memory")]

    @ViewBuilder
    private var memoryStep: some View {
        if let settings = store.settings {
            LabeledField("Memory step (this request)") {
                KitSegmented(options: Self.modes,
                             selection: Binding(get: { mode }, set: { mode = $0; modeTouched = true }))
                if mode == "low_memory", !settings.ladder.isEmpty {
                    StepSlider(count: settings.ladder.count,
                               index: Binding(get: { min(rungIndex, settings.ladder.count - 1) },
                                              set: { rungIndex = $0; modeTouched = true }),
                               label: { i in settings.ladder.indices.contains(i) ? settings.ladder[i].label : "" })
                    if settings.ladder.indices.contains(rungIndex) {
                        Text(settings.ladder[rungIndex].label)
                            .font(Theme.Fonts.meta)
                            .foregroundStyle(Theme.Palette.ink2)
                            .lineLimit(1)
                    }
                }
            }
        }
    }

    private func adoptEngineMode() {
        guard let s = store.settings else { return }
        mode = s.mode
        rungIndex = s.index(for: s.efficiency)
    }

    /// Prefills the Advanced fields from the chosen model's defaults. Older engines (404) and errors leave
    /// `options` nil, which hides the section.
    private func loadOptions() async {
        if options?.model != model {
            options = nil
            estimate = nil
        }
        guard !model.isEmpty else { return }
        let loaded = try? await APIClient.current().requestOptions(model: model)
        guard !Task.isCancelled, let loaded else { return }
        options = loaded
        draft = RequestDraft(loaded.defaults, image: loaded.isImage)
        cleanTexture = loaded.textureFixDefault
        estimate = loaded.estimate
        // Testing: SHUTTER_TEST_FRAMES=<n> and SHUTTER_TEST_UNTILED=1 change the fields (for screenshots).
        let env = ProcessInfo.processInfo.environment
        if let f = env["SHUTTER_TEST_FRAMES"] { draft.frames = f }
        if env["SHUTTER_TEST_UNTILED"] != nil { draft.vaeTile = false }
    }

    /// The estimate follows the fields, 300 ms after the last change.
    private func refreshEstimate() async {
        guard let options, options.editable, options.model == model else { return }
        if draft == RequestDraft(options.defaults, image: options.isImage), let e = options.estimate {
            estimate = e
            return
        }
        try? await Task.sleep(for: .milliseconds(300))
        guard !Task.isCancelled else { return }
        if let e = try? await APIClient.current().estimate(model: model, values: draft.values), !Task.isCancelled {
            estimate = e
        }
    }

    @ViewBuilder
    private var modelField: some View {
        if modelChoices.isEmpty {
            TextField("", text: $model, prompt: Text("longcat-avatar-1.6-8bit")).kitField()
        } else {
            Picker("Model", selection: $model) {
                ForEach(modelChoices) { m in
                    Text(m.installed ? m.id : "\(m.id) (not installed)").tag(m.id)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func submit() {
        let trimmedDuration = durationText.trimmingCharacters(in: .whitespaces)
        var duration: Double?
        if !trimmedDuration.isEmpty {
            guard let d = Double(trimmedDuration), d > 0 else {
                error = "Duration must be a number of seconds, or empty for the model default."
                return
            }
            duration = d
        }
        if editable, let problem = draft.problem {
            error = problem
            advancedOpen = true
            return
        }
        if isImage {
            if prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                error = "Write what the picture should show."
                return
            }
            if let missing = referencePaths.first(where: { !FileManager.default.fileExists(atPath: $0) }) {
                error = "Reference picture not found: \((missing as NSString).lastPathComponent)"
                return
            }
            if !loraPath.isEmpty && !FileManager.default.fileExists(atPath: (loraPath as NSString).expandingTildeInPath) {
                error = "The LoRA file was not found."
                advancedOpen = true
                return
            }
        }
        if isVoice {
            if prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                error = "Write the text to speak."
                return
            }
            if !audioPath.isEmpty && refText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                error = "Type what the reference clip says, so the voice can be copied."
                return
            }
        }
        var req = NewJobRequest(
            model: model,
            prompt: prompt.trimmingCharacters(in: .whitespacesAndNewlines),
            image: imagePath.isEmpty ? nil : (imagePath as NSString).expandingTildeInPath,
            audio: audioPath.isEmpty ? nil : (audioPath as NSString).expandingTildeInPath,
            output: nil,
            framing: framing,
            durationS: editable ? nil : duration,
            seed: seed,
            priority: priority
        )
        // The request carries exactly what the sheet shows.
        var params: [String: JSONValue] = [:]
        if isVoice {
            req.image = nil
            req.durationS = nil
            if !audioPath.isEmpty { params["ref_text"] = .string(refText.trimmingCharacters(in: .whitespacesAndNewlines)) }
            if takes > 1 { params["takes"] = .number(Double(takes)) }
            params["max_chars"] = .number(splitText ? 120 : 300)
        }
        if isImage {
            req.image = nil
            req.audio = nil
            req.durationS = nil
            let neg = negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines)
            if !neg.isEmpty { req.negativePrompt = neg }
            if imageCount > 1 { params["count"] = .number(Double(imageCount)) }
            if !style.isEmpty { params["style"] = .string(style) }
            let refs = referencePaths
            if !refs.isEmpty { params["images"] = .array(refs.map { .string($0) }) }
            if options?.textureFix == true && refs.isEmpty { params["texture_fix"] = .bool(cleanTexture) }
            if !loraPath.isEmpty && loraStrength != 0 {
                params["loras"] = .array([.object(["path": .string((loraPath as NSString).expandingTildeInPath),
                                                   "strength": .number(loraStrength)])])
            }
            if editable {
                let v = draft.values
                req.width = v.width
                req.height = v.height
                req.steps = v.steps
                req.guidance = v.guidance
            }
        } else if editable && !isVoice {
            let v = draft.values
            req.width = v.width
            req.height = v.height
            req.frames = v.frames
            req.steps = v.steps
            req.guidance = v.guidance
            if let shift = v.shift { params["shift"] = .number(shift) }
            params["vae_tile"] = .bool(draft.vaeTile)
        }
        if showsMemoryStep, let settings = store.settings {
            req.mode = mode
            if mode == "low_memory", settings.ladder.indices.contains(rungIndex) {
                params["efficiency"] = .number(settings.ladder[rungIndex].at)
            }
        }
        if !params.isEmpty { req.params = params }
        submitting = true
        error = nil
        Task { @MainActor in
            let ok = await store.submit(req)
            submitting = false
            if ok {
                dismiss()
            } else {
                error = store.lastError ?? "Could not send the request."
                store.lastError = nil
            }
        }
    }
}

/// 12 pt label above a control.
struct LabeledField<Content: View>: View {
    let label: String
    @ViewBuilder var content: () -> Content

    init(_ label: String, @ViewBuilder content: @escaping () -> Content) {
        self.label = label
        self.content = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s2) {
            Text(label).font(Theme.Fonts.meta).foregroundStyle(Theme.Palette.muted)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Editable path with a "Choose…" button that opens a file panel.
struct PathField: View {
    @Binding var path: String
    let placeholder: String
    let types: [UTType]
    let title: String

    var body: some View {
        HStack(spacing: Theme.Space.s2) {
            TextField("", text: $path, prompt: Text(placeholder))
                .kitField()
                .help(path)
            Button("Choose…") { choose() }
                .buttonStyle(.kit(.line, small: true))
        }
    }

    @MainActor
    private func choose() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = types
        panel.message = "Choose the \(title) file"
        if panel.runModal() == .OK, let url = panel.url {
            path = url.path
        }
    }
}
