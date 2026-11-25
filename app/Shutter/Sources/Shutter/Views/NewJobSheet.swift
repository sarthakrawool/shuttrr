import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// Sheet for POST /v1/jobs: model, prompt, image path, audio path, framing (Full / Small /
/// Auto), seed, priority, and an optional duration. Same fields and wording as the dashboard.
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
    @State private var submitting = false
    @State private var error: String?

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

            VStack(alignment: .leading, spacing: Theme.Space.s4) {
                LabeledField("Model") { modelField }

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
                    LabeledField("Duration (seconds)") {
                        TextField("", text: $durationText, prompt: Text("Model default")).kitField()
                    }
                }
            }
            .padding(Theme.Space.s5)

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
        let req = NewJobRequest(
            model: model,
            prompt: prompt.trimmingCharacters(in: .whitespacesAndNewlines),
            image: imagePath.isEmpty ? nil : (imagePath as NSString).expandingTildeInPath,
            audio: audioPath.isEmpty ? nil : (audioPath as NSString).expandingTildeInPath,
            output: nil,
            framing: framing,
            durationS: duration,
            seed: seed,
            priority: priority
        )
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
