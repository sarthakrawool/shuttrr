import AppKit
import SwiftUI

extension Job {
    /// Every picture this request wrote (the first, then the others), for image models.
    var imageOutputs: [String] {
        let list = (result["outputs"]?.arrayValue ?? []).compactMap(\.stringValue)
        let all = list.isEmpty ? [outputPath].compactMap { $0 } : list
        return all.filter { ["png", "jpg", "jpeg", "webp"].contains(URL(fileURLWithPath: $0).pathExtension.lowercased()) }
    }
}

/// A small thumbnail of a finished image request; click for a larger preview with Open and Reveal in Finder.
struct ImageThumbButton: View {
    let files: [String]
    @State private var showing = false

    var body: some View {
        Button { showing.toggle() } label: {
            Thumb(path: files.first ?? "", side: 26)
        }
        .buttonStyle(.plain)
        .help(files.count > 1 ? "Show the \(files.count) pictures" : "Show the picture")
        .accessibilityLabel(files.count > 1 ? "Show the \(files.count) pictures" : "Show the picture")
        .popover(isPresented: $showing, arrowEdge: .leading) {
            ImagePreviewPanel(files: files)
        }
    }
}

/// The pictures of one request, large, each with Open and Reveal in Finder.
struct ImagePreviewPanel: View {
    let files: [String]
    @State private var index = 0

    var body: some View {
        let path = files.indices.contains(index) ? files[index] : (files.first ?? "")
        VStack(alignment: .leading, spacing: Theme.Space.s3) {
            Thumb(path: path, side: 420)
                .onTapGesture(count: 2) { NSWorkspace.shared.open(URL(fileURLWithPath: path)) }
            if files.count > 1 {
                HStack(spacing: Theme.Space.s2) {
                    ForEach(Array(files.enumerated()), id: \.offset) { i, f in
                        Button { index = i } label: {
                            Thumb(path: f, side: 44)
                                .overlay(RoundedRectangle(cornerRadius: 6)
                                    .strokeBorder(i == index ? Theme.Palette.violet : Color.clear, lineWidth: 2))
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel("Picture \(i + 1)")
                    }
                }
            }
            HStack(spacing: Theme.Space.s2) {
                Text(URL(fileURLWithPath: path).lastPathComponent)
                    .font(Theme.Fonts.meta)
                    .foregroundStyle(Theme.Palette.muted)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer(minLength: Theme.Space.s3)
                Button("Open") { NSWorkspace.shared.open(URL(fileURLWithPath: path)) }
                    .buttonStyle(.kit(.line, small: true))
                Button("Reveal in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                }
                .buttonStyle(.kit(.ink, small: true))
            }
        }
        .padding(Theme.Space.s4)
        .frame(width: 452)
        .background(PaperBackground(glows: false))
        .preferredColorScheme(.light)
    }
}

/// A square, aspect-fit image read from disk (a quiet placeholder when it cannot be read).
struct Thumb: View {
    let path: String
    let side: CGFloat

    var body: some View {
        Group {
            if let img = NSImage(contentsOfFile: path) {
                Image(nsImage: img).resizable().interpolation(.high).aspectRatio(contentMode: .fit)
            } else {
                Image(systemName: "photo").foregroundStyle(Theme.Palette.muted)
            }
        }
        .frame(width: side, height: side)
        .background(Theme.Palette.white.opacity(0.6))
        .clipShape(RoundedRectangle(cornerRadius: side > 100 ? 10 : 5))
    }
}
