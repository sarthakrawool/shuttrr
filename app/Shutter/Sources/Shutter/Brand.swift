import AppKit
import CoreText
import SwiftUI

/// The personal-brand look: Fraunces (soft, wonky) for headlines, DM Sans for text, JetBrains Mono for
/// numbers and labels, on warm paper with amber and violet accents. Fonts and the paper texture ship in
/// the app bundle (Contents/Resources/Brand); if they are missing the system fonts and a flat paper
/// colour are used, so the app never breaks.
enum Brand {
    private static let registered: Bool = {
        guard let dir = resourceDirectory,
              let files = try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else { return false }
        var any = false
        for url in files where ["ttf", "otf"].contains(url.pathExtension.lowercased()) {
            if CTFontManagerRegisterFontsForURL(url as CFURL, .process, nil) { any = true }
        }
        return any
    }()

    static var resourceDirectory: URL? {
        var candidates: [URL] = []
        if let r = Bundle.main.resourceURL { candidates.append(r.appendingPathComponent("Brand")) }
        #if DEBUG
        // `swift run` from the package: Sources/Shutter/../../Resources/Brand.
        // Debug builds only, so a release binary carries no build-machine path.
        candidates.append(URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Resources/Brand"))
        #endif
        return candidates.first { FileManager.default.fileExists(atPath: $0.path) }
    }

    private static func tag(_ s: String) -> UInt32 {
        s.utf8.reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
    }

    private static func font(family: String, size: CGFloat, axes: [String: CGFloat], fallback: Font) -> Font {
        _ = registered
        var variation: [NSNumber: NSNumber] = [:]
        for (name, value) in axes { variation[NSNumber(value: tag(name))] = NSNumber(value: Double(value)) }
        let attributes: [NSFontDescriptor.AttributeName: Any] = [.family: family, .variation: variation]
        let descriptor = NSFontDescriptor(fontAttributes: attributes)
        guard let font = NSFont(descriptor: descriptor, size: size), font.familyName == family else { return fallback }
        return Font(font as CTFont)
    }

    /// Fraunces, soft and wonky: headlines and the big number.
    static func display(_ size: CGFloat, weight: CGFloat = 500) -> Font {
        font(family: "Fraunces", size: size,
             axes: ["wght": weight, "SOFT": 100, "WONK": 1, "opsz": min(max(size, 9), 144)],
             fallback: .system(size: size, weight: weight >= 600 ? .semibold : .medium, design: .serif))
    }

    /// DM Sans: everything you read.
    static func sans(_ size: CGFloat, weight: CGFloat = 400) -> Font {
        font(family: "DM Sans", size: size, axes: ["wght": weight, "opsz": min(max(size, 9), 40)],
             fallback: .system(size: size, weight: weight >= 600 ? .semibold : (weight >= 500 ? .medium : .regular)))
    }

    /// JetBrains Mono: numbers and small labels.
    static func mono(_ size: CGFloat, weight: CGFloat = 400) -> Font {
        font(family: "JetBrains Mono", size: size, axes: ["wght": weight],
             fallback: .system(size: size, weight: weight >= 500 ? .medium : .regular, design: .monospaced))
    }

    static let paperImage: NSImage? = {
        guard let url = resourceDirectory?.appendingPathComponent("paper-light.jpg"),
              let image = NSImage(contentsOf: url) else { return nil }
        image.size = NSSize(width: 720, height: 720)      // same tile size as the website
        return image
    }()

    /// Violet to amber, the brand's one gradient: progress, memory in use, the mark.
    static let gradient = LinearGradient(colors: [Theme.Palette.violet, Theme.Palette.amber],
                                         startPoint: .leading, endPoint: .trailing)
}

/// Warm paper: the off-white base, the paper texture, and (optionally) the two soft lamp glows.
struct PaperBackground: View {
    var glows = true

    var body: some View {
        ZStack {
            Theme.Palette.paper
            if let image = Brand.paperImage {
                Image(nsImage: image).resizable(resizingMode: .tile).opacity(0.7)
            }
            if glows {
                GeometryReader { geo in
                    let side = max(min(geo.size.width, geo.size.height) * 0.9, 260)
                    ZStack {
                        Circle().fill(RadialGradient(colors: [Theme.Palette.amberSoft, Theme.Palette.amber.opacity(0.28), .clear],
                                                     center: .center, startRadius: 0, endRadius: side * 0.5))
                            .frame(width: side, height: side)
                            .position(x: geo.size.width + side * 0.12, y: geo.size.height + side * 0.1)
                        Circle().fill(RadialGradient(colors: [Theme.Palette.violetSoft, Theme.Palette.violet.opacity(0.22), .clear],
                                                     center: .center, startRadius: 0, endRadius: side * 0.45))
                            .frame(width: side * 0.9, height: side * 0.9)
                            .position(x: -side * 0.12, y: side * 0.25)
                    }
                    .blur(radius: 50)
                    .opacity(0.75)
                }
                .allowsHitTesting(false)
            }
        }
        .ignoresSafeArea()
    }
}

/// Makes the hosting window one flat sheet of paper: no title bar strip, content under it.
struct PaperWindow: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async { Self.configure(view.window) }
        return view
    }

    func updateNSView(_ view: NSView, context: Context) {
        DispatchQueue.main.async { Self.configure(view.window) }
    }

    private static func configure(_ window: NSWindow?) {
        guard let window else { return }
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.styleMask.insert(.fullSizeContentView)
        window.isMovableByWindowBackground = true
        window.isOpaque = true
        window.backgroundColor = NSColor.fromHex(0xF4F4F2)
        window.appearance = NSAppearance(named: .aqua)      // the brand theme is light
    }
}
