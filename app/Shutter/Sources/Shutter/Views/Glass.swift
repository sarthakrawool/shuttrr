import AppKit
import SwiftUI

// Paper cards and menu rows, plus the one real glass surface: the menu bar panel's background.

/// An `NSVisualEffectView`. Used only behind the menu bar panel, always in the light appearance
/// so the brand's ink colours stay readable whatever the system appearance is.
struct VisualEffectView: NSViewRepresentable {
    var material: NSVisualEffectView.Material = .popover
    var blending: NSVisualEffectView.BlendingMode = .behindWindow

    func makeNSView(context: Context) -> NSVisualEffectView {
        let view = NSVisualEffectView()
        view.state = .active
        view.material = material
        view.blendingMode = blending
        view.appearance = NSAppearance(named: .aqua)
        return view
    }

    func updateNSView(_ view: NSVisualEffectView, context: Context) {
        view.material = material
        view.blendingMode = blending
    }
}

struct GlassCard: ViewModifier {
    var padding: CGFloat
    var radius: CGFloat

    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: radius, style: .continuous)
        content
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(padding)
            .background(shape.fill(Theme.Palette.glass))
            .overlay(shape.strokeBorder(Theme.Palette.line, lineWidth: Theme.Metric.hairline))
            .shadow(color: Theme.Palette.glassShadow, radius: 12, x: 0, y: 6)
    }
}

extension View {
    /// A paper card: white, hairline edge, soft drop shadow.
    func glassCard(padding: CGFloat = Theme.Space.s4, radius: CGFloat = Theme.Radius.card) -> some View {
        modifier(GlassCard(padding: padding, radius: radius))
    }
}

/// A row in the menu bar panel, like the rows in the system's Wi-Fi or Sound panels:
/// full-width, highlights on hover, optional symbol and shortcut hint.
struct MenuRow: View {
    let title: String
    var symbol: String?
    var shortcut: String?
    var role: ButtonRole?
    let action: () -> Void

    @State private var hovering = false
    @Environment(\.isEnabled) private var isEnabled

    var body: some View {
        Button(role: role, action: action) {
            HStack(spacing: Theme.Space.s2) {
                if let symbol {
                    Image(systemName: symbol)
                        .font(.system(size: Theme.Size.row))
                        .foregroundStyle(Theme.Palette.muted)
                        .frame(width: Theme.Metric.icon + Theme.Space.s1)
                }
                Text(title)
                    .font(Theme.Fonts.row)
                    .foregroundStyle(Theme.Palette.ink)
                Spacer(minLength: Theme.Space.s3)
                if let shortcut {
                    Text(shortcut)
                        .font(Theme.Fonts.mono)
                        .foregroundStyle(Theme.Palette.faint)
                }
            }
            .padding(.horizontal, Theme.Space.s2)
            .frame(height: Theme.Metric.smallButtonHeight + Theme.Space.s1)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.small, style: .continuous)
                    .fill(hovering && isEnabled ? Theme.Palette.fill : Color.clear)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .opacity(isEnabled ? 1 : 0.4)
        .animation(Motion.quick, value: hovering)
        .onHover { hovering = $0 }
    }
}

/// Small uppercase mono label, like the website's kickers.
struct ModuleLabel: View {
    let text: String
    var body: some View {
        Text(text.uppercased())
            .font(Theme.Fonts.label)
            .tracking(1.2)
            .foregroundStyle(Theme.Palette.muted)
    }
}
