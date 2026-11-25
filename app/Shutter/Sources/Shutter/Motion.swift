import SwiftUI

/// One motion language for the whole app: three springs and two eases, so everything moves alike.
/// Every helper here goes still when the system's Reduce Motion is on.
enum Motion {
    /// Selection moving between options: quick, no wobble.
    static let snappy = Animation.spring(response: 0.28, dampingFraction: 0.86)
    /// Things that grow, shrink or arrive: bars, sections, numbers.
    static let smooth = Animation.spring(response: 0.45, dampingFraction: 0.88)
    /// A touch of life for presses and toggles.
    static let lively = Animation.spring(response: 0.32, dampingFraction: 0.68)
    /// Page changes and fades.
    static let gentle = Animation.easeOut(duration: 0.22)
    /// Hover feedback.
    static let quick = Animation.easeOut(duration: 0.12)
}

/// Fades a view in while it rises a few points, later for a higher `index`, so a panel
/// assembles from top to bottom instead of popping in.
private struct Rise: ViewModifier {
    let index: Int
    @State private var shown = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    func body(content: Content) -> some View {
        content
            .opacity(shown ? 1 : 0)
            .offset(y: shown || reduceMotion ? 0 : 8)
            .onAppear {
                let animation = reduceMotion ? Animation.easeOut(duration: 0.15)
                                             : Motion.smooth.delay(Double(index) * 0.04)
                withAnimation(animation) { shown = true }
            }
            .onDisappear { shown = false }
    }
}

extension View {
    /// Fade and rise in on appear; `index` staggers siblings.
    func rise(_ index: Int = 0) -> some View { modifier(Rise(index: index)) }

    /// Standard arrival and departure for a section that comes and goes.
    func sectionTransition() -> some View {
        transition(.opacity.combined(with: .move(edge: .top)))
    }
}

/// A status dot that sends out a soft ring while `active` (the engine is working).
struct PulseDot: View {
    let color: Color
    var active = false
    var size: CGFloat = 6

    @State private var pulsing = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        Circle()
            .fill(color)
            .frame(width: size, height: size)
            .background(
                Circle()
                    .fill(color.opacity(0.4))
                    .scaleEffect(pulsing ? 2.8 : 1)
                    .opacity(pulsing ? 0 : (active ? 0.9 : 0))
            )
            .animation(Motion.gentle, value: color)
            .onAppear { update() }
            .onChange(of: active) { _, _ in update() }
    }

    private func update() {
        if active && !reduceMotion {
            withAnimation(.easeOut(duration: 1.5).repeatForever(autoreverses: false)) { pulsing = true }
        } else {
            withAnimation(Motion.quick) { pulsing = false }
        }
    }
}

/// Lifts a card slightly while the pointer is over it.
private struct HoverLift: ViewModifier {
    @State private var hovering = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    func body(content: Content) -> some View {
        content
            .offset(y: hovering && !reduceMotion ? -2 : 0)
            .animation(Motion.lively, value: hovering)
            .onHover { hovering = $0 }
    }
}

extension View {
    func hoverLift() -> some View { modifier(HoverLift()) }
}
