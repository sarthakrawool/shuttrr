import SwiftUI

/// The brand slider for a few fixed steps (the memory ladder): a thick rounded track filled in ink up to the head,
/// a dot for every step, and a round white head that follows the finger and springs to the nearest step on release.
/// Clicking the track jumps (with the same spring); arrow keys move one step; VoiceOver reads `label(index)`.
struct StepSlider: View {
    let count: Int
    @Binding var index: Int
    var label: (Int) -> String = { "\($0 + 1)" }
    /// Called while dragging with the step under the finger (for live previews); `index` is set on release.
    var onPreview: ((Int) -> Void)? = nil

    static let track: CGFloat = 8
    static let head: CGFloat = 22

    @State private var drag: CGFloat?          // head x while dragging
    @Environment(\.isEnabled) private var isEnabled
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @FocusState private var focused: Bool

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let usable = max(1, w - Self.head)
            let stepX: (Int) -> CGFloat = { i in
                Self.head / 2 + usable * (count > 1 ? CGFloat(i) / CGFloat(count - 1) : 0)
            }
            let x = drag ?? stepX(index)
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(Theme.Palette.fillStrong)
                    .frame(height: Self.track)
                Capsule()
                    .fill(Theme.Palette.ink)
                    .frame(width: max(Self.track, x + Self.track / 2), height: Self.track)
                ForEach(0..<max(count, 1), id: \.self) { i in
                    Circle()
                        .fill(stepX(i) <= x ? Theme.Palette.white.opacity(0.75) : Theme.Palette.ink.opacity(0.28))
                        .frame(width: 4, height: 4)
                        .position(x: stepX(i), y: geo.size.height / 2)
                }
                Circle()
                    .fill(Theme.Palette.white)
                    .overlay(Circle().strokeBorder(Theme.Palette.ink, lineWidth: 1))
                    .shadow(color: Theme.Palette.ink.opacity(0.18), radius: 4, y: 2)
                    .frame(width: Self.head, height: Self.head)
                    .scaleEffect(drag != nil ? 1.08 : 1)
                    .position(x: x, y: geo.size.height / 2)
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { g in
                        let nx = min(max(g.location.x, Self.head / 2), w - Self.head / 2)
                        if drag == nil {
                            // a click on the track: let the spring carry the head there
                            withAnimation(reduceMotion ? nil : Motion.snappy) { drag = nx }
                        } else {
                            drag = nx
                        }
                        onPreview?(nearest(nx, stepX))
                    }
                    .onEnded { g in
                        let nx = min(max(g.location.x, Self.head / 2), w - Self.head / 2)
                        let i = nearest(nx, stepX)
                        withAnimation(reduceMotion ? nil : Motion.smooth) {
                            drag = nil
                            index = i
                        }
                    }
            )
            .animation(reduceMotion ? nil : Motion.smooth, value: index)
        }
        .frame(height: Self.head + 4)
        .opacity(isEnabled ? 1 : 0.4)
        .focusable()
        .focusEffectDisabled()      // no blue focus rectangle; arrow keys still move the step
        .focused($focused)
        .onMoveCommand { dir in
            switch dir {
            case .left, .down: step(-1)
            case .right, .up: step(1)
            default: break
            }
        }
        .accessibilityElement()
        .accessibilityLabel("Memory step")
        .accessibilityValue(label(index))
        .accessibilityAdjustableAction { dir in
            switch dir {
            case .increment: step(1)
            case .decrement: step(-1)
            @unknown default: break
            }
        }
    }

    private func nearest(_ x: CGFloat, _ stepX: (Int) -> CGFloat) -> Int {
        (0..<max(count, 1)).min { abs(stepX($0) - x) < abs(stepX($1) - x) } ?? 0
    }

    private func step(_ d: Int) {
        let i = min(max(index + d, 0), max(count - 1, 0))
        guard i != index else { return }
        withAnimation(reduceMotion ? nil : Motion.smooth) { index = i }
    }
}
