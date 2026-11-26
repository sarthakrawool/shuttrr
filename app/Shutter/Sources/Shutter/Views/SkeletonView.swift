import SwiftUI

/// Placeholders shown while the engine starts (0.2.12): soft grey blocks in the shape of the page that is coming,
/// with a slow light sweep across them. With Reduce Motion on they stay still. The dashboard swaps them for the real
/// page as soon as the engine answers (EngineStore.isStarting).
struct SkeletonBlock: View {
    var width: CGFloat? = nil
    var height: CGFloat = 12
    var radius: CGFloat = 6

    var body: some View {
        RoundedRectangle(cornerRadius: radius, style: .continuous)
            .fill(Color(hex: 0x0B0B0C, alpha: 0.08))
            .frame(width: width, height: height)
            .modifier(Shimmer())
            .clipShape(RoundedRectangle(cornerRadius: radius, style: .continuous))
    }
}

/// A light band that sweeps left to right every 1.6 s, clipped to the block.
struct Shimmer: ViewModifier {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var phase: CGFloat = -1

    func body(content: Content) -> some View {
        if reduceMotion {
            content
        } else {
            content
                .overlay {
                    GeometryReader { geo in
                        LinearGradient(colors: [.clear, Color.white.opacity(0.55), .clear],
                                       startPoint: .leading, endPoint: .trailing)
                            .frame(width: geo.size.width * 0.6)
                            .offset(x: phase * geo.size.width * 1.6)
                    }
                    .allowsHitTesting(false)
                }
                .onAppear {
                    withAnimation(.linear(duration: 1.6).repeatForever(autoreverses: false)) { phase = 1 }
                }
        }
    }
}

/// The page while the engine starts: a title, a "Starting engine…" line, and rows shaped like the section's.
struct EngineStartingView: View {
    let section: DashboardSection
    let detail: String

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s5) {
            HStack(alignment: .center) {
                VStack(alignment: .leading, spacing: Theme.Space.s2) {
                    Text(section.rawValue)
                        .font(Theme.Fonts.title)
                        .foregroundStyle(Theme.Palette.ink)
                    HStack(spacing: Theme.Space.s2) {
                        ProgressView().controlSize(.small)
                        Text(detail)
                            .font(Theme.Fonts.meta)
                            .foregroundStyle(Theme.Palette.muted)
                    }
                    .accessibilityElement(children: .combine)
                }
                Spacer()
                SkeletonBlock(width: 120, height: 32, radius: 16)
            }
            switch section {
            case .models:
                card(rows: 1, title: 90)
                card(rows: 3, title: 70)
            case .queue:
                SkeletonBlock(height: 8, radius: 4)
                table(rows: 6)
            case .memory:
                SkeletonBlock(height: 160, radius: Theme.Radius.card)
                table(rows: 3)
            default:
                card(rows: 4, title: 80)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, Theme.Space.s6)
        .padding(.top, Theme.Space.s5)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .accessibilityLabel("Starting the engine")
    }

    private func card(rows: Int, title: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.s3) {
            SkeletonBlock(width: title, height: 14)
            VStack(spacing: 0) {
                ForEach(0..<rows, id: \.self) { i in row(i) }
            }
        }
    }

    private func table(rows: Int) -> some View {
        VStack(spacing: 0) {
            ForEach(0..<rows, id: \.self) { i in row(i) }
        }
    }

    private func row(_ i: Int) -> some View {
        HStack(spacing: Theme.Space.s4) {
            VStack(alignment: .leading, spacing: 6) {
                SkeletonBlock(width: [220, 180, 240, 160, 200, 190][i % 6], height: 12)
                SkeletonBlock(width: [120, 90, 140, 100, 110, 80][i % 6], height: 9)
            }
            Spacer()
            SkeletonBlock(width: 70, height: 20, radius: 10)
            SkeletonBlock(width: 44, height: 12)
        }
        .frame(height: Theme.Metric.rowHeight)
        .overlay(alignment: .bottom) { Hairline() }
    }
}
