import AppKit
import SwiftUI

/// Design tokens for the personal-brand look (the same as the website in Agency/website/light-hero):
/// playful type on warm paper, flat and clean. The web dashboard uses the same values.
///
/// - Fraunces for headlines and the one big number, DM Sans for text, JetBrains Mono for numbers and labels.
/// - Surfaces are paper and white cards. Ink for selection and primary buttons; the violet-to-amber
///   gradient for progress and memory; green, amber and red only for meaning.
/// - Spacing comes from the 4 pt scale below.
enum Theme {

    // MARK: Type

    enum Size {
        static let meta: CGFloat = 12      // metadata, captions, table headers
        static let row: CGFloat = 13       // rows, body, buttons
        static let section: CGFloat = 17   // section headers (Fraunces)
        static let title: CGFloat = 28     // the one title or big number per screen (Fraunces)
    }

    /// Fraunces for headlines and the big number, DM Sans for text, JetBrains Mono for numbers and labels.
    enum Fonts {
        static let row = Brand.sans(Size.row)
        static let rowMedium = Brand.sans(Size.row, weight: 600)
        static let button = Brand.sans(Size.row, weight: 600)
        static let meta = Brand.sans(Size.meta)
        static let metaMedium = Brand.sans(Size.meta, weight: 500)
        static let metaSemibold = Brand.sans(Size.meta, weight: 600)
        static let section = Brand.display(Size.section, weight: 600)
        static let sectionStrong = Brand.display(Size.section, weight: 600)
        static let title = Brand.display(Size.title, weight: 500)
        static let big = Brand.display(Size.title + 6, weight: 500)
        static let mono = Brand.mono(Size.meta)
        static let monoMedium = Brand.mono(Size.meta, weight: 500)
        /// Small uppercase label ("kicker"), always with tracking.
        static let label = Brand.mono(11, weight: 500)
        static let brand = Brand.display(20, weight: 600)
    }

    // MARK: Spacing (4 pt scale: --space-1 … --space-8)

    enum Space {
        static let s0: CGFloat = 2      // sidebar row gap, tight label stacks
        static let s1: CGFloat = 4
        static let s2: CGFloat = 8
        static let s3: CGFloat = 12
        static let s4: CGFloat = 16
        static let s5: CGFloat = 24
        static let s6: CGFloat = 32
        static let s7: CGFloat = 48
        static let s8: CGFloat = 64
    }

    // MARK: Metrics

    enum Metric {
        static let sidebarWidth: CGFloat = 216
        static let sidebarRowHeight: CGFloat = 30
        static let toolbarHeight: CGFloat = 64
        static let rowHeight: CGFloat = 48           // table / list rows
        static let tableHeaderHeight: CGFloat = 28
        static let kvRowHeight: CGFloat = 40         // key/value rows in groups
        static let statusBarHeight: CGFloat = 28
        static let buttonHeight: CGFloat = 30
        static let smallButtonHeight: CGFloat = 22
        static let segmentHeight: CGFloat = 28
        static let searchWidth: CGFloat = 280
        static let searchHeight: CGFloat = 32
        static let fieldHeight: CGFloat = 32
        static let promptHeight: CGFloat = 72
        static let keyCap: CGFloat = 20
        static let badgeHeight: CGFloat = 20
        static let countHeight: CGFloat = 20
        static let brandIcon: CGFloat = 24
        static let icon: CGFloat = 16
        static let bannerIcon: CGFloat = 40
        static let progressWidth: CGFloat = 48
        static let progressHeight: CGFloat = 4
        static let progressHeightWide: CGFloat = 6
        static let memoryBarHeight: CGFloat = 8
        static let legendKey: CGFloat = 8
        static let dot: CGFloat = 8
        static let sheetWidth: CGFloat = 560
        static let panelWidth: CGFloat = 320
        static let kvLabelWidth: CGFloat = 200
        static let chartHeight: CGFloat = 192
        static let minWindowWidth: CGFloat = 900
        static let minWindowHeight: CGFloat = 560
        static let hairline: CGFloat = 1
        static let dash: [CGFloat] = [4, 3]

        // queue columns
        static let colModel: CGFloat = 120
        static let colStatus: CGFloat = 190
        static let colTime: CGFloat = 56
        static let colPeak: CGFloat = 72
        static let colAction: CGFloat = 76
        // models columns
        static let colState: CGFloat = 96
        static let colFamily: CGFloat = 140
        static let colModelActions: CGFloat = 148
    }

    enum Radius {
        static let legend: CGFloat = 2       // legend keys
        static let key: CGFloat = 5          // key caps
        static let small: CGFloat = 8        // small buttons, segments, key caps
        static let control: CGFloat = 10      // fields, search, sidebar rows
        static let card: CGFloat = 18
        static let window: CGFloat = 20      // windows, sheets, banners
        static let pill: CGFloat = 7         // regular buttons (macOS large control)
    }

    // MARK: Colour
    //
    // The personal-brand palette: warm paper, near-black ink, lamp amber and monitor violet.
    // Light only, like the website. No pink.

    enum Palette {
        static let paper = Color(hex: 0xF4F4F2)
        static let white = Color(hex: 0xFFFFFF)
        static let window = Color.clear
        static let chrome = Color(hex: 0x0B0B0C, alpha: 0.04)
        static let sidebar = Color.clear
        /// Quiet control / track fill on paper.
        static let fill = Color(hex: 0x0B0B0C, alpha: 0.06)
        static let fillStrong = Color(hex: 0x0B0B0C, alpha: 0.12)
        static let ink = Color(hex: 0x0B0B0C)
        static let ink2 = Color(hex: 0x4F4F54)
        static let muted = Color(hex: 0x6E6E73)
        static let faint = Color(hex: 0x8A8A8F)
        static let line = Color(hex: 0x0B0B0C, alpha: 0.10)
        static let lineStrong = Color(hex: 0x0B0B0C, alpha: 0.22)
        static let amber = Color(hex: 0xF2A93B)
        static let amberSoft = Color(hex: 0xFFE3B0)
        static let violet = Color(hex: 0x9C82E0)
        static let violetSoft = Color(hex: 0xDCD0FA)
        /// Selection and the primary button are ink, as on the website; colour is for accents.
        static let accent = Color(hex: 0x0B0B0C)
        static let accentSoft = Color(hex: 0xDCD0FA)
        static let onAccent = Color.white
        /// Meaning only: running / ready.
        static let good = Color(hex: 0x5E7A45)
        /// Meaning only: waiting on memory / paused.
        static let warn = Color(hex: 0xA8660A)
        /// Meaning only: failed / error.
        static let danger = Color(hex: 0xC2412D)
        static let banner = Color(hex: 0xFFE3B0, alpha: 0.7)
        static let bannerIcon = Color(hex: 0xA8660A)
        static let bannerInk = Color(hex: 0x4F4F54)
        static let bannerButtonInk = Color(hex: 0x0B0B0C)
        static let scrim = Color.black.opacity(0.18)
        /// Cards: white paper with a hairline and a soft drop shadow.
        static let glass = Color.white
        static let glassEdge = Color.white
        static let glassShadow = Color(hex: 0x0B0B0C, alpha: 0.08)
        static let content = Color.clear
        static let sidebarVeil = Color(hex: 0xFFFFFF, alpha: 0.55)
        static let card = Color.white
    }

    /// Hierarchical styles for text inside selectable rows: they turn white on the accent.
    static func rowTitle(selected: Bool) -> Color { selected ? Palette.onAccent : Palette.ink }
    static func rowMeta(selected: Bool) -> Color { selected ? Palette.onAccent.opacity(0.85) : Palette.muted }
}

// MARK: - Colour helpers

extension Color {
    init(hex: UInt32, alpha: Double = 1) {
        self.init(.sRGB,
                  red: Double((hex >> 16) & 0xFF) / 255,
                  green: Double((hex >> 8) & 0xFF) / 255,
                  blue: Double(hex & 0xFF) / 255,
                  opacity: alpha)
    }

    /// A colour that follows the window's appearance (light / dark).
    static func dynamic(light: UInt32, dark: UInt32, lightAlpha: CGFloat = 1, darkAlpha: CGFloat = 1) -> Color {
        Color(nsColor: NSColor(name: nil) { appearance in
            let isDark = appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
            return NSColor.fromHex(isDark ? dark : light, alpha: isDark ? darkAlpha : lightAlpha)
        })
    }
}

extension NSColor {
    static func fromHex(_ hex: UInt32, alpha: CGFloat = 1) -> NSColor {
        NSColor(srgbRed: CGFloat((hex >> 16) & 0xFF) / 255,
                green: CGFloat((hex >> 8) & 0xFF) / 255,
                blue: CGFloat(hex & 0xFF) / 255,
                alpha: alpha)
    }
}
