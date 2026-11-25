// Draws Shuttrr's app icon: the aperture mark (same structure as the header mark: a ring with six blades around a
// hexagon) in white on a black macOS rounded square. Drawn with Core Graphics, so no SF Symbol is used in the icon
// (Apple's SF Symbols licence does not allow symbols in app icons).
// usage: swift make_icon.swift <out.iconset dir> <preview_1024.png>
import AppKit
import CoreGraphics

func render(_ px: Int) -> CGImage {
    let s = CGFloat(px)
    let ctx = CGContext(data: nil, width: px, height: px, bitsPerComponent: 8, bytesPerRow: 0,
                        space: CGColorSpace(name: CGColorSpace.sRGB)!,
                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.clear(CGRect(x: 0, y: 0, width: s, height: s))
    // macOS icon grid: 824/1024 body, 100/1024 margin, ~185/1024 corner radius
    let body = s * 824 / 1024, inset = (s - body) / 2, radius = s * 185 / 1024
    let rect = CGRect(x: inset, y: inset, width: body, height: body)
    ctx.addPath(CGPath(roundedRect: rect, cornerWidth: radius, cornerHeight: radius, transform: nil))
    ctx.setFillColor(CGColor(red: 0.04, green: 0.04, blue: 0.045, alpha: 1))
    ctx.fillPath()

    let c = CGPoint(x: s / 2, y: s / 2)
    let R = body * 0.30            // ring
    let lw = body * 0.040          // stroke
    let r = R * 0.42               // inner hexagon
    ctx.setStrokeColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    ctx.setLineWidth(lw)
    ctx.setLineCap(.round)
    ctx.setLineJoin(.round)
    ctx.addEllipse(in: CGRect(x: c.x - R, y: c.y - R, width: 2 * R, height: 2 * R))
    ctx.strokePath()
    // six blades: each hexagon edge runs on until it meets the ring
    var v: [CGPoint] = []
    for k in 0..<6 {
        let a = (30.0 + 60.0 * Double(k)) * .pi / 180
        v.append(CGPoint(x: c.x + r * CGFloat(cos(a)), y: c.y + r * CGFloat(sin(a))))
    }
    for k in 0..<6 {
        let p = v[k], q = v[(k + 1) % 6]
        var d = CGPoint(x: q.x - p.x, y: q.y - p.y)
        let n = sqrt(d.x * d.x + d.y * d.y); d.x /= n; d.y /= n
        // solve |p + t d - c| = R for t > 0
        let fx = p.x - c.x, fy = p.y - c.y
        let b = fx * d.x + fy * d.y, cc = fx * fx + fy * fy - R * R
        let t = -b + sqrt(b * b - cc)
        ctx.move(to: p)
        ctx.addLine(to: CGPoint(x: p.x + t * d.x, y: p.y + t * d.y))
    }
    ctx.strokePath()
    return ctx.makeImage()!
}

func save(_ img: CGImage, _ path: String) {
    let rep = NSBitmapImageRep(cgImage: img)
    try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: path))
}

let set = CommandLine.arguments[1], preview = CommandLine.arguments[2]
try? FileManager.default.createDirectory(atPath: set, withIntermediateDirectories: true)
for (base, scales) in [(16, [1, 2]), (32, [1, 2]), (128, [1, 2]), (256, [1, 2]), (512, [1, 2])] {
    for sc in scales {
        let name = sc == 1 ? "icon_\(base)x\(base).png" : "icon_\(base)x\(base)@2x.png"
        save(render(base * sc), "\(set)/\(name)")
    }
}
save(render(1024), preview)
