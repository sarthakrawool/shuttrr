// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Shutter",
    platforms: [.macOS(.v14)],
    dependencies: [
        // Sparkle 2: in-app updates, checked with an EdDSA signature (see scripts/release.sh).
        .package(url: "https://github.com/sparkle-project/Sparkle", from: "2.6.0"),
    ],
    targets: [
        .executableTarget(
            name: "Shutter",
            dependencies: [.product(name: "Sparkle", package: "Sparkle")],
            path: "Sources/Shutter",
            linkerSettings: [.unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "@executable_path/../Frameworks"])]
        )
    ]
)
