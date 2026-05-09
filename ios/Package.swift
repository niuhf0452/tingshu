// swift-tools-version: 5.9
// ZIPFoundation is needed to handle the `/api/books/{id}/download` zip the
// server returns (meta.json + chapter text files bundled in one archive).
import PackageDescription

let package = Package(
    name: "TingShu",
    platforms: [
        // iOS 17 / macOS 14 are SwiftData minimums — we use it for play
        // progress persistence (§3.4). Bumped from 16/13 when persistence
        // landed; users of an older OS can't ship the app anyway.
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(name: "TingShuCore", targets: ["TingShuCore"]),
    ],
    dependencies: [
        .package(url: "https://github.com/weichsel/ZIPFoundation.git", from: "0.9.0"),
    ],
    targets: [
        .target(
            name: "TingShuCore",
            dependencies: ["ZIPFoundation"],
            path: "TingShu/Sources",
            exclude: [
                "App/Info.plist",
                "App/TingShuApp.swift",  // @main entry, only for the app target
                "App/Assets.xcassets",
            ]
        ),
    ]
)
