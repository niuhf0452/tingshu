// Cross-platform shims so `swift build` can compile these Views on macOS
// (for dev-time type checking) even though the production target is iOS.
import SwiftUI

extension ToolbarItemPlacement {
    static var topLeading: ToolbarItemPlacement {
        #if os(iOS)
        return .navigationBarLeading
        #else
        return .cancellationAction
        #endif
    }
    static var topTrailing: ToolbarItemPlacement {
        #if os(iOS)
        return .navigationBarTrailing
        #else
        return .primaryAction
        #endif
    }
}

enum Platform {
    /// Width of the narrower screen edge, used for gesture dead-zones. On
    /// macOS we use the window size instead — the exact number doesn't
    /// matter for the dev-time build.
    static var screenWidth: CGFloat {
        #if os(iOS)
        return UIScreen.main.bounds.width
        #else
        return 390
        #endif
    }
}
