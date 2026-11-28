import Foundation

/// Makes sure TMPDIR points at a folder that exists and can be written to.
///
/// macOS sometimes starts an app with a TMPDIR it inherited from somewhere else, for example a folder inside
/// an installer's sandbox (/private/tmp/PKInstallSandbox.*/tmp) that was deleted when the installer finished.
/// Every command the app runs (bootstrap.sh, the engine, pip, uv) would inherit it and fail to make temporary
/// files. This runs once at launch, before anything is started, and swaps a broken TMPDIR for this user's own
/// temporary folder.
enum TempDir {
    /// The broken TMPDIR found at launch, or nil when it was fine.
    nonisolated(unsafe) private(set) static var replaced: String?

    /// Fixes TMPDIR for this process (and so for everything it starts). Returns the folder now in use.
    @discardableResult
    static func repair() -> String {
        let env = ProcessInfo.processInfo.environment["TMPDIR"] ?? ""
        if !env.isEmpty, usable(env) { return env }
        let fixed = [userTemp()].compactMap { $0 }.first(where: usable)
            ?? ownTemp().flatMap { usable($0) ? $0 : nil } ?? "/tmp/"
        if !env.isEmpty { replaced = env }
        setenv("TMPDIR", fixed, 1)
        return fixed
    }

    static func usable(_ path: String) -> Bool {
        var isDir: ObjCBool = false
        return FileManager.default.fileExists(atPath: path, isDirectory: &isDir) && isDir.boolValue
            && FileManager.default.isWritableFile(atPath: path)
    }

    /// This user's own temporary folder from macOS (the /var/folders/…/T/ one).
    private static func userTemp() -> String? {
        let n = confstr(_CS_DARWIN_USER_TEMP_DIR, nil, 0)
        guard n > 0 else { return nil }
        var buf = [CChar](repeating: 0, count: n)
        return confstr(_CS_DARWIN_USER_TEMP_DIR, &buf, n) > 0 ? String(cString: buf) : nil
    }

    /// A tmp folder inside the Shuttrr home, made only when macOS gives no usable one.
    private static func ownTemp() -> String? {
        let home = ProcessInfo.processInfo.environment["SHUTTER_HOME"].flatMap { $0.isEmpty ? nil : $0 }
            ?? (NSHomeDirectory() as NSString).appendingPathComponent(".shutter")
        let own = ((home as NSString).expandingTildeInPath as NSString).appendingPathComponent("tmp")
        guard (try? FileManager.default.createDirectory(atPath: own, withIntermediateDirectories: true)) != nil
        else { return nil }
        return own + "/"
    }
}
