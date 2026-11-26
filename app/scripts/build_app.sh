#!/usr/bin/env bash
# Build Shuttrr.app from the Swift package and ad-hoc sign it. The product is called Shuttrr; the executable,
# bundle id, Swift package and folders keep the old "shutter" spelling so settings and paths keep working.
# Usage: app/scripts/build_app.sh [--dmg]   (output: app/build/Shuttrr.app, with --dmg also app/build/Shuttrr.dmg)
# The engine sources (pyproject.toml, README.md, scripts/bootstrap.sh, the shutter package) are copied into
# Contents/Resources/engine, so the first-run window can install the engine with no repo checkout.
set -euo pipefail

MAKE_DMG=0
for arg in "$@"; do
    case "$arg" in
        --dmg) MAKE_DMG=1 ;;
        -h|--help) sed -n '2,6p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG_DIR="$APP_ROOT/Shutter"
REPO_ROOT="$(cd "$APP_ROOT/.." && pwd)"
BUILD_DIR="$APP_ROOT/build"
APP="$BUILD_DIR/Shuttrr.app"
VERSION="${SHUTTER_VERSION:-0.2.1}"

echo "==> swift build -c release"
cd "$PKG_DIR"
swift build -c release
BIN_DIR="$(swift build -c release --show-bin-path)"
BIN="$BIN_DIR/Shutter"
if [[ ! -x "$BIN" ]]; then
    echo "error: built binary not found at $BIN" >&2
    exit 1
fi

echo "==> assembling $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/Shutter"
# Drop debug symbols: they hold absolute build paths (the build machine's user name).
strip -S -x "$APP/Contents/MacOS/Shutter"
# Brand fonts (Fraunces, DM Sans, JetBrains Mono; all OFL) and the paper texture.
cp -R "$PKG_DIR/Resources/Brand" "$APP/Contents/Resources/Brand"
# App icon (white aperture mark on black; drawn by scripts/make_icon.swift)
cp "$PKG_DIR/Resources/Icon/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>Shutter</string>
    <key>CFBundleIdentifier</key>
    <string>com.shutter.app</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>Shuttrr</string>
    <key>CFBundleDisplayName</key>
    <string>Shuttrr</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSAppTransportSecurity</key>
    <dict>
        <key>NSAllowsLocalNetworking</key>
        <true/>
    </dict>
</dict>
</plist>
PLIST

printf 'APPL????' > "$APP/Contents/PkgInfo"
cp "$REPO_ROOT/LICENSE" "$REPO_ROOT/THIRD_PARTY_NOTICES.md" "$APP/Contents/Resources/"

echo "==> bundling the engine sources"
ENGINE_DIR="$APP/Contents/Resources/engine"
mkdir -p "$ENGINE_DIR"
cp "$REPO_ROOT/pyproject.toml" "$REPO_ROOT/README.md" "$REPO_ROOT/scripts/bootstrap.sh" "$ENGINE_DIR/"
(cd "$REPO_ROOT" && tar --exclude '__pycache__' --exclude '*.pyc' -cf - shutter) | (cd "$ENGINE_DIR" && tar -xf -)
chmod +x "$ENGINE_DIR/bootstrap.sh"

echo "==> ad-hoc codesign"
codesign --force --deep -s - "$APP"
codesign --verify --verbose=1 "$APP" || true

if (( MAKE_DMG )); then
    DMG="$BUILD_DIR/Shuttrr.dmg"
    echo "==> making $DMG"
    STAGE="$(mktemp -d "${TMPDIR:-/tmp}/shuttrr-dmg.XXXXXX")"
    cp -R "$APP" "$STAGE/Shuttrr.app"
    ln -s /Applications "$STAGE/Applications"
    rm -f "$DMG"
    hdiutil create -volname Shuttrr -srcfolder "$STAGE" -ov -format UDZO "$DMG"
    rm -rf "$STAGE"
fi

echo "==> done: $APP"
