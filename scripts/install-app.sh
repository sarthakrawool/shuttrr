#!/usr/bin/env bash
# Install the Shuttrr Mac app from the latest GitHub release.
#
#   curl -fsSL https://raw.githubusercontent.com/sarthakrawool/shuttrr/main/scripts/install-app.sh | bash
#
# The app is ad-hoc signed (no paid Apple developer account), so macOS would block a copy
# downloaded in a browser. This script downloads the DMG with curl, copies Shuttrr.app to
# /Applications (or ~/Applications when /Applications is not writable), and removes the
# quarantine attribute so Gatekeeper lets it open. It never uses sudo.
#
# Options (environment variables):
#   SHUTTRR_DMG_URL   download from this URL instead of the latest release
#   SHUTTRR_DEST      install into this folder (default: /Applications, else ~/Applications)
#   SHUTTRR_NO_OPEN=1 do not open the app when done
set -euo pipefail

OWNER="sarthakrawool"   # GitHub user or organisation that publishes the releases
REPO="shuttrr"
DMG_URL="${SHUTTRR_DMG_URL:-https://github.com/${OWNER}/${REPO}/releases/latest/download/Shuttrr.dmg}"
APP_NAME="Shuttrr.app"

say() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Darwin" ]] || die "Shuttrr runs on macOS only."
[[ "$(uname -m)" == "arm64" ]] || die "Shuttrr needs an Apple Silicon Mac (M1 or newer)."
command -v hdiutil >/dev/null || die "hdiutil not found."

# Pick the destination folder: /Applications if we can write there, otherwise ~/Applications.
if [[ -n "${SHUTTRR_DEST:-}" ]]; then
    DEST="$SHUTTRR_DEST"
elif [[ -w /Applications ]]; then
    DEST="/Applications"
else
    DEST="$HOME/Applications"
fi
mkdir -p "$DEST" || die "cannot create $DEST"
[[ -w "$DEST" ]] || die "$DEST is not writable. Set SHUTTRR_DEST to a folder you own."

WORK="$(mktemp -d "${TMPDIR:-/tmp}/shuttrr-install.XXXXXX")"
MOUNT=""
cleanup() {
    if [[ -n "$MOUNT" ]]; then hdiutil detach "$MOUNT" -quiet >/dev/null 2>&1 || true; fi
    rm -rf "$WORK"
}
trap cleanup EXIT

say "Downloading $DMG_URL"
curl -fL --progress-bar -o "$WORK/Shuttrr.dmg" "$DMG_URL" || die "download failed."

say "Opening the disk image"
MOUNT="$WORK/mnt"
mkdir -p "$MOUNT"
hdiutil attach "$WORK/Shuttrr.dmg" -nobrowse -readonly -noautoopen -mountpoint "$MOUNT" -quiet \
    || die "could not open the disk image."
[[ -d "$MOUNT/$APP_NAME" ]] || die "$APP_NAME not found in the disk image."

# Quit a running copy so the files can be replaced. Quitting stops its engine.
if pgrep -f "$APP_NAME/Contents/MacOS/Shutter" >/dev/null 2>&1; then
    say "Quitting the running Shuttrr"
    osascript -e 'tell application id "com.shutter.app" to quit' >/dev/null 2>&1 || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        pgrep -f "$APP_NAME/Contents/MacOS/Shutter" >/dev/null 2>&1 || break
        sleep 1
    done
fi

say "Installing to $DEST/$APP_NAME"
rm -rf "$DEST/$APP_NAME"
ditto "$MOUNT/$APP_NAME" "$DEST/$APP_NAME"

# The app is ad-hoc signed, not notarised: drop the quarantine flag so Gatekeeper does not block it.
xattr -dr com.apple.quarantine "$DEST/$APP_NAME" 2>/dev/null || true

say "Done: $DEST/$APP_NAME"
if [[ "${SHUTTRR_NO_OPEN:-0}" != "1" ]]; then
    open "$DEST/$APP_NAME"
fi
