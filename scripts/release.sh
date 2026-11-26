#!/usr/bin/env bash
# Build a Shuttrr release: Shuttrr.dmg (first install), Shuttrr.zip (in-app updates) and appcast.xml (the feed
# Sparkle reads from https://github.com/sarthakrawool/shuttrr/releases/latest/download/appcast.xml).
#
#   scripts/release.sh VERSION [--notes FILE] [--publish]
#
# VERSION must match shutter/__init__.py and pyproject.toml (for example 0.2.1). The zip is signed with the EdDSA
# key in this Mac's login Keychain (Keychain account "shuttrr", made once with Sparkle's generate_keys); the private
# key never leaves the Keychain. Without --publish the script prints the `gh release create` command; with
# --publish it runs it (tag vVERSION on the current commit, marked Latest).
#
# Output: app/build/release/VERSION/{Shuttrr.dmg,Shuttrr.zip,appcast.xml}
# Test-only environment overrides: RELEASE_DOWNLOAD_BASE (where the appcast says the zip lives),
# SHUTTER_BUNDLE_ID (passed to build_app.sh), RELEASE_OUT (output folder).
set -euo pipefail

OWNER="sarthakrawool"
REPO="shuttrr"
KEY_ACCOUNT="shuttrr"
MIN_MACOS="14.0"

VERSION=""
NOTES=""
PUBLISH=0
while (( $# )); do
    case "$1" in
        --notes) NOTES="${2:-}"; shift 2 ;;
        --publish) PUBLISH=1; shift ;;
        -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
        *) VERSION="$1"; shift ;;
    esac
done
[[ -n "$VERSION" ]] || { echo "usage: scripts/release.sh VERSION [--notes FILE] [--publish]" >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TAG="v$VERSION"
OUT="${RELEASE_OUT:-$ROOT/app/build/release/$VERSION}"
BASE="${RELEASE_DOWNLOAD_BASE:-https://github.com/$OWNER/$REPO/releases/download/$TAG}"
SPARKLE_BIN="$ROOT/app/Shutter/.build/artifacts/sparkle/Sparkle/bin"

die() { echo "error: $*" >&2; exit 1; }

# The version is written in three places; they must agree.
PYV="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' shutter/__init__.py)"
TOMLV="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)"
[[ "$PYV" == "$VERSION" ]] || die "shutter/__init__.py says $PYV, not $VERSION"
[[ "$TOMLV" == "$VERSION" ]] || die "pyproject.toml says $TOMLV, not $VERSION"

# The app's engine upgrade reinstalls from the bundled pyproject.toml: it must parse.
python3 -c 'import sys, tomllib; tomllib.load(open(sys.argv[1], "rb"))' pyproject.toml \
    || die "pyproject.toml does not parse"

echo "==> building Shuttrr $VERSION"
SHUTTER_VERSION="$VERSION" app/scripts/build_app.sh --dmg
APP="$ROOT/app/build/Shuttrr.app"
[[ -x "$SPARKLE_BIN/sign_update" ]] || die "sign_update not found in $SPARKLE_BIN (run swift package resolve in app/Shutter)"
python3 -c 'import sys, tomllib; tomllib.load(open(sys.argv[1], "rb"))' "$APP/Contents/Resources/engine/pyproject.toml" \
    || die "the bundled pyproject.toml does not parse"
BUILT="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$APP/Contents/Info.plist")"
[[ "$BUILT" == "$VERSION" ]] || die "built app says $BUILT"

rm -rf "$OUT"
mkdir -p "$OUT"
cp "$ROOT/app/build/Shuttrr.dmg" "$OUT/Shuttrr.dmg"
echo "==> zipping"
ditto -c -k --sequesterRsrc --keepParent "$APP" "$OUT/Shuttrr.zip"

echo "==> signing the zip (EdDSA key from the Keychain)"
SIG_ATTRS="$("$SPARKLE_BIN/sign_update" --account "$KEY_ACCOUNT" "$OUT/Shuttrr.zip")"
SIG="$(printf '%s' "$SIG_ATTRS" | sed -n 's/.*sparkle:edSignature="\([^"]*\)".*/\1/p')"
LEN="$(printf '%s' "$SIG_ATTRS" | sed -n 's/.*length="\([0-9]*\)".*/\1/p')"
[[ -n "$SIG" && -n "$LEN" ]] || die "could not read the signature from sign_update"
"$SPARKLE_BIN/sign_update" --account "$KEY_ACCOUNT" --verify "$OUT/Shuttrr.zip" "$SIG" >/dev/null \
    || die "the signature does not verify"

PUBDATE="$(LC_ALL=C date -u '+%a, %d %b %Y %H:%M:%S +0000')"
cat > "$OUT/appcast.xml" <<XML
<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>Shuttrr</title>
    <link>https://github.com/$OWNER/$REPO</link>
    <description>Shuttrr updates</description>
    <language>en</language>
    <item>
      <title>Shuttrr $VERSION</title>
      <pubDate>$PUBDATE</pubDate>
      <sparkle:version>$VERSION</sparkle:version>
      <sparkle:shortVersionString>$VERSION</sparkle:shortVersionString>
      <sparkle:minimumSystemVersion>$MIN_MACOS</sparkle:minimumSystemVersion>
      <sparkle:releaseNotesLink>https://github.com/$OWNER/$REPO/releases/tag/$TAG</sparkle:releaseNotesLink>
      <enclosure url="$BASE/Shuttrr.zip" length="$LEN" type="application/octet-stream" sparkle:edSignature="$SIG"/>
    </item>
  </channel>
</rss>
XML
xmllint --noout "$OUT/appcast.xml" 2>/dev/null || plutil -lint "$OUT/appcast.xml" >/dev/null 2>&1 || true

echo "==> SHA-256"
(cd "$OUT" && shasum -a 256 Shuttrr.dmg Shuttrr.zip appcast.xml)

COMMIT="$(git rev-parse HEAD)"
CMD=(gh release create "$TAG" "$OUT/Shuttrr.dmg" "$OUT/Shuttrr.zip" "$OUT/appcast.xml"
     --repo "$OWNER/$REPO" --target "$COMMIT" --title "Shuttrr $VERSION" --latest)
if [[ -n "$NOTES" ]]; then CMD+=(--notes-file "$NOTES"); else CMD+=(--generate-notes); fi

if (( PUBLISH )); then
    [[ -z "$(git status --porcelain --untracked-files=no)" ]] || die "uncommitted changes: commit and push first"
    git fetch -q origin main
    [[ "$(git rev-parse origin/main)" == "$COMMIT" ]] || die "HEAD is not origin/main: push first"
    echo "==> publishing $TAG"
    "${CMD[@]}"
else
    echo
    echo "Release files are in $OUT. To publish:"
    printf ' %q' "${CMD[@]}"; echo
fi
