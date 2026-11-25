#!/usr/bin/env bash
# Shuttrr installer for Apple silicon Macs.
#
#   scripts/install.sh                 set up the engine and build the app
#   scripts/install.sh --download      also download LongCat Video Avatar 1.6 now (38 GB)
#   scripts/install.sh --no-app        skip building the Mac app (use the dashboard in a browser)
#
# What it does, in order: checks this Mac, then runs scripts/bootstrap.sh (the same set-up the app's
# first-run window uses: private Python in ~/.shutter/venv, Shuttrr and what LongCat needs, config.json), builds Shuttrr.app into ~/Applications when Swift is installed, then runs `shutter doctor`.
# It is safe to run again: it only adds what is missing. Nothing outside ~/.shutter, ~/Applications
# and ~/.local/bin is touched, and it never needs sudo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOME_DIR="${SHUTTER_HOME:-$HOME/.shutter}"
VENV="$HOME_DIR/venv"
MODEL="longcat-avatar-1.6-8bit"
DOWNLOAD=0
BUILD_APP=1
for arg in "$@"; do
    case "$arg" in
        --download) DOWNLOAD=1 ;;
        --no-app) BUILD_APP=0 ;;
        -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

say() { printf '\n==> %s\n' "$*"; }
fail() { printf '\nShuttrr cannot be installed: %s\n' "$*" >&2; exit 1; }

say "Checking this Mac"
[[ "$(uname -s)" == "Darwin" ]] || fail "this is not a Mac."
[[ "$(uname -m)" == "arm64" ]] || fail "it needs an Apple silicon Mac (M1 or newer)."
MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
FREE_GB=$(df -g "$HOME" | awk 'NR==2 {print $4}')
echo "Memory: ${MEM_GB} GB. Free disk space: ${FREE_GB} GB."
if (( MEM_GB < 16 )); then
    echo "Note: LongCat needs 16 GB of memory. Shuttrr still installs, with only the built-in test model."
fi

say "Setting up the engine in $HOME_DIR (scripts/bootstrap.sh)"
SHUTTER_HOME="$HOME_DIR" bash "$ROOT/scripts/bootstrap.sh" --source "$ROOT" \
    || fail "the set-up stopped (the lines above say why)."

if (( BUILD_APP )); then
    if command -v swift >/dev/null 2>&1; then
        say "Building Shuttrr.app"
        if bash "$ROOT/app/scripts/build_app.sh" >"$HOME_DIR/app-build.log" 2>&1; then
            mkdir -p "$HOME/Applications"
            rm -rf "$HOME/Applications/Shuttrr.app"
            cp -R "$ROOT/app/build/Shuttrr.app" "$HOME/Applications/Shuttrr.app"
            echo "Installed ~/Applications/Shuttrr.app"
        else
            echo "The app did not build (details in $HOME_DIR/app-build.log). The engine still works:"
            echo "run 'shutter serve' and open http://127.0.0.1:8484"
        fi
    else
        echo "Swift is not installed, so the Mac app was skipped (install Xcode or its command line tools"
        echo "and run this again). Until then: run 'shutter serve' and open http://127.0.0.1:8484"
    fi
fi

if (( DOWNLOAD )); then
    say "Downloading LongCat Video Avatar 1.6 (38 GB, it can be stopped and resumed)"
    SHUTTER_HOME="$HOME_DIR" "$VENV/bin/shutter" download "$MODEL"
fi

say "Checking the result"
SHUTTER_HOME="$HOME_DIR" "$VENV/bin/shutter" doctor || true

cat <<EOF

Shuttrr is installed.
  Open the app:        open ~/Applications/Shuttrr.app      (or: shutter serve, then http://127.0.0.1:8484)
  Get the model:       press Download in the app's set-up card, or run: shutter download $MODEL
  Check this Mac:      shutter doctor
If 'shutter' is not found, add ~/.local/bin to your PATH or use $VENV/bin/shutter
EOF
