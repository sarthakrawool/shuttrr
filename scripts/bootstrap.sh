#!/usr/bin/env bash
# Shuttrr bootstrap: everything the engine needs, with no Terminal knowledge and no password.
#
#   bootstrap.sh --source DIR [--copy]
#
# DIR holds pyproject.toml and the shutter package (the repo root, or the copy inside Shuttrr.app).
# --copy first copies DIR into ~/.shutter/engine and installs from there (the app uses this, so the engine
# keeps working after the app is moved or updated).
#
# Safe to run again. Touches only ~/.shutter (or $SHUTTER_HOME) and ~/.local/bin, never needs sudo.
# Progress lines for the app look like:   @@ <step> start|done|fail <plain sentence>
# Steps, in order: check, python, engine, packages, finish. Every other line is log detail.
# For testing: SHUTTER_BOOTSTRAP_BARE=1 ignores this Mac's Python and ffmpeg, as on a fresh Mac.
set -Eeuo pipefail

SOURCE=""
COPY=0
while (( $# )); do
    case "$1" in
        --source) SOURCE="${2:-}"; shift 2 || { echo "--source needs a folder" >&2; exit 2; } ;;
        --source=*) SOURCE="${1#--source=}"; shift ;;
        --copy) COPY=1; shift ;;
        -h|--help) sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
done

DEFAULT_HOME="$HOME/.shutter"
HOME_DIR="${SHUTTER_HOME:-$DEFAULT_HOME}"
HOME_DIR="${HOME_DIR/#\~/$HOME}"
VENV="$HOME_DIR/venv"
BIN="$HOME_DIR/bin"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH:$BIN"
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_PROGRESS_BAR=off PYTHONDONTWRITEBYTECODE=1
# uv (only used on a Mac with no suitable Python) keeps its Pythons and cache inside our folder too
export UV_PYTHON_INSTALL_DIR="$HOME_DIR/python" UV_CACHE_DIR="$HOME_DIR/cache/uv"

# A TMPDIR that no longer exists (macOS can hand an app the temporary folder of an installer's sandbox,
# /private/tmp/PKInstallSandbox.*/tmp, which is deleted when the installer ends) breaks mktemp, uv and pip.
# Use this user's own temporary folder instead, then one inside our folder, then /tmp.
tmp_ok() { [[ -n "$1" && -d "$1" && -w "$1" ]]; }
if ! tmp_ok "${TMPDIR:-}"; then
    BAD_TMP="${TMPDIR:-}"
    for cand in "$(getconf DARWIN_USER_TEMP_DIR 2>/dev/null || true)" "$HOME_DIR/tmp" /tmp; do
        [[ "$cand" == "$HOME_DIR/tmp" ]] && mkdir -p "$cand" 2>/dev/null || true
        if tmp_ok "$cand"; then export TMPDIR="$cand"; break; fi
    done
    [[ -n "$BAD_TMP" ]] && echo "TMPDIR $BAD_TMP is missing or not writable; using $TMPDIR instead" || true
fi

STEP=""
ERR_SENTENCE=""
progress() { printf '@@ %s %s %s\n' "$1" "$2" "$3"; }
begin() { STEP="$1"; ERR_SENTENCE="$3"; progress "$1" start "$2"; }
done_() { progress "$STEP" done "$1"; STEP=""; }
fail() { progress "${STEP:-check}" fail "$1"; exit 1; }
on_err() {
    local code=$?
    trap - ERR
    echo "command failed (exit $code) at line $1" >&2
    progress "${STEP:-check}" fail "${ERR_SENTENCE:-Something went wrong. Show the details, then try again.}"
    exit 1
}
trap 'on_err $LINENO' ERR

py_ok() {   # $1 = a python; true when it is 3.11 to 3.13
    "$1" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)' >/dev/null 2>&1
}

BARE="${SHUTTER_BOOTSTRAP_BARE:-0}"
T0=$SECONDS

# ------------------------------------------------------------------ check
begin check "Checking this Mac." "Could not check this Mac. Show the details, then try again."
[[ "$(uname -s)" == "Darwin" ]] || fail "This is not a Mac. Shuttrr needs macOS."
[[ "$(uname -m)" == "arm64" ]] || fail "This Mac has an Intel chip. Shuttrr needs Apple silicon (M1 or newer)."
[[ -n "$SOURCE" ]] || fail "The installer was started without its files. Reinstall Shuttrr and try again."
SOURCE="$(cd "$SOURCE" 2>/dev/null && pwd)" || fail "The Shuttrr files are missing. Reinstall Shuttrr and try again."
[[ -f "$SOURCE/pyproject.toml" && -f "$SOURCE/shutter/__init__.py" ]] \
    || fail "The Shuttrr files are incomplete. Reinstall Shuttrr and try again."
mkdir -p "$HOME_DIR" "$BIN"
MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
FREE_GB=$(df -g "$HOME_DIR" | awk 'NR==2 {print $4}')
CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple silicon')"
echo "Chip: $CHIP. Memory: ${MEM_GB} GB. Free disk: ${FREE_GB} GB. Home: $HOME_DIR. Source: $SOURCE"
(( FREE_GB >= 4 )) || fail "Only ${FREE_GB} GB of disk space is free. Shuttrr needs at least 4 GB. Free some space and try again."
MEM_NOTE=""
(( MEM_GB >= 16 )) || MEM_NOTE=" Models need 16 GB, so only the built-in test model will run."
done_ "$CHIP with ${MEM_GB} GB of memory and ${FREE_GB} GB free.${MEM_NOTE}"

# ------------------------------------------------------------------ python
begin python "Getting Python ready." "Could not set up Python. Show the details, then try again."
if [[ -x "$VENV/bin/python" ]] && py_ok "$VENV/bin/python"; then
    PYV="$("$VENV/bin/python" -c 'import platform; print(platform.python_version())')"
    echo "Reusing $VENV (Python $PYV)"
    done_ "Using the Python already set up (version $PYV)."
else
    if [[ -e "$VENV" ]]; then
        OLD="$VENV.old-$(date +%Y%m%d-%H%M%S)"
        echo "Existing $VENV has no usable Python; moving it to $OLD"
        mv "$VENV" "$OLD"
    fi
    PY=""
    [[ "$BARE" == 1 ]] && echo "SHUTTER_BOOTSTRAP_BARE=1: ignoring this Mac's Python" || true
    for cand in python3.12 python3.13 python3.11 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.13 \
                /opt/homebrew/bin/python3.11; do
        if [[ "$BARE" != 1 ]] && command -v "$cand" >/dev/null 2>&1 && py_ok "$(command -v "$cand")"; then
            PY="$(command -v "$cand")"; break
        fi
    done
    if [[ -n "$PY" ]]; then
        echo "Making $VENV with $PY"
        "$PY" -m venv "$VENV" || fail "Could not make the Python environment. Show the details, then try again."
        HOW="this Mac's Python"
    else
        UV="$BIN/uv"
        if [[ ! -x "$UV" ]]; then
            echo "No Python 3.11 to 3.13 found; downloading uv"
            TMP="$(mktemp -d "${TMPDIR%/}/shuttrr-uv.XXXXXX")" \
                || fail "Could not make a temporary folder in ${TMPDIR}. Show the details, then try again."
            curl -fsSL --retry 3 -o "$TMP/uv.tar.gz" \
                https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz \
                || fail "Could not download Python. Check the internet connection and try again."
            tar -xzf "$TMP/uv.tar.gz" -C "$TMP" || fail "Could not unpack Python. Try again."
            UVBIN="$(find "$TMP" -type f -name uv -perm -u+x | head -1)"
            [[ -n "$UVBIN" ]] || fail "Could not unpack Python. Try again."
            cp "$UVBIN" "$UV"; chmod +x "$UV"; rm -rf "$TMP"
        fi
        echo "Making $VENV with uv (Python 3.12)"
        "$UV" venv --seed --python 3.12 "$VENV" \
            || fail "Could not download Python. Check the internet connection and try again."
        HOW="a private copy of Python"
    fi
    py_ok "$VENV/bin/python" || fail "The new Python environment does not work. Show the details, then try again."
    PYV="$("$VENV/bin/python" -c 'import platform; print(platform.python_version())')"
    done_ "Set up Python $PYV from $HOW."
fi
PYBIN="$VENV/bin/python"
"$PYBIN" -m pip --version >/dev/null 2>&1 || "$PYBIN" -m ensurepip --upgrade

# ------------------------------------------------------------------ engine
begin engine "Installing the Shuttrr engine." "Could not install the Shuttrr engine. Check the internet connection and try again."
ENGINE="$SOURCE"
if (( COPY )); then
    ENGINE="$HOME_DIR/engine"
    if [[ "$SOURCE" != "$ENGINE" ]]; then
        echo "Copying $SOURCE to $ENGINE"
        rm -rf "$ENGINE.new"
        mkdir -p "$ENGINE.new"
        (cd "$SOURCE" && tar --exclude '__pycache__' --exclude '*.pyc' --exclude '*.egg-info' -cf - .) \
            | (cd "$ENGINE.new" && tar -xf -) || fail "Could not copy the engine files. Free some disk space and try again."
        rm -rf "$ENGINE"
        mv "$ENGINE.new" "$ENGINE"
    fi
fi
"$PYBIN" -m pip install -e "$ENGINE[mlx]" \
    || fail "Could not install the Shuttrr engine. Check the internet connection and try again."
"$PYBIN" -c 'import shutter, mlx.core' \
    || fail "The engine installed but does not start. Show the details, then try again."
VER="$("$PYBIN" -c 'from importlib.metadata import version; print(version("shutter-engine"))' 2>/dev/null || echo "")"
done_ "Installed the Shuttrr engine${VER:+ $VER}."

# ------------------------------------------------------------------ packages
begin packages "Installing what the models need." "Could not install what the models need. Check the internet connection and try again."
NEEDS=()
while IFS= read -r line; do if [[ -n "$line" ]]; then NEEDS+=("$line"); fi; done < <("$PYBIN" -c '
from shutter.core.catalog import CATALOG
print("\n".join(sorted({n for e in CATALOG.values() for n in e["needs"]})))')
echo "Packages: ${NEEDS[*]}"
if (( ${#NEEDS[@]} )); then
    "$PYBIN" -m pip install "${NEEDS[@]}" \
        || fail "Could not install what the models need. Check the internet connection and try again."
fi
"$PYBIN" - "${NEEDS[@]}" <<'PY' || fail "Some model packages installed but do not load. Show the details, then try again."
import importlib, re, sys
special = {"pillow": "PIL", "imageio-ffmpeg": "imageio_ffmpeg", "huggingface-hub": "huggingface_hub"}
bad = []
for spec in sys.argv[1:]:
    name = re.split(r"[<>=!~\[ ;]", spec, maxsplit=1)[0].strip().lower()
    mod = special.get(name, name.replace("-", "_"))
    try:
        importlib.import_module(mod)
    except Exception as e:
        bad.append(f"{mod}: {e}")
if bad:
    print("\n".join(bad))
    raise SystemExit(1)
PY
FF_NOTE=""
if [[ "$BARE" != 1 ]] && command -v ffmpeg >/dev/null 2>&1 && [[ "$(command -v ffmpeg)" != "$BIN/ffmpeg" ]]; then
    echo "ffmpeg: $(command -v ffmpeg)"
else
    FFEXE="$("$PYBIN" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" \
        || fail "Could not set up ffmpeg, which writes the videos. Try again."
    ln -sf "$FFEXE" "$BIN/ffmpeg"
    echo "ffmpeg: linked $BIN/ffmpeg -> $FFEXE"
    FF_NOTE=" Added a private copy of ffmpeg."
fi
done_ "Installed ${#NEEDS[@]} packages the models use.${FF_NOTE}"

# ------------------------------------------------------------------ finish
begin finish "Finishing up." "Could not finish the set-up. Show the details, then try again."
if [[ ! -f "$HOME_DIR/config.json" ]]; then
    SHUTTER_HOME="$HOME_DIR" "$VENV/bin/shutter" init >/dev/null \
        || fail "Could not write the settings file. Show the details, then try again."
    echo "Wrote $HOME_DIR/config.json"
fi
LINK_NOTE=""
if [[ "$HOME_DIR" == "$DEFAULT_HOME" ]]; then
    mkdir -p "$HOME/.local/bin"
    ln -sf "$VENV/bin/shutter" "$HOME/.local/bin/shutter"
    echo "Linked ~/.local/bin/shutter"
    LINK_NOTE=" The shutter command is in ~/.local/bin."
else
    echo "SHUTTER_HOME is not the default, so ~/.local/bin/shutter was left alone"
fi
echo "---- shutter doctor"
SHUTTER_HOME="$HOME_DIR" "$VENV/bin/shutter" doctor || true
echo "Finished in $(( SECONDS - T0 )) s"
done_ "Shuttrr is set up.${LINK_NOTE}"
exit 0
