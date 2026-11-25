#!/bin/bash
# Build metal4_bench with swiftc (no Xcode project) and, with --yes, run it under the GPU lock.
#
#   bash build_and_run.sh              # checks + build only, prints the run command
#   bash build_and_run.sh --yes [args] # checks + build + run (args go to metal4_bench, e.g. --quick)
#
# Changes no system setting. Writes only into this folder (binary, results/).
set -u
cd "$(dirname "$0")"
HERE="$(pwd)"

RUN=0
if [ "${1:-}" = "--yes" ]; then RUN=1; shift; fi

die() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }

[ "$(uname -s)" = "Darwin" ] || die "this needs macOS (Metal); uname says $(uname -s)"
[ "$(uname -m)" = "arm64" ] || warn "not arm64 ($(uname -m)); Apple-silicon GPU expected"
echo "== macOS $(sw_vers -productVersion) ($(sw_vers -buildVersion)), $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"

command -v xcrun >/dev/null || die "xcrun not found: install the Xcode command line tools (xcode-select --install)"
command -v swiftc >/dev/null || die "swiftc not found: install Xcode or the command line tools"
swiftc --version 2>&1 | head -2

SDK_VER="$(xcrun --sdk macosx --show-sdk-version 2>/dev/null || echo 0)"
SDK_PATH="$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || echo)"
echo "== macOS SDK $SDK_VER at $SDK_PATH"

# Metal 4 (MTLLanguageVersion.version4_0, MTLGPUFamily.metal4) arrived with the macOS 26 SDK; bfloat
# TensorOps need OS 26.1, and MLX gates its own TensorOps kernels on SDK and OS >= 26.2.
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }
if ! ver_ge "$SDK_VER" "26.0"; then
  die "SDK $SDK_VER has no Metal 4 API (need the macOS 26 SDK or newer, i.e. Xcode 26+ / matching CLT). Update Xcode; nothing else to do."
fi
ver_ge "$SDK_VER" "26.2" || warn "SDK $SDK_VER < 26.2: bfloat TensorOps / cooperative-tensor inputs may be missing; MLX requires 26.2 for its own TensorOps kernels"

# The MetalPerformancePrimitives shader header ships in the SDK as a framework (exact path unverified).
MPP_HDR="$(find "$SDK_PATH/System/Library/Frameworks" -maxdepth 4 -name 'MetalPerformancePrimitives.h' 2>/dev/null | head -1)"
if [ -n "$MPP_HDR" ]; then
  echo "== MPP header: $MPP_HDR"
else
  warn "MetalPerformancePrimitives.h not found under the SDK frameworks. The runtime shader compiler"
  warn "may still find it (the OS ships its own copy); if the tensor_ops library then fails to compile,"
  warn "the SDK/OS lacks the TensorOps API. The simdgroup baseline still runs."
fi

# Optional offline pre-check of the shaders. Since Xcode 26 the 'metal' command-line compiler can be a
# separately downloaded component; the benchmark itself compiles at runtime and does not need it.
TMPD="$(mktemp -d "${TMPDIR:-/tmp}/metal4_bench.XXXXXX")"
if xcrun -sdk macosx -f metal >/dev/null 2>&1; then
  for def in BENCH_SIMD BENCH_MPP BENCH_MPP_INT8; do
    std="-std=metal4.0"; [ "$def" = BENCH_SIMD ] && std="-std=metal3.1"
    if xcrun -sdk macosx metal $std -D$def=1 -c kernels.metal -o "$TMPD/$def.air" 2>"$TMPD/$def.log"; then
      echo "== offline compile $def ($std): ok"
    else
      warn "offline compile $def ($std) FAILED (first lines below; full log $TMPD/$def.log). The run reports the runtime compiler's view too."
      head -25 "$TMPD/$def.log" >&2
    fi
  done
else
  echo "== 'metal' CLI not installed (optional; 'xcodebuild -downloadComponent MetalToolchain' adds it). Skipping offline shader check."
fi

echo "== building"
TARGET_OPT=(-target arm64-apple-macos26.0)
if ! swiftc -O -swift-version 5 "${TARGET_OPT[@]}" main.swift -o metal4_bench -framework Metal -framework Foundation 2>"$TMPD/swiftc.log"; then
  cat "$TMPD/swiftc.log" >&2
  if grep -q "version4_0\|metal4\|apple10" "$TMPD/swiftc.log"; then
    die "the SDK's Metal Swift API lacks Metal 4 symbols (version4_0 / .metal4 / .apple10): Xcode/SDK too old"
  fi
  die "swiftc failed (see README.md 'Likely compile errors')"
fi
grep -i warning "$TMPD/swiftc.log" | head -5
echo "== built $HERE/metal4_bench"

mkdir -p results
OUT="$HERE/results/metal4_bench_$(date +%Y%m%d_%H%M%S).json"
LOCK="$HOME/longcat-mlx/speed/gpulock.sh"
if [ -x "$LOCK" ]; then
  CMD=("$LOCK" shutter-metal4 -- "$HERE/metal4_bench" --yes --out "$OUT" "$@")
else
  CMD=("$HERE/metal4_bench" --yes --out "$OUT" "$@")
fi

if [ "$RUN" != 1 ]; then
  echo "Not running (pass --yes to run). Command would be:"
  echo "  ${CMD[*]}"
  exit 0
fi
[ -x "$LOCK" ] || warn "GPU lock wrapper $LOCK not found: running without the lock; make sure nothing else uses the GPU"
echo "== running: ${CMD[*]}"
"${CMD[@]}" >"$OUT.stdout"
rc=$?
echo "== exit $rc; JSON: $OUT (stdout copy: $OUT.stdout)"
exit $rc
