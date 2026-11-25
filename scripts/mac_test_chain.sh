#!/bin/bash
# Shutter test chain for the Mac Studio: every Fast / Low-memory measurement in order, resumable.
#
#   bash ~/shutter/scripts/mac_test_chain.sh                 # run everything still missing
#   bash ~/shutter/scripts/mac_test_chain.sh --dry-run       # print the plan, write + validate the config
#   bash ~/shutter/scripts/mac_test_chain.sh --only T1,T3    # just these (prefix match: T3 = all T3_*)
#   bash ~/shutter/scripts/mac_test_chain.sh --skip T5
#   bash ~/shutter/scripts/mac_test_chain.sh --model dummy --python python3   # orchestration test, no model
#
# Steps (each a Shutter job: 3.72 s clip, seed 42, ~/longcat-mlx/speed/inputs/{start_desk.png,vo.wav,prompt.txt}):
#   T1               480x832 fast                     latents + frames vs ~/longcat-mlx/speed/exact/p480 (must match)
#   T2               the same job again, warm         keep-warm + prompt/audio cache gain
#   T3_<rung>        480x832 at split, staged, stream_wide, stream_tight   time, per-stage peaks, vs T1
#   T4               704x1312 fast                    vs ~/longcat-mlx/speed/exact/lossless (must match)
#   T5_<rung>        704x1312 staged, stream_tight    vs T4
# Results: ~/shutter-test/results/<step>.json (+ clip, latents), summary.md, measured.json (also written to the
# test SHUTTER_HOME, the shape docs/MODES.md describes). A step whose <step>.json exists is skipped (idempotent) unless
# the job itself errored (status ERROR); delete that file (or pass --redo) to run it again.
# `touch ~/shutter-test/PAUSE` stops before the next step.
#
# Safety, as ~/longcat-mlx/speed/gpulock.sh: refuses to start a step when `sysctl -n kern.memorystatus_level`
# < 78 (a big model is loaded) or a render process runs; GPU locks ~/studio-gpu.lock.d and
# ~/ImaginationStudio/queue/.render.lock (taken by the daemon, or by the helper in worker mode); the job is
# killed if swap grows > 300 MB; swap is recorded before/after every step and the chain stops if it grew
# > 300 MB. Engine: the Rust daemon (~/shutter/daemon/target/release/shutterd) on port 8494 with its own
# SHUTTER_HOME; if it cannot run here, the chain drives `python -m shutter.worker` directly over its JSON-lines
# protocol (scripts/chain_job.py: no web packages needed). bash 3.2 compatible.

set -u
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

# ------------------------------------------------------------------------------------------------ defaults
MODEL=longcat
MODEL_ID=longcat-avatar-1.6-8bit
PY="$HOME/longcat-mlx/code/.venv/bin/python"
SHUTTER_DIR="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"   # the checkout this script is in
TEST_ROOT="$HOME/shutter-test"
CODE_DIR="$HOME/longcat-mlx/code-exact"
WEIGHTS="$HOME/longcat-mlx/weights/LongCat-Video-Avatar-1.6-q8"
INPUTS="$HOME/longcat-mlx/speed/inputs"
REF_ROOT="$HOME/longcat-mlx/speed/exact"
LOCK_DIRS="$HOME/studio-gpu.lock.d:$HOME/ImaginationStudio/queue/.render.lock"
BUSY_PATTERN="h3_generate.py|zig_fasth3|run_mps_avatar|sd-cli"
MIN_LEVEL_START=78
SWAP_MAX_MB=300
PORT=8494
VIA=auto
DRY=0
ONLY=""
SKIP=""
REDO=0
PY_SET=0

usage() { sed -n '2,30p' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --only) ONLY="$2"; shift ;;
    --skip) SKIP="$2"; shift ;;
    --redo) REDO=1 ;;
    --model) MODEL="$2"; shift ;;
    --python) PY="$2"; PY_SET=1; shift ;;
    --shutter-dir) SHUTTER_DIR="$2"; shift ;;
    --test-root) TEST_ROOT="$2"; shift ;;
    --via) VIA="$2"; shift ;;
    --port) PORT="$2"; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 2 ;;
  esac
  shift
done

case "$MODEL" in longcat|dummy) ;; *) echo "--model must be longcat or dummy" >&2; exit 2 ;; esac
case "$VIA" in auto|daemon|worker) ;; *) echo "--via must be auto, daemon or worker" >&2; exit 2 ;; esac

SD_ABS=$(cd "$SHUTTER_DIR" 2>/dev/null && pwd) || { echo "no Shutter checkout at $SHUTTER_DIR" >&2; exit 2; }
SHUTTER_DIR="$SD_ABS"
[ -f "$SHUTTER_DIR/shutter/worker.py" ] || { echo "$SHUTTER_DIR is not a Shutter checkout" >&2; exit 2; }
SCRIPTS="$SHUTTER_DIR/scripts"
HELPER="$SCRIPTS/chain_job.py"
DAEMON="$SHUTTER_DIR/daemon/target/release/shutterd"
HOME_T="$TEST_ROOT/home"
RESULTS="$TEST_ROOT/results"
if [ "$MODEL" = dummy ]; then
  LOCK_DIRS="$TEST_ROOT/locks/studio-gpu.lock.d:$TEST_ROOT/locks/render.lock"   # never touch the real locks
  [ $PY_SET = 1 ] || PY=python3
fi
IS_MAC=0; [ "$(uname -s)" = Darwin ] && IS_MAC=1

log() { echo "[chain $(date '+%H:%M:%S')] $*"; }
die() { echo "[chain] ERROR: $*" >&2; exit "${2:-1}"; }

# ------------------------------------------------------------------------------------------------ steps
# id|title|size|mode|rung|est_seconds|reference (p480|lossless|T1|T4|-)|must_match
STEPS="T1|480p fast|480x832|fast||480|p480|1
T2|480p fast, warm repeat|480x832|fast||430|T1|1
T3_split|480p low memory: split|480x832|low_memory|split|440|T1|1
T3_staged|480p low memory: staged|480x832|low_memory|staged|460|T1|1
T3_stream_wide|480p low memory: stream wide|480x832|low_memory|stream_wide|490|T1|1
T3_stream_tight|480p low memory: stream tight|480x832|low_memory|stream_tight|520|T1|1
T4|704x1312 fast|704x1312|fast||1600|lossless|1
T5_staged|704x1312 low memory: staged|704x1312|low_memory|staged|1650|T4|1
T5_stream_tight|704x1312 low memory: stream tight|704x1312|low_memory|stream_tight|1800|T4|1"
ORDER=$(echo "$STEPS" | cut -d'|' -f1 | tr '\n' ',' | sed 's/,$//')

selected() {   # $1 = step id
  local id="$1" x ok
  if [ -n "$ONLY" ]; then
    ok=0
    for x in $(echo "$ONLY" | tr ',' ' '); do case "$id" in "$x"|"$x"_*) ok=1 ;; esac; done
    [ $ok = 1 ] || return 1
  fi
  for x in $(echo "$SKIP" | tr ',' ' '); do case "$id" in "$x"|"$x"_*) return 1 ;; esac; done
  return 0
}

field() { echo "$1" | cut -d'|' -f"$2"; }

# a step is done when its <step>.json exists and the job finished (status ok or FAIL); ERROR runs again
step_done() { [ -f "$RESULTS/$1.json" ] && ! grep -q '"status": "ERROR"' "$RESULTS/$1.json"; }

# ------------------------------------------------------------------------------------------------ machine checks
swap_mb() {
  if [ $IS_MAC = 1 ]; then
    sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'
  else
    awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{printf "%.1f\n", (t-f)/1024}' /proc/meminfo
  fi
}

preflight() {   # refuses like gpulock.sh: 97 = big model loaded, 98 = a render is running
  if [ $IS_MAC = 1 ]; then
    local lvl; lvl=$(sysctl -n kern.memorystatus_level)
    if [ "$lvl" -lt "$MIN_LEVEL_START" ]; then
      echo "[chain] memory level $lvl < $MIN_LEVEL_START: something big is loaded (Flash Next in oMLX?). NOT running." >&2
      return 97
    fi
  elif [ "$MODEL" = longcat ]; then
    echo "[chain] not macOS: no kern.memorystatus_level check" >&2
  fi
  if pgrep -f "$BUSY_PATTERN" >/dev/null 2>&1; then
    echo "[chain] a render is running ($BUSY_PATTERN). NOT running." >&2
    return 98
  fi
  return 0
}

swap_growth_ok() {   # $1 before, $2 after (MB, decimals allowed)
  awk -v a="$1" -v b="$2" -v m="$SWAP_MAX_MB" 'BEGIN{exit !((b - a) <= m)}'
}

# ------------------------------------------------------------------------------------------------ references
REF_P480_LAT=""; REF_P480_MP4=""; REF_LL_LAT=""; REF_LL_MP4=""
find_refs() {
  if [ "$MODEL" = dummy ]; then
    REF_P480_LAT="$TEST_ROOT/fake-ref/p480/latents_denorm.npy"; REF_P480_MP4=""
    REF_LL_LAT="$TEST_ROOT/fake-ref/lossless/latents_denorm.npy"; REF_LL_MP4=""
    return
  fi
  local d
  d="$REF_ROOT/p480"
  if [ -d "$d" ]; then
    [ -f "$d/latents_denorm.npy" ] && REF_P480_LAT="$d/latents_denorm.npy"
    if [ -f "$d/p480.mp4" ]; then REF_P480_MP4="$d/p480.mp4"; else REF_P480_MP4=$(ls "$d"/*.mp4 2>/dev/null | grep -v -e side -e partial | head -1); fi
  fi
  d="$REF_ROOT/lossless"
  if [ -d "$d" ]; then
    [ -f "$d/latents_denorm.npy" ] && REF_LL_LAT="$d/latents_denorm.npy"
    if [ -f "$d/lossless.mp4" ]; then REF_LL_MP4="$d/lossless.mp4"; else REF_LL_MP4=$(ls "$d"/*.mp4 2>/dev/null | grep -v -e side -e partial | head -1); fi
  fi
}

# ------------------------------------------------------------------------------------------------ plan
mkdir -p "$TEST_ROOT" "$RESULTS" "$HOME_T"
find_refs
TOTAL=0; N=0; PLAN=""
while IFS= read -r line; do
  id=$(field "$line" 1)
  selected "$id" || continue
  est=$(field "$line" 6)
  [ "$MODEL" = dummy ] && est=2
  if step_done "$id" && [ $REDO = 0 ]; then
    PLAN="$PLAN
  $id  $(field "$line" 2)  -> done already (skip; delete $RESULTS/$id.json to redo)"
    continue
  fi
  if [ "$id" = T2 ] && { ! selected T1 || { step_done T1 && [ $REDO = 0 ]; }; }; then
    est=$((est * 2))   # T2 without T1 in the same session runs T1's job first as a warm-up
  fi
  TOTAL=$((TOTAL + est)); N=$((N + 1))
  PLAN="$PLAN
  $id  $(field "$line" 2)  rung=$(field "$line" 5)  ~$((est / 60)) min"
done <<EOF
$STEPS
EOF

echo "Shutter test chain ($MODEL) on $(hostname)"
echo "  python      $PY"
echo "  shutter     $SHUTTER_DIR"
echo "  test home   $HOME_T"
echo "  results     $RESULTS"
echo "  references  p480: ${REF_P480_LAT:-none} ${REF_P480_MP4:-}"
echo "              lossless: ${REF_LL_LAT:-none} ${REF_LL_MP4:-}"
echo "Plan:$PLAN"
echo "Estimated: $N step(s), ~$((TOTAL / 60)) min ($(awk -v t=$TOTAL 'BEGIN{printf "%.1f", t/3600}') h) of GPU time, plus model loads."

# ------------------------------------------------------------------------------------------------ interpreter + config
if ! "$PY" -c 'import sys' >/dev/null 2>&1; then
  [ $DRY = 1 ] && log "WARNING: python $PY not found (dry run continues with python3 for the config check)" && PY_CHECK=python3 || die "python $PY not found"
else
  PY_CHECK="$PY"
fi
NEED='import numpy'
[ "$MODEL" = longcat ] && NEED='import numpy, mlx.core'
if ! "$PY_CHECK" -c "$NEED" >/dev/null 2>&1; then
  if [ $DRY = 1 ]; then log "WARNING: '$NEED' fails in $PY_CHECK (the real run would stop here)"
  else die "'$NEED' fails in $PY: use the LongCat venv python (--python), the chain needs numpy and mlx"; fi
fi
export PYTHONPATH="$SHUTTER_DIR"

log "writing $HOME_T/config.json"
CFG_OUT=$("$PY_CHECK" "$HELPER" config --home "$HOME_T" --python "$PY" --shutter-dir "$SHUTTER_DIR" --model "$MODEL" \
  --model-id "$MODEL_ID" --code-dir "$CODE_DIR" --weights "$WEIGHTS" --lock-dirs "$LOCK_DIRS" --port "$PORT" \
  --swap-max-mb "$SWAP_MAX_MB") || die "config generation failed"
echo "  $CFG_OUT"

if [ "$MODEL" = longcat ]; then
  for f in "$INPUTS/start_desk.png" "$INPUTS/vo.wav" "$INPUTS/prompt.txt"; do
    [ -f "$f" ] || { [ $DRY = 1 ] && log "WARNING: missing input $f" || die "missing input $f"; }
  done
  [ -d "$CODE_DIR/longcat_video_avatar" ] || { [ $DRY = 1 ] && log "WARNING: no patched tree at $CODE_DIR" || die "no patched tree at $CODE_DIR"; }
  [ -d "$WEIGHTS/dit" ] || { [ $DRY = 1 ] && log "WARNING: no weights at $WEIGHTS" || die "no weights at $WEIGHTS"; }
  command -v ffmpeg >/dev/null 2>&1 || log "WARNING: ffmpeg not on PATH: the mp4 comparisons will error"
fi

# engine: daemon if it runs on this machine
if [ "$VIA" = auto ]; then
  if [ -x "$DAEMON" ] && "$DAEMON" --help >/dev/null 2>&1; then VIA=daemon; else VIA=worker; fi
fi
log "engine: $VIA$([ $VIA = daemon ] && echo " ($DAEMON, port $PORT)" || echo " (python -m shutter.worker over JSON lines)")"

if [ $DRY = 1 ]; then
  log "dry run: config written and validated, nothing started"
  exit 0
fi

# ------------------------------------------------------------------------------------------------ run
DAEMON_PID=""
cleanup() {
  if [ -n "$DAEMON_PID" ]; then
    kill "$DAEMON_PID" 2>/dev/null
    sleep 2
    kill -9 "$DAEMON_PID" 2>/dev/null
  fi
}
trap cleanup EXIT
trap 'echo "[chain] interrupted"; exit 130' INT TERM

start_daemon() {
  [ -n "$DAEMON_PID" ] && return 0
  if curl -sf "http://127.0.0.1:$PORT/v1/system" >/dev/null 2>&1; then
    die "something already answers on port $PORT: stop it or pass --port"
  fi
  log "starting the daemon on port $PORT"
  SHUTTER_HOME="$HOME_T" "$DAEMON" serve --port "$PORT" >>"$RESULTS/daemon.log" 2>&1 &
  DAEMON_PID=$!
  local i=0
  while [ $i -lt 60 ]; do
    curl -sf "http://127.0.0.1:$PORT/v1/system" >/dev/null 2>&1 && return 0
    kill -0 "$DAEMON_PID" 2>/dev/null || break
    sleep 1; i=$((i + 1))
  done
  log "the daemon did not come up (see $RESULTS/daemon.log): falling back to the worker"
  kill "$DAEMON_PID" 2>/dev/null; DAEMON_PID=""; VIA=worker
}

if [ "$MODEL" = dummy ]; then   # fake references with the dummy adapter's deterministic latents
  "$PY" - "$TEST_ROOT/fake-ref" <<'PYEOF'
import sys, pathlib, numpy as np
root = pathlib.Path(sys.argv[1])
for name in ("p480", "lossless"):
    (root / name).mkdir(parents=True, exist_ok=True)
    if (root / name / "latents_denorm.npy").exists():
        continue   # kept, so a test can perturb it and watch the comparison fail
    a = np.random.default_rng(42).standard_normal((1, 16, 3, 4, 4)).astype(np.float32) + len("test chain")
    np.save(root / name / "latents_denorm.npy", a)
PYEOF
fi

FAILS=0
run_group() {   # $1 = group name, then step ids; runs the jobs in ONE engine session
  local group="$1"; shift
  local jobs="$RESULTS/$group.jobs.json" id line req sep="" s0 s1 ref base rung title must FA
  [ -f "$TEST_ROOT/PAUSE" ] && { log "PAUSE file present: stopping before $group"; exit 0; }
  preflight || exit $?
  s0=$(swap_mb)
  echo "[" >"$jobs"
  for id in "$@"; do
    if [ "${id#warmup:}" != "$id" ]; then
      local real="${id#warmup:}"
      line=$(echo "$STEPS" | grep "^$real|")
      req="$RESULTS/$real.warmup.request.json"
      mkdir -p "$RESULTS/$real-warmup"
      "$PY" "$HELPER" request --step "$real-warmup" --model "$MODEL" --model-id "$MODEL_ID" --size "$(field "$line" 3)" \
        --mode "$(field "$line" 4)" --rung "$(field "$line" 5)" --out-dir "$RESULTS/$real-warmup" --out "$req" \
        --image "$INPUTS/start_desk.png" --audio "$INPUTS/vo.wav" --prompt-file "$INPUTS/prompt.txt" || return 1
      printf '%s{"name": "%s-warmup", "warmup": true, "request": %s}\n' "$sep" "$real" "$(cat "$req")" >>"$jobs"
    else
      line=$(echo "$STEPS" | grep "^$id|")
      req="$RESULTS/$id.request.json"
      mkdir -p "$RESULTS/$id"
      "$PY" "$HELPER" request --step "$id" --model "$MODEL" --model-id "$MODEL_ID" --size "$(field "$line" 3)" \
        --mode "$(field "$line" 4)" --rung "$(field "$line" 5)" --out-dir "$RESULTS/$id" --out "$req" \
        --image "$INPUTS/start_desk.png" --audio "$INPUTS/vo.wav" --prompt-file "$INPUTS/prompt.txt" || return 1
      printf '%s{"name": "%s", "request": %s}\n' "$sep" "$id" "$(cat "$req")" >>"$jobs"
    fi
    sep=","
  done
  echo "]" >>"$jobs"
  [ "$VIA" = daemon ] && start_daemon
  log "group $group: $* (via $VIA)"
  "$PY" "$HELPER" run --via "$VIA" --url "http://127.0.0.1:$PORT" --python "$PY" --jobs "$jobs" \
    --results "$RESULTS" --lock-dirs "$LOCK_DIRS" --swap-max-mb "$SWAP_MAX_MB" 2>&1 | tee -a "$RESULTS/chain.log"
  s1=$(swap_mb)
  for id in "$@"; do
    [ "${id#warmup:}" != "$id" ] && continue
    line=$(echo "$STEPS" | grep "^$id|")
    title=$(field "$line" 2); ref=$(field "$line" 7); must=$(field "$line" 8)
    FA=(--step "$id" --title "$title" --job "$RESULTS/$id.job.json" --out "$RESULTS/$id.json"
      --swap-before "$s0" --swap-after "$s1")
    case "$ref" in
      p480) [ -n "$REF_P480_LAT" ] && FA=("${FA[@]}" --ref-latents "$REF_P480_LAT")
            [ -n "$REF_P480_MP4" ] && FA=("${FA[@]}" --ref-mp4 "$REF_P480_MP4")
            [ -z "$REF_P480_LAT$REF_P480_MP4" ] && log "$id: no p480 reference under $REF_ROOT: comparison skipped" ;;
      lossless) [ -n "$REF_LL_LAT" ] && FA=("${FA[@]}" --ref-latents "$REF_LL_LAT")
            [ -n "$REF_LL_MP4" ] && FA=("${FA[@]}" --ref-mp4 "$REF_LL_MP4")
            [ -z "$REF_LL_LAT$REF_LL_MP4" ] && log "$id: no lossless reference under $REF_ROOT: comparison skipped" ;;
      T1|T4) base="$RESULTS/$ref"
            if [ -f "$base/latents/latents_denorm.npy" ]; then FA=("${FA[@]}" --base-latents "$base/latents/latents_denorm.npy")
            else log "$id: $ref has no latents yet: comparison with $ref skipped"; fi
            [ -f "$base/$ref.mp4" ] && FA=("${FA[@]}" --base-mp4 "$base/$ref.mp4") ;;
    esac
    [ "$must" = 1 ] && FA=("${FA[@]}" --must-match)
    if ! "$PY" "$HELPER" finish "${FA[@]}"; then FAILS=$((FAILS + 1)); log "$id: FAILED (see $RESULTS/$id.json)"; fi
  done
  log "swap ${s0} MB -> ${s1} MB"
  if ! swap_growth_ok "$s0" "$s1"; then
    log "swap grew more than $SWAP_MAX_MB MB during $group: stopping the chain"
    finish_up
    exit 3
  fi
  return 0
}

finish_up() {
  "$PY" "$HELPER" summary --results "$RESULTS" --out "$RESULTS/summary.md" --order "$ORDER" >/dev/null || true
  "$PY" "$HELPER" measured --results "$RESULTS" --out "$RESULTS/measured.json,$HOME_T/measured.json" \
    --model-id "$([ "$MODEL" = dummy ] && echo dummy || echo "$MODEL_ID")" --shutter-dir "$SHUTTER_DIR" || true
  log "summary: $RESULTS/summary.md"
}

pending() { selected "$1" && { [ $REDO = 1 ] || ! step_done "$1"; }; }

# T1 + T2 share one engine session (T2 shows the warm gain); T2 alone gets a warm-up run of T1's job first
if pending T1 && pending T2; then run_group T1T2 T1 T2
elif pending T1; then run_group T1 T1
elif pending T2; then run_group T2 warmup:T1 T2
fi
for id in T3_split T3_staged T3_stream_wide T3_stream_tight T4 T5_staged T5_stream_tight; do
  pending "$id" && run_group "$id" "$id"
done

finish_up
cat "$RESULTS/summary.md"
BAD=""
for id in $(echo "$ORDER" | tr ',' ' '); do   # this run's failures and earlier ones still on record
  selected "$id" && [ -f "$RESULTS/$id.json" ] && grep -q -e '"status": "FAIL"' -e '"status": "ERROR"' "$RESULTS/$id.json" && BAD="$BAD $id"
done
if [ $FAILS -gt 0 ] || [ -n "$BAD" ]; then log "FAILED:${BAD:- $FAILS step(s)}"; exit 1; fi
log "all steps ok"
exit 0
