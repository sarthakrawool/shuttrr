#!/bin/bash
# Read-only probe: which chip, macOS and MLX this Mac has, and whether the GPU neural
# accelerators (M5 family, macOS 26.2+) are in play. Changes nothing. The matmul timing at
# the end needs the GPU for about 20 s, so run it under the GPU lock:
#   ~/longcat-mlx/speed/gpulock.sh shutter-probe -- bash scripts/chip_probe.sh [python]
PY="${1:-$HOME/longcat-mlx/code/.venv/bin/python}"
echo "== system"
sw_vers
sysctl -n machdep.cpu.brand_string hw.model hw.memsize
sysctl iogpu.wired_limit_mb iogpu.wired_lwm_mb 2>/dev/null
sysctl -n kern.memorystatus_level vm.swapusage
echo "== tools"
xcrun --find xctrace 2>/dev/null || echo "xctrace: not installed (no Metal System Trace)"
xcrun --sdk macosx --show-sdk-version 2>/dev/null
echo "== mlx"
"$PY" - <<'EOF'
import time, mlx.core as mx
print("mlx", mx.__version__)
try:
    print("device", mx.metal.device_info())
except Exception as e:
    print("device_info failed:", e)
N = 4096
for dt in (mx.float16, mx.bfloat16, mx.float32):
    a = mx.random.normal((N, N)).astype(dt); b = mx.random.normal((N, N)).astype(dt)
    mx.eval(a, b)
    mx.eval(a @ b)                      # warm up
    reps = 20
    t = time.perf_counter()
    for _ in range(reps):
        c = a @ b; mx.eval(c)
    s = (time.perf_counter() - t) / reps
    print(f"matmul {N}x{N} {dt}: {s*1e3:.1f} ms  {2*N**3/s/1e12:.1f} TFLOP/s")
EOF
