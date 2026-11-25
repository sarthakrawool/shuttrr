#!/usr/bin/env python3
"""A fake Shutter worker for the daemon's integration tests. Speaks protocol 1
(docs/WORKER_PROTOCOL.md) on stdin/stdout; stdlib only.

Behaviour per job, from request.params:
  peak_gb (1.0)        memory estimate
  estimate_error (str) answer the estimate with an error
  estimate_hang        never answer the estimate
  steps (5), step_s (0.05)
  fail (str), fail_at  report `failed` with that error at step fail_at
  memory_refused (str) report `failed` with memory_refused: true
  crash_at (int)       print a traceback to stderr and exit(3) at that step
  hang                 after the first step, block forever and ignore cancel
  garbage              print junk lines on stdout while running
  slow_cancel_s        wait this long before honouring a cancel
  component_gb (0.5)   size of the warm "<model>/dit" pool entry kept after a run
  stderr_lines (int)   print that many numbered lines to stderr

Worker-wide, from the environment:
  FAKE_WORKER_OPS_LOG   append every received op name to this file
  FAKE_WORKER_START     "fail": log an error and exit 1 without ready
  FAKE_IGNORE_SHUTDOWN  "1": ignore `shutdown`
"""

import json
import os
import queue
import sys
import threading
import time

GB = 1024 ** 3
OUT = sys.stdout
LOCK = threading.Lock()
cancel = threading.Event()
runs: "queue.Queue[dict]" = queue.Queue()
pool = {}  # key -> (size, last_used, residency)
running_id = [None]


def send(obj):
    with LOCK:
        OUT.write(json.dumps(obj) + "\n")
        OUT.flush()


def junk(text):
    with LOCK:
        OUT.write(text + "\n")
        OUT.flush()


def log_op(op):
    p = os.environ.get("FAKE_WORKER_OPS_LOG")
    if p:
        with open(p, "a") as f:
            f.write(f"{op}\n")


def pool_event():
    now = time.time()
    send({"ev": "pool", "total_bytes": sum(v[0] for v in pool.values()),
          "entries": [{"key": k, "size_bytes": v[0], "idle_s": now - v[1], "pinned": False, "in_use": False,
                       "residency": v[2]} for k, v in sorted(pool.items())]})


def params(req):
    return (req or {}).get("params") or {}


def reader():
    try:
        read_ops()
    finally:
        os._exit(0)  # stdin EOF, or a reply to a daemon that is already gone (EPIPE): never linger


def read_ops():
    for raw in sys.stdin.buffer:
        try:
            msg = json.loads(raw)
        except ValueError:
            continue
        op = msg.get("op")
        log_op(op)
        if op == "hello":
            send({"ev": "ready", "version": "fake-1", "pid": os.getpid(), "protocol": 1})
        elif op == "estimate":
            p = params(msg.get("request"))
            if p.get("estimate_hang"):
                continue
            if p.get("estimate_error"):
                send({"ev": "estimate", "id": msg["id"], "error": p["estimate_error"]})
            else:
                send({"ev": "estimate", "id": msg["id"], "peak": int(float(p.get("peak_gb", 1.0)) * GB),
                      "steady": 0, "disk_scratch": 0, "note": "fake"})
        elif op == "run":
            runs.put(msg)
        elif op == "cancel":
            if msg.get("id") == running_id[0]:
                cancel.set()
        elif op == "stats":
            pool_event()
        elif op == "release":
            key = msg.get("key")
            keys = sorted(pool) if key is None else ([key] if key in pool else [])
            for k in keys:
                pool.pop(k, None)
            send({"ev": "released", "keys": keys})
            pool_event()
        elif op == "models":
            send({"ev": "models", "models": [{"id": "dummy", "family": "dummy", "kind": "video",
                                              "dir": "", "installed": True}]})
        elif op == "shutdown":
            if os.environ.get("FAKE_IGNORE_SHUTDOWN") != "1":
                os._exit(0)


def run(msg):
    jid = msg["id"]
    job = msg["job"]
    req = job["request"]
    p = params(req)
    steps = int(p.get("steps", 5))
    step_s = float(p.get("step_s", 0.05))
    running_id[0] = jid
    cancel.clear()
    for i in range(int(p.get("stderr_lines", 0))):
        print(f"stderr line {i}", file=sys.stderr, flush=True)
    t0 = time.time()
    for i in range(steps):
        if cancel.is_set():
            time.sleep(float(p.get("slow_cancel_s", 0)))
            send({"ev": "failed", "id": jid, "error": "cancelled", "cancelled": True})
            return
        if p.get("crash_at") is not None and i >= int(p["crash_at"]):
            print("Traceback (most recent call last):", file=sys.stderr)
            print("RuntimeError: fake crash", file=sys.stderr, flush=True)
            os._exit(3)
        if p.get("fail") and i >= int(p.get("fail_at", 0)):
            send({"ev": "failed", "id": jid, "error": p["fail"], "cancelled": False})
            return
        if p.get("memory_refused"):
            send({"ev": "failed", "id": jid, "error": p["memory_refused"], "cancelled": False,
                  "memory_refused": True})
            return
        if p.get("garbage"):
            junk("this is not json {")
            junk('{"no_ev": true}')
            junk("[1, 2, 3]")
        time.sleep(step_s)
        send({"ev": "progress", "id": jid, "progress": (i + 1) / max(1, steps),
              "stage": f"denoise {i + 1}/{steps}"})
        if p.get("hang"):
            send({"ev": "log", "level": "info", "message": "hanging now", "id": jid})
            while True:
                time.sleep(3600)
    if cancel.is_set():
        time.sleep(float(p.get("slow_cancel_s", 0)))
        send({"ev": "failed", "id": jid, "error": "cancelled", "cancelled": True})
        return
    out = req.get("output")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump({"fake": True, "id": jid}, f)
    pool[f"{req['model']}/dit"] = (int(float(p.get("component_gb", 0.5)) * GB), time.time(),
                                   p.get("residency", "resident"))
    send({"ev": "done", "id": jid, "result": {"output": out, "seconds": round(time.time() - t0, 3),
                                              "peak_bytes": 123456789, "steps": steps}})


def main():
    if os.environ.get("FAKE_WORKER_START") == "fail":
        print("ImportError: no module named mlx", file=sys.stderr, flush=True)
        send({"ev": "log", "level": "error", "message": "worker failed to start: ImportError: no mlx"})
        sys.exit(1)
    threading.Thread(target=reader, daemon=True).start()
    while True:
        msg = runs.get()
        try:
            run(msg)
        finally:
            running_id[0] = None


if __name__ == "__main__":
    main()
