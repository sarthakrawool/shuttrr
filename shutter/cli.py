"""Command line: `shutter serve` runs the daemon; the other commands talk to it over HTTP."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Optional, Sequence

from shutter.core.config import Config

GB = 1024**3
TERMINAL = {"done", "failed", "cancelled"}


class ClientError(Exception):
    pass


# ---------------------------------------------------------------- HTTP client


class Client:
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def request(self, method: str, path: str, body: Any = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base_url + path, data=data, method=method)
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                msg = json.loads(raw).get("error") or raw.decode()
            except Exception:
                msg = raw.decode("utf-8", "replace")
            raise ClientError(f"HTTP {e.code}: {msg}") from None
        except (urllib.error.URLError, OSError) as e:
            raise ClientError(
                f"cannot reach the Shuttrr engine at {self.base_url} ({getattr(e, 'reason', e)}). "
                f"Start it with `shutter serve`.") from None
        return json.loads(raw) if raw else None


def _client(args: argparse.Namespace) -> Client:
    cfg = None
    try:
        cfg = Config.load()
    except Exception:
        pass
    url = args.url or os.environ.get("SHUTTER_URL")
    if not url:
        host = (cfg.host if cfg else "127.0.0.1")
        if host in ("0.0.0.0", "::", ""):
            host = "127.0.0.1"
        url = f"http://{host}:{cfg.port if cfg else 8484}"
    key = os.environ.get("SHUTTER_API_KEY") or (cfg.api_key if cfg else None)
    return Client(url, key)


# ---------------------------------------------------------------- formatting


def _gb(n: Any) -> str:
    try:
        return f"{float(n) / GB:.1f} GB"
    except (TypeError, ValueError):
        return "-"


def _job_line(j: dict[str, Any]) -> str:
    pct = f"{100 * (j.get('progress') or 0):5.1f}%"
    detail = j.get("stage") if j.get("state") == "running" else j.get("message")
    return f"{j['id']}  {j['state']:<15} {pct}  {j['request']['model']:<26} {detail or ''}".rstrip()


def _print(args: argparse.Namespace, data: Any, text: str) -> None:
    print(json.dumps(data, indent=2) if args.json else text)


def parse_param(s: str) -> tuple[str, Any]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--param needs key=value, got {s!r}")
    k, v = s.split("=", 1)
    k = k.strip()
    if not k:
        raise argparse.ArgumentTypeError(f"--param needs a key, got {s!r}")
    try:
        return k, json.loads(v)
    except ValueError:
        return k, v


# ---------------------------------------------------------------- commands


def cmd_serve(args: argparse.Namespace) -> int:
    from shutter.core.engine import Engine
    from shutter.server.api import check_bind, serve

    cfg = Config.load()
    host = args.host or cfg.host
    port = args.port or cfg.port
    try:
        check_bind(host, cfg.api_key)
    except ValueError as e:
        print(f"shutter: {e}", file=sys.stderr)
        return 2
    engine = Engine(cfg)
    print(f"Shuttrr {__import__('shutter').__version__} on http://{host}:{port}  (home {cfg.home})")
    serve(engine, host=host, port=port)
    return 0


def build_request(args: argparse.Namespace) -> dict[str, Any]:
    body: dict[str, Any] = {"model": args.model, "seed": args.seed, "priority": args.priority,
                            "client": f"shutter-cli/{__import__('shutter').__version__}"}
    if args.prompt is not None:
        body["prompt"] = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            body["prompt"] = f.read().strip()
    inputs = {}
    for k in ("image", "audio", "video"):
        v = getattr(args, k, None)
        if v:
            inputs[k] = os.path.abspath(os.path.expanduser(v))
    for role, v in getattr(args, "input", None) or []:
        inputs[role] = os.path.abspath(os.path.expanduser(str(v)))
    if inputs:
        body["inputs"] = inputs
    if args.output:
        body["output"] = os.path.abspath(os.path.expanduser(args.output))
    if args.duration is not None:
        body["duration_s"] = args.duration
    for k in ("negative_prompt", "width", "height", "frames", "fps", "steps", "guidance"):
        v = getattr(args, k, None)
        if v is not None:
            body[k] = v
    params = dict(args.param or [])
    if getattr(args, "mode", None):
        params["mode"] = args.mode
    if getattr(args, "efficiency", None) is not None:
        params["efficiency"] = args.efficiency
    if params:
        body["params"] = params
    return body


def parse_mode(s: str) -> str:
    k = s.strip().lower().replace("-", "_").replace(" ", "_")
    if k not in ("fast", "low_memory"):
        raise argparse.ArgumentTypeError(f"must be fast or low_memory, got {s!r}")
    return k


def parse_efficiency(s: str) -> float:
    try:
        x = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a number from 0 to 1, got {s!r}") from None
    if not 0.0 <= x <= 1.0:
        raise argparse.ArgumentTypeError(f"must be from 0 to 1, got {s}")
    return x


def settings_text(v: dict[str, Any]) -> str:
    mode = v.get("mode", "")
    lines = [f"mode        {'low memory' if mode == 'low_memory' else mode}",
             f"efficiency  {float(v.get('efficiency') or 0):.2f}" + (" (used in low memory mode)" if mode == "fast" else ""),
             f"rung        {v.get('rung', '')}"]
    m = v.get("measured_for")
    if isinstance(m, dict):
        lines.append(f"measured    {m.get('model', '?')} at {m.get('size', '?')}")
    lines.append("ladder")
    for r in v.get("ladder") or []:
        here = "*" if r.get("id") == v.get("rung") else " "
        peak = _gb(r["peak_bytes"]) if r.get("peak_bytes") is not None else "–"
        secs = f"{r['seconds']:.0f} s" if isinstance(r.get("seconds"), (int, float)) else "–"
        lines.append(f"  {here} {float(r.get('at', 0)):.2f}  {r.get('id', ''):<13} {peak:>9}  {secs:>6}  {r.get('detail', '')}")
    return "\n".join(lines)


def cmd_settings(args: argparse.Namespace) -> int:
    c = _client(args)
    if args.mode is None and args.efficiency is None:
        v = c.request("GET", "/v1/settings")
    else:
        body: dict[str, Any] = {}
        if args.mode is not None:
            body["mode"] = args.mode
        if args.efficiency is not None:
            body["efficiency"] = args.efficiency
        v = c.request("POST", "/v1/settings", body)
    _print(args, v, settings_text(v))
    return 0


def wait_for(client: Client, job_id: str, poll_s: float = 1.0, out=sys.stdout,
             timeout: Optional[float] = None) -> dict[str, Any]:
    last = None
    t0 = time.monotonic()
    while True:
        j = client.request("GET", f"/v1/jobs/{job_id}")
        line = _job_line(j)
        if line != last:
            print(line, file=out, flush=True)
            last = line
        if j["state"] in TERMINAL:
            return j
        if timeout is not None and time.monotonic() - t0 > timeout:
            raise ClientError(f"timed out waiting for job {job_id}")
        time.sleep(poll_s)


def cmd_submit(args: argparse.Namespace) -> int:
    c = _client(args)
    job = c.request("POST", "/v1/jobs", build_request(args))
    if not args.wait:
        _print(args, job, job["id"])
        return 0
    if not args.json:
        print(f"submitted {job['id']}", flush=True)
    j = wait_for(c, job["id"], args.poll, out=sys.stderr if args.json else sys.stdout)
    if args.json:
        print(json.dumps(j, indent=2))
    elif j["state"] == "done":
        r = j.get("result", {})
        print(f"done in {r.get('seconds', '?')} s, peak {_gb(r.get('peak_bytes'))}: {r.get('output', '')}")
    else:
        print(f"{j['state']}: {j.get('message', '')}")
    return 0 if j["state"] == "done" else 1


def cmd_status(args: argparse.Namespace) -> int:
    c = _client(args)
    if args.id:
        j = c.request("GET", f"/v1/jobs/{args.id}")
        lines = [_job_line(j)]
        for k in ("message", "stage"):
            if j.get(k):
                lines.append(f"  {k}: {j[k]}")
        for k, v in (j.get("result") or {}).items():
            if k != "traceback":
                lines.append(f"  {k}: {v}")
        _print(args, j, "\n".join(lines))
        return 0
    path = f"/v1/jobs?limit={args.limit}" + (f"&state={args.state}" if args.state else "")
    d = c.request("GET", path)
    text = "\n".join(_job_line(j) for j in d["jobs"]) or "no jobs"
    _print(args, d, text)
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    j = _client(args).request("DELETE", f"/v1/jobs/{args.id}")
    _print(args, j, _job_line(j))
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    d = _client(args).request("GET", "/v1/models")
    rows = []
    for m in d["models"]:
        status = "error: " + m["error"] if m.get("error") else ("installed" if m.get("installed") else "not installed")
        rows.append(f"{m['id']:<28} {m['family']:<16} {status}  {m.get('dir', '')}")
    _print(args, d, "\n".join(rows) or "no models configured")
    return 0


def cmd_system(args: argparse.Namespace) -> int:
    s = _client(args).request("GET", "/v1/system")
    m, q, g = s["memory"], s["queue"], s["gpu_lock"]
    lines = [
        f"Shuttrr {s['version']}",
        f"memory   {_gb(m['total'] - m['available'])} used of {_gb(m['total'])}, "
        f"{_gb(m['available'])} available, swap {_gb(m['swap_used'])}, pressure {m['pressure']}",
        f"headroom {_gb(s['guard']['headroom_bytes'])}"
        + (" (automatic: an eighth of memory, 2 to 8 GB)" if s["guard"].get("headroom_auto") else ""),
    ]
    for p in s["guard"]["protected"]:
        lines.append(f"protect  {p['name']}: {p['busy'] or 'idle'}")
    lines.append(f"queue    {q['queued']} queued, {q['waiting_memory']} waiting, {q['running']} running, "
                 f"{q['done']} done, {q['failed']} failed" + (" (paused)" if q.get("paused") else ""))
    holder = (g.get("holder") or {}).get("label") if g.get("held") else None
    lines.append(f"gpu lock {'held by ' + str(holder) if g.get('held') else 'free'}")
    lines.append(f"pool     {_gb(s['pool']['total_bytes'])} in {len(s['pool']['entries'])} components")
    for e in s["pool"]["entries"]:
        lines.append(f"         {e['key']}  {_gb(e['size_bytes'])}  idle {e['idle_s']:.0f} s")
    _print(args, s, "\n".join(lines))
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    cfg = Config.load()
    if os.path.exists(cfg.path) and not args.force:
        print(f"{cfg.path} already exists (use --force to overwrite it with defaults)")
        return 1
    fresh = Config(home=cfg.home, path=cfg.path)
    fresh.ensure_dirs()
    print(f"wrote {fresh.save()}")
    return 0


def cmd_catalog(args: argparse.Namespace) -> int:
    from shutter.core import catalog

    rows = catalog.catalog_rows()
    lines = []
    for r in rows:
        d = r["download"] or {}
        if r["installed"]:
            state = "installed"
        elif d.get("state") == "downloading":
            state = f"downloading {d.get('done_bytes', 0) / GB:.1f} of {d.get('total_bytes', 0) / GB:.1f} GB"
        elif d.get("state") in ("failed", "stopped", "cancelled"):
            state = f"{d['state']}" + (f": {d['error']}" if d.get("error") else "")
        else:
            state = "not downloaded"
        lines.append(f"{r['id']:<28} {r['size_bytes'] / GB:>5.1f} GB  {state}")
        lines.append(f"{'':<28} {r['name']}. {r['summary']}")
    _print(args, {"models": rows}, "\n".join(lines))
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from shutter.core import catalog

    if args.cancel:
        stopped = catalog.cancel(args.model)
        print("stopped" if stopped else "no download is running for that model")
        return 0
    last = [0.0]

    def show(s: dict[str, Any]) -> None:
        if args.quiet or time.time() - last[0] < 2 and s.get("state") == "downloading":
            return
        last[0] = time.time()
        total = s.get("total_bytes") or 1
        done = s.get("done_bytes", 0)
        print(f"\r{done / GB:6.1f} of {total / GB:.1f} GB  {100 * done / total:5.1f}%", end="", flush=True)

    try:
        done = catalog.download(args.model, folder=args.dir, on_progress=show)
    except catalog.DownloadError as e:
        print(f"\nshutter: {e}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"\ninstalled in {done['dir']}")
        for n in done.get("notes") or []:
            print(n)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from shutter.core import catalog

    rows = catalog.doctor()
    mark = {True: "ok  ", False: "FAIL", None: "note"}
    _print(args, {"checks": rows}, "\n".join(f"{mark[r['ok']]}  {r['check']}: {r['detail']}" for r in rows))
    return 1 if any(r["ok"] is False for r in rows) else 0


# ---------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="shutter", description="Shuttrr: runs big video and voice models in less memory")
    p.add_argument("--url", help="daemon URL (default from config, http://127.0.0.1:8484)")
    p.add_argument("--json", action="store_true", help="print raw JSON")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the daemon")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("submit", help="queue a job")
    s.add_argument("--model", required=True)
    s.add_argument("--prompt")
    s.add_argument("--prompt-file")
    s.add_argument("--negative-prompt")
    s.add_argument("--image", help="input image (inputs.image)")
    s.add_argument("--audio", help="input audio (inputs.audio)")
    s.add_argument("--video", help="input video (inputs.video)")
    s.add_argument("--input", action="append", type=parse_param, default=[], metavar="ROLE=PATH",
                   help="any other input file (repeatable)")
    s.add_argument("--output")
    s.add_argument("--width", type=int)
    s.add_argument("--height", type=int)
    s.add_argument("--frames", type=int)
    s.add_argument("--fps", type=float)
    s.add_argument("--steps", type=int)
    s.add_argument("--guidance", type=float)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--duration", type=float, help="seconds of video")
    s.add_argument("--priority", type=int, default=0)
    s.add_argument("--param", action="append", type=parse_param, default=[], metavar="K=V",
                   help="adapter option; value parsed as JSON when possible (repeatable)")
    s.add_argument("--mode", type=parse_mode, help="fast or low_memory (default: the engine setting)")
    s.add_argument("--efficiency", type=parse_efficiency, metavar="X",
                   help="low memory efficiency 0..1 (default: the engine setting)")
    s.add_argument("--wait", action="store_true", help="wait for the job and show progress")
    s.add_argument("--poll", type=float, default=1.0, help=argparse.SUPPRESS)
    s.set_defaults(func=cmd_submit)

    s = sub.add_parser("status", help="list jobs, or show one")
    s.add_argument("id", nargs="?")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--state")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("cancel", help="cancel a job")
    s.add_argument("id")
    s.set_defaults(func=cmd_cancel)

    s = sub.add_parser("settings", help="show or change the default mode and efficiency")
    s.add_argument("--mode", type=parse_mode, help="fast or low_memory")
    s.add_argument("--efficiency", type=parse_efficiency, metavar="X", help="0..1, used in low memory mode")
    s.set_defaults(func=cmd_settings)

    sub.add_parser("models", help="list models").set_defaults(func=cmd_models)
    sub.add_parser("system", help="memory, queue, GPU lock, loaded models").set_defaults(func=cmd_system)

    sub.add_parser("catalog", help="models Shuttrr can download").set_defaults(func=cmd_catalog)
    s = sub.add_parser("download", help="download a model and set it up for this Mac")
    s.add_argument("model", help="model id from `shutter catalog`")
    s.add_argument("--dir", help="where to put it (default ~/.shutter/models/<name>)")
    s.add_argument("--cancel", action="store_true", help="stop a running download (it can resume later)")
    s.add_argument("--quiet", action="store_true", help="no progress output")
    s.set_defaults(func=cmd_download)
    sub.add_parser("doctor", help="check that this Mac is ready").set_defaults(func=cmd_doctor)

    s = sub.add_parser("init", help="write a default ~/.shutter/config.json")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_init)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    from shutter.core.config import add_private_bin
    add_private_bin()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except ClientError as e:
        print(f"shutter: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
