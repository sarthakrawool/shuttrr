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
        elif d.get("state") in ("downloading", "converting"):
            state = f"{d['state']} {d.get('done_bytes', 0) / GB:.1f} of {d.get('total_bytes', 0) / GB:.1f} GB"
        elif d.get("state") in ("failed", "stopped", "cancelled"):
            state = f"{d['state']}" + (f": {d['error']}" if d.get("error") else "")
        else:
            state = "not downloaded"
        approx = "~" if r.get("size_approximate") else " "
        lines.append(f"{r['id']:<28}{approx}{r['size_bytes'] / GB:>5.1f} GB  {state}")
        lines.append(f"{'':<28} {r['name']}. {r['summary']}")
        if r.get("needs_conversion") and not r["installed"]:
            lines.append(f"{'':<28} Downloads the original weights, then converts them on this Mac "
                         f"(about {r['built_size_bytes'] / GB:.0f} GB when built). Needs {r['min_memory_bytes'] / GB:.0f} GB "
                         f"of memory" + ("" if r.get("fits_this_mac", True) else "; too big for this Mac") + ".")
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
        if args.quiet or time.time() - last[0] < 2 and s.get("state") in ("downloading", "converting"):
            return
        last[0] = time.time()
        total = s.get("total_bytes") or 1
        done = s.get("done_bytes", 0)
        what = "converting " if s.get("state") == "converting" else ""
        print(f"\r{what}{done / GB:6.1f} of {total / GB:.1f} GB  {100 * done / total:5.1f}%", end="", flush=True)

    try:
        done = catalog.download(args.model, folder=args.dir, on_progress=show)
    except catalog.DownloadError as e:
        print(f"\nshutter: {e}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"\ninstalled in {done['dir']}")
        if done.get("verified_detail"):
            print(f"Checked: {done['verified_detail']}.")
        for n in done.get("notes") or []:
            print(n)
    return 0


TASKS = {"t2v": "text to video", "i2v": "image to video", "ti2v": "text or image to video", "lora": "LoRA"}
PART_NAMES = {"dit": "video model", "dit_high": "video model, high-noise expert",
              "dit_low": "video model, low-noise expert", "text_encoder": "text encoder", "vae": "VAE",
              "tokenizer": "tokenizer", "lora": "LoRA", "vae_encoder": "VAE encoder"}


def _part_name(n: str) -> str:
    return PART_NAMES.get(n, n.replace("_", " "))


def _this_mac(ck: Any) -> dict[str, Any]:
    """Can this Mac run `ck` at the lowest memory step? {"total_bytes", "fits" (True/False/None), "reasons"}."""
    from shutter.core.catalog import total_memory
    from shutter.families import family

    total = total_memory()
    out: dict[str, Any] = {"total_bytes": total, "fits": None, "reasons": []}
    if not total or ck.task == "lora" or "dim" not in (ck.config or {}):
        return out
    try:
        reasons = family(ck.family).check_memory(ck, total)
    except Exception as e:  # a half-recognised checkpoint: say we could not tell
        out["reasons"] = [f"could not work out its memory ({type(e).__name__})"]
        return out
    out["fits"] = not reasons
    out["reasons"] = reasons
    return out


def inspect_text(ck: Any, mac: dict[str, Any]) -> str:
    title = ck.label or "Unrecognised Wan checkpoint"
    lines = [title]
    kind = [x for x in (ck.variant, TASKS.get(ck.task, ck.task)) if x]
    if kind:
        lines[0] += f"  ({', '.join(kind)})"
    lines.append(f"path      {ck.path}")
    if ck.task == "lora":
        lines.append("format    a LoRA: give it with --lora to `shutter convert`, or add it to a request")
    elif ck.needs_conversion:
        lines.append("format    original weights: convert it first (shutter convert PATH OUT)")
    else:
        lines.append("format    a Shuttrr build: runs as it is")
    if ck.parts:
        lines.append("parts")
        for name, p in ck.parts.items():
            size = (_gb(p.total_bytes) if p.total_bytes >= GB // 10 else f"{p.total_bytes / 1024**2:.1f} MB") \
                if p.total_bytes else ""
            extra = []
            if p.n_blocks and p.role != "text_encoder":
                extra.append(f"{p.n_blocks} blocks")
            if p.quant:
                extra.append(f"{p.quant.get('bits')}-bit")
            elif p.dtype:
                extra.append(p.dtype.lower())
            if p.layout and p.layout != "shuttrr":
                extra.append(f"{p.layout} layout")
            lines.append(f"  {_part_name(name):<32} {size:>9}  {', '.join(extra)}".rstrip())
    if ck.missing:
        lines.append("missing   " + ", ".join(_part_name(m) for m in ck.missing)
                     + " (shutter convert finds them in an installed Wan or LongCat model, or name them with "
                       "--text-encoder / --vae / --tokenizer)")
    for n in ck.notes:
        lines.append(f"note      {n}")
    for pr in ck.problems:
        lines.append(f"problem   {pr}")
    total = mac.get("total_bytes") or 0
    if mac.get("fits") is True:
        lines.append(f"this Mac  can run it ({_gb(total)} of memory)")
    elif mac.get("fits") is False:
        lines.append(f"this Mac  cannot run it: {' '.join(mac['reasons'])}")
    elif ck.task != "lora":
        lines.append("this Mac  " + (mac["reasons"][0] if mac.get("reasons") else "memory could not be checked"))
    if ck.problems:
        lines.append("verdict   cannot run, for the reasons above")
    elif mac.get("fits") is False:
        lines.append("verdict   supported, but too big for this Mac")
    elif ck.task == "lora":
        lines.append("verdict   usable as a LoRA")
    else:
        lines.append("verdict   ready to " + ("convert" if ck.needs_conversion else "run"))
    return "\n".join(lines)


def cmd_inspect(args: argparse.Namespace) -> int:
    from shutter.families import inspect

    path = os.path.expanduser(args.path)
    if not os.path.exists(path):
        print(f"shutter: {path} does not exist", file=sys.stderr)
        return 1
    try:
        ck = inspect(path)
    except (OSError, ValueError) as e:
        print(f"shutter: could not read {path}: {e}", file=sys.stderr)
        return 1
    if ck is None:
        msg = "not a model Shuttrr recognises (supported families: Wan 2.1 / 2.2)"
        _print(args, {"path": path, "recognised": False, "problems": [msg]}, f"{path}: {msg}")
        return 1
    mac = _this_mac(ck)
    data = ck.to_dict()
    data.update(recognised=True, this_mac=mac)
    _print(args, data, inspect_text(ck, mac))
    return 0 if ck.ok else 1


def parse_lora(s: str) -> tuple[str, float]:
    from shutter.families.wan.sources import parse_lora as parse

    path, strength = parse(s)
    if not os.path.isfile(os.path.expanduser(path)):
        raise argparse.ArgumentTypeError(f"LoRA file not found: {path}")
    return os.path.expanduser(path), strength


def cmd_convert(args: argparse.Namespace) -> int:
    from pathlib import Path

    from shutter.core import catalog
    from shutter.families import inspect
    from shutter.families.wan import sources
    from shutter.families.wan.convert import ConvertError, convert

    source = Path(os.path.expanduser(args.source))
    out = Path(os.path.expanduser(args.out))
    given = {"text_encoder": args.text_encoder, "vae": args.vae, "tokenizer": args.tokenizer}
    given = {k: Path(os.path.expanduser(v)) for k, v in given.items() if v}
    cfg = Config.load()
    log = sys.stderr if args.json else sys.stdout

    def report(line: str) -> None:
        print(line, file=log, flush=True)

    ck = inspect(source) if source.exists() else None
    if ck is not None and ck.needs_conversion and ck.task != "lora":
        todo = [n for n in sources.needed(ck) if n not in given]
        if todo:
            dirs = sources.candidate_dirs(catalog._model_roots(cfg), exclude=[source, out])
            found = sources.find_parts(ck, dirs, names=todo)
            for name, d in found.items():
                report(f"Using the {_part_name(name)} from {d}")
            given.update(found)
    try:
        man = convert(source, out, bits=args.bits or None, loras=args.lora or None,
                      text_encoder=given.get("text_encoder"), vae=given.get("vae"),
                      tokenizer=given.get("tokenizer"), report=report)
    except ConvertError as e:
        print(f"shutter: {e}", file=sys.stderr)
        if ck is not None and "vae" in sources.needed(ck) and "vae" not in given:
            print("The VAE has to come from the original model (Wan2.1_VAE.pth or Wan2.2_VAE.pth) or ComfyUI's "
                  "wan_2.1_vae / wan2.2_vae .safetensors: name it with --vae. A Shuttrr build keeps its VAE in a "
                  "converted form that cannot be converted again.", file=sys.stderr)
        return 2
    if args.add:
        catalog.register_build(cfg, args.add, out, adapter="wan")
        report(f"Added {args.add} to {cfg.path}. Restart the engine (or the app) to use it.")
    if args.json:
        print(json.dumps(man, indent=2))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from shutter.core import catalog

    rows = catalog.doctor()
    mark = {True: "ok  ", False: "FAIL", None: "note"}
    _print(args, {"checks": rows}, "\n".join(f"{mark[r['ok']]}  {r['check']}: {r['detail']}" for r in rows))
    return 1 if any(r["ok"] is False for r in rows) else 0


def _engine_pids(port: int) -> list[int]:
    """Process ids of running Shuttrr engines: whatever listens on the engine port, plus `shutter serve` processes."""
    import subprocess

    pids: set[int] = set()
    for cmd in (["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], ["pgrep", "-f", "shutter(\\.cli)? serve"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        pids.update(int(x) for x in out.split() if x.isdigit())
    pids.discard(os.getpid())
    return sorted(pids)


def stop_engine(port: int, timeout: float = 10.0) -> list[int]:
    """Stop running engines (SIGTERM, then SIGKILL after `timeout`). Returns the pids that were running."""
    import signal

    pids = _engine_pids(port)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    end = time.time() + timeout
    alive = list(pids)
    while alive and time.time() < end:
        time.sleep(0.2)
        alive = [p for p in alive if _alive(p)]
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return pids


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def uninstall(home: str, *, keep_models: bool = False, port: int = 8484) -> dict[str, Any]:
    """Stop the engine and delete the Shuttrr home folder. With keep_models, <home>/models stays.
    Returns {"stopped": [pids], "removed": [paths], "kept": [paths]}."""
    import shutil
    from pathlib import Path

    root = Path(os.path.expanduser(home)).resolve()
    if root == Path.home().resolve() or len(root.parts) < 3:
        raise ValueError(f"refusing to delete {root}: it does not look like a Shuttrr folder")
    stopped = stop_engine(port)
    removed: list[str] = []
    kept: list[str] = []
    if not root.exists():
        return {"stopped": stopped, "removed": removed, "kept": kept}
    for child in sorted(root.iterdir()):
        if keep_models and child.name == "models":
            kept.append(str(child))
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except OSError:
                pass
        removed.append(str(child))
    if not kept:
        try:
            root.rmdir()
        except OSError:
            pass
    return {"stopped": stopped, "removed": removed, "kept": kept}


def cmd_uninstall(args: argparse.Namespace) -> int:
    cfg = Config.load()
    home = os.path.expanduser(cfg.home)
    what = f"{home} (engine, settings, job history" + (")" if args.keep_models else ", downloaded models)")
    if not args.yes:
        if not sys.stdin.isatty():
            print("shutter: uninstall needs --yes when it cannot ask", file=sys.stderr)
            return 2
        ans = input(f"This stops the Shuttrr engine and deletes {what}. Continue? [y/N] ")
        if ans.strip().lower() not in ("y", "yes"):
            print("Nothing was removed.")
            return 1
    try:
        res = uninstall(home, keep_models=args.keep_models, port=cfg.port)
    except ValueError as e:
        print(f"shutter: {e}", file=sys.stderr)
        return 2
    models_dir = cfg.extra.get("models_dir")
    if args.json:
        print(json.dumps({**res, "home": home}, indent=2))
    else:
        if res["stopped"]:
            print(f"Stopped the engine ({', '.join(map(str, res['stopped']))}).")
        print(f"Removed {home}" + (f", kept {res['kept'][0]}" if res["kept"] else "") + ".")
        if models_dir:
            print(f"Models in {models_dir} were not touched.")
        print("To remove the app too, drag Shuttrr.app to the Trash.")
    return 0


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

    s = sub.add_parser("uninstall", help="stop the engine and delete ~/.shutter (asks first)")
    s.add_argument("--keep-models", action="store_true", help="keep ~/.shutter/models")
    s.add_argument("--yes", "-y", action="store_true", help="do not ask")
    s.set_defaults(func=cmd_uninstall)

    s = sub.add_parser("inspect", help="say what a model folder or file is, and whether this Mac can run it")
    s.add_argument("path", help="a folder (Hugging Face repo, Shuttrr build) or a .safetensors file")
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("convert", help="convert a Wan checkpoint into Shuttrr's 8-bit low-memory build")
    s.add_argument("source", help="folder or .safetensors file (see `shutter inspect`)")
    s.add_argument("out", help="empty folder for the build")
    s.add_argument("--bits", type=int, default=8, choices=[8, 4, 0], help="8 (default), 4, or 0 to keep bf16")
    s.add_argument("--lora", action="append", type=parse_lora, default=[], metavar="FILE[:STRENGTH]",
                   help="LoRA merged into the build before quantizing (repeatable; strength defaults to 1)")
    s.add_argument("--text-encoder", metavar="DIR", help="folder with the umT5 text encoder")
    s.add_argument("--vae", metavar="FILE", help="the Wan VAE (.safetensors or .pth), or a folder holding it")
    s.add_argument("--tokenizer", metavar="DIR", help="folder with the umT5 tokenizer")
    s.add_argument("--add", metavar="MODEL_ID", help="also add the build to ~/.shutter/config.json under this id")
    s.set_defaults(func=cmd_convert)

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
