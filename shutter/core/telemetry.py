"""Opt-in, anonymous usage and crash reports ("Help The Dev Improve").

Off by default. Nothing is collected for sending, and nothing is sent, while it is off. When the person turns it on
(first-run window, Settings, or the one-time question after an update) the engine sends, to ENDPOINT:

* usage, at most once a week: for each model, how many requests ran and how many finished in that week. Models from
  Shuttrr's catalog go by their catalog id; any other model is counted as "custom" (a folder name could be personal).
* issues, at most once an hour: engine crashes, worker crashes, failed requests and app crashes (from the app), each
  with the error type, a short message and a stack trace, after `scrub` removed every file path, user name, home
  folder, file name, e-mail address and quoted text (a prompt could sit in quotes).
* with both: app and engine version, macOS major version, chip family ("M5", "M5 Max") and memory size rounded to a
  Mac size (16, 24, 32 ... GB), so a crash can be matched to the kind of Mac it happened on.

Never: prompts, outputs, pictures or audio, file or folder names, model paths, the Mac's name or serial number, any
account, any ID. There is no install ID; reports cannot be linked to each other or to a Mac. The endpoint stores
neither the IP address nor request headers.

State lives in <home>/telemetry/ (state.json: this week's counts, queued issues, the last payload sent, so Settings can
show exactly what went out). Sending runs on a background thread with a short timeout; a network failure keeps the
data for the next try and never affects a request.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

ENDPOINT = os.environ.get("SHUTTER_TELEMETRY_URL", "https://telemetry.shuttrr.sarthakrawool.com/v1/report")
SCHEMA = 1
WEEK_S = 7 * 24 * 3600
ISSUE_INTERVAL_S = 3600
TIMEOUT_S = 5.0
MAX_ISSUES = 50
MAX_TRACE = 3000
MAC_SIZES_GB = (8, 16, 18, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512)

WHAT = [
    "Which models you use: for each model, how many requests ran and finished in a week (at most one report a week).",
    "Crashes and failed requests: the error type, a short message and where in Shuttrr's code it happened, with "
    "file names, folder names and your user name removed.",
    "Shuttrr version, macOS version, chip family (for example M5) and memory size (for example 16 GB), so a crash "
    "can be matched to the kind of Mac.",
]
NEVER = [
    "Your prompts, pictures, videos or audio.",
    "File names, folder names or model paths.",
    "Your name, your Mac's name or serial number, your IP address, or any ID that could link reports together.",
]

_HOME = str(Path.home())
_USER = os.environ.get("USER") or Path.home().name
_PATH = re.compile(r"(?:(?<=\s)|(?<=^)|(?<=[\"'(=:,\[]))(?:~|/|[A-Za-z]:\\)[^\s\"'),\]]*")
_FILE = re.compile(r"\b[\w\-. ]{1,120}\.(?:png|jpe?g|webp|heic|mp4|mov|wav|mp3|m4a|flac|safetensors|gguf|json|txt|"
                   r"pt|pth|bin|npy|npz|py|log)\b", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_QUOTED = re.compile(r"([\"'])(?:(?!\1).){24,}?\1")
_PYFRAME = re.compile(r'File "[^"]*?(?:/site-packages/|/shutter/)([^"]+)", line (\d+), in (\S+)')


def scrub(text: Any, limit: int = 600) -> str:
    """Remove anything personal from an error message or stack trace: paths, the home folder, the user name, file
    names, e-mail addresses and long quoted text. Python frames keep only the module path inside the package."""
    s = str(text or "")
    frames: list[str] = []

    def keep(m: "re.Match[str]") -> str:      # Shuttrr's own module paths stay readable; set aside meanwhile
        frames.append(f'File "{m.group(1)}", line {m.group(2)}, in {m.group(3)}')
        return f"\x00{len(frames) - 1}\x00"
    s = _PYFRAME.sub(keep, s)
    if _HOME and len(_HOME) > 1:
        s = s.replace(_HOME, "~")
    s = _EMAIL.sub("<email>", s)
    s = _QUOTED.sub(r"\1<text>\1", s)
    s = _PATH.sub(lambda m: m.group(0) if _looks_like_module(m.group(0)) else "<path>", s)
    s = _FILE.sub("<file>", s)
    if _USER and len(_USER) > 2:
        s = re.sub(re.escape(_USER), "<user>", s, flags=re.I)
    s = re.sub("\x00(\\d+)\x00", lambda m: frames[int(m.group(1))], s)
    return s[:limit]


def _looks_like_module(p: str) -> bool:
    # "shutter/core/engine.py" style paths left by _PYFRAME start without a slash and never reach here; keep
    # API routes like /v1/jobs (no dot, short) so messages stay readable
    return p.startswith("/v1/") and "." not in p


def mac_class() -> dict[str, Any]:
    """macOS major version, chip family and memory rounded to a Mac size: the only things about the Mac sent."""
    out: dict[str, Any] = {"macos": (platform.mac_ver()[0] or "").split(".")[0] or None}
    try:
        from shutter.core.chip import chip_info
        info = chip_info()
        brand = str(info.get("name") or "")
        m = re.search(r"Apple (M\d+(?: (?:Pro|Max|Ultra))?)", brand)
        out["chip"] = m.group(1) if m else ("Apple" if "Apple" in brand else "other")
    except Exception:
        out["chip"] = None
    try:
        import subprocess
        b = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2).stdout)
        gb = b / 1024**3
        out["memory_gb"] = min(MAC_SIZES_GB, key=lambda x: abs(x - gb))
    except Exception:
        out["memory_gb"] = None
    return out


def public_model_id(model_id: str) -> str:
    try:
        from shutter.core.catalog import CATALOG
        return model_id if model_id in CATALOG else "custom"
    except Exception:
        return "custom"


class Telemetry:
    """One per engine. `enabled` reads the live config each time, so turning it off takes effect at once."""

    def __init__(self, config, engine_version: str = "", app_version_fn: Optional[Callable[[], str]] = None,
                 post: Optional[Callable[[str, dict], bool]] = None, clock: Callable[[], float] = time.time):
        self.config = config
        self.engine_version = engine_version
        self.app_version_fn = app_version_fn or (lambda: "")
        self._post = post or _http_post
        self.clock = clock
        self.dir = Path(config.home) / "telemetry"
        self.path = self.dir / "state.json"
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.state = self._load()

    # ---- settings
    @property
    def enabled(self) -> bool:
        return bool(getattr(self.config, "telemetry_enabled", False))

    def set(self, enabled: Optional[bool] = None, asked: Optional[bool] = None) -> dict[str, Any]:
        from shutter.core.config import Config
        fresh = Config.load(self.config.path)
        if enabled is not None:
            fresh.telemetry_enabled = bool(enabled)
            fresh.telemetry_asked = True
        if asked:
            fresh.telemetry_asked = True
        fresh.save()
        self.config.telemetry_enabled = fresh.telemetry_enabled
        self.config.telemetry_asked = fresh.telemetry_asked
        if enabled is False:
            with self._lock:     # turned off: forget what was collected but not sent
                self.state["usage"] = {}
                self.state["issues"] = []
                self._save()
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"enabled": self.enabled, "asked": bool(getattr(self.config, "telemetry_asked", False)),
                    "endpoint": ENDPOINT, "what": WHAT, "never": NEVER,
                    "last_payload": self.state.get("last_payload"), "last_sent_at": self.state.get("last_sent_at"),
                    "queued_issues": len(self.state.get("issues") or [])}

    # ---- collecting (only while enabled)
    def record_job(self, model_id: str, done: bool) -> None:
        if not self.enabled:
            return
        mid = public_model_id(model_id)
        with self._lock:
            u = self.state.setdefault("usage", {}).setdefault(mid, {"requests": 0, "finished": 0})
            u["requests"] += 1
            u["finished"] += 1 if done else 0
            self.state.setdefault("week_start", self.clock())
            self._save()

    def record_issue(self, kind: str, error_type: str = "", message: str = "", trace: str = "",
                     model_id: str = "", extra: Optional[dict] = None) -> bool:
        if not self.enabled:
            return False
        item = {"kind": str(kind)[:40], "error_type": scrub(error_type, 80), "message": scrub(message, 400),
                "trace": scrub(trace, MAX_TRACE), "model": public_model_id(model_id) if model_id else None,
                "at_day": time.strftime("%Y-%m-%d", time.gmtime(self.clock())), "count": 1}
        if extra:
            item.update({k: scrub(v, 600) if isinstance(v, str) else v for k, v in extra.items()})
        key = hashlib.sha256(json.dumps([item["kind"], item["error_type"], item["message"], item["trace"]])
                             .encode()).hexdigest()[:16]
        with self._lock:
            issues = self.state.setdefault("issues", [])
            for it in issues:
                if it.get("_key") == key:
                    it["count"] = int(it.get("count", 1)) + 1
                    self._save()
                    return True
            item["_key"] = key
            issues.append(item)
            del issues[:-MAX_ISSUES]
            self._save()
        return True

    # ---- engine crash detection: a marker file that a clean shutdown removes
    def engine_started(self) -> None:
        marker = self.dir / "running"
        if marker.exists():
            try:
                info = json.loads(marker.read_text() or "{}")
            except Exception:
                info = {}
            self.record_issue("engine_unexpected_exit", "EngineExit",
                              "the engine stopped without shutting down (crash, kill or power loss)",
                              extra={"engine_version_then": str(info.get("version") or "")})
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"version": self.engine_version, "at": int(self.clock())}))
        except OSError:
            pass

    def engine_stopped(self) -> None:
        try:
            (self.dir / "running").unlink()
        except OSError:
            pass

    # ---- sending
    def payload(self, force_usage: bool = False) -> Optional[dict[str, Any]]:
        with self._lock:
            now = self.clock()
            usage_due = bool(self.state.get("usage")) and (
                force_usage or now - float(self.state.get("week_start") or now) >= WEEK_S)
            issues_due = bool(self.state.get("issues")) and (
                force_usage or now - float(self.state.get("issues_sent_at") or 0) >= ISSUE_INTERVAL_S)
            if not usage_due and not issues_due:
                return None
            p: dict[str, Any] = {"schema": SCHEMA, "app_version": self.app_version_fn() or None,
                                 "engine_version": self.engine_version, **mac_class()}
            if usage_due:
                p["usage_days"] = max(1, round((now - float(self.state.get("week_start") or now)) / 86400))
                p["usage"] = dict(self.state["usage"])
            if issues_due:
                p["issues"] = [{k: v for k, v in it.items() if not k.startswith("_")} for it in self.state["issues"]]
            return p

    def send_now(self, force: bool = False) -> dict[str, Any]:
        """Send what is due (everything with force). Returns {"sent", "payload"}; never raises."""
        if not self.enabled:
            return {"sent": False, "reason": "off", "payload": None}
        p = self.payload(force_usage=force)
        if p is None:
            return {"sent": False, "reason": "nothing due", "payload": None}
        try:
            ok = bool(self._post(ENDPOINT, p))
        except Exception:
            ok = False
        if ok:
            with self._lock:
                now = self.clock()
                if "usage" in p:
                    self.state["usage"] = {}
                    self.state["week_start"] = now
                if "issues" in p:
                    self.state["issues"] = []
                    self.state["issues_sent_at"] = now
                self.state["last_payload"] = p
                self.state["last_sent_at"] = now
                self._save()
        return {"sent": ok, "payload": p}

    def start(self, interval_s: float = 600.0) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop():
            while not self._stop.wait(interval_s):
                try:
                    if self.enabled:
                        self.send_now()
                except Exception:
                    pass
        self._thread = threading.Thread(target=loop, name="shutter-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ---- storage
    def _load(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return {}

    def _save(self) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=1))
            os.replace(tmp, self.path)
        except OSError:
            pass


def _http_post(url: str, payload: dict) -> bool:
    import urllib.request
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": "Shuttrr"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        return 200 <= r.status < 300
