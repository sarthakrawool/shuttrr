"""Memory safety for Shutter: probes, protected services, the guard and the GPU lock.

Rule 2 of the architecture: never push the Mac into swap. Before every load and every
job the `MemoryGuard` checks that the job's peak fits in what macOS can give right now
(free + reclaimable memory, no fixed reserve), memory pressure and the protected services
(the user's chat model on oMLX, Splash, Strata). During a run the `Watchdog` frees caches
at the first sign of trouble (critical pressure or swap growth) and stops the run if it
does not ease, before the machine thrashes.

Everything here is stdlib only and imports on Linux so it can be tested in CI. The macOS
parsers are pure functions fed with captured command output.
"""

from __future__ import annotations

import errno
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence, Union, runtime_checkable

from shutter.types import GB, MemoryRefused

try:  # fcntl is POSIX only; Shutter targets macOS (and Linux for CI).
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

MiB = 1024**2
GiB = GB

PRESSURE_NORMAL = "normal"
PRESSURE_WARN = "warn"
PRESSURE_CRITICAL = "critical"


def fmt_gb(n: Union[int, float]) -> str:
    """Bytes as a human GB figure with one decimal, e.g. '12.3 GB'."""
    return f"{n / GB:.1f} GB"


# --------------------------------------------------------------------------------------
# Samples and probes
# --------------------------------------------------------------------------------------


@dataclass
class MemorySample:
    """One reading of system memory, in bytes.

    `available` = free + inactive + purgeable + speculative + active file cache (macOS), or MemAvailable on
    Linux. The raw fields are None when the platform does not report them.
    """

    total: int
    available: int
    swap_used: int
    pressure: str = PRESSURE_NORMAL  # "normal" | "warn" | "critical"
    ts: float = 0.0
    free: Optional[int] = None
    inactive: Optional[int] = None
    purgeable: Optional[int] = None
    speculative: Optional[int] = None
    wired: Optional[int] = None
    compressed: Optional[int] = None
    swap_total: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.ts:
            self.ts = time.time()

    @property
    def used(self) -> int:
        return max(0, self.total - self.available)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["used"] = self.used
        d["total_gb"] = round(self.total / GB, 1)
        d["available_gb"] = round(self.available / GB, 1)
        d["swap_used_gb"] = round(self.swap_used / GB, 1)
        return d


@runtime_checkable
class SystemProbe(Protocol):
    def sample(self) -> MemorySample: ...


# ---- pure parsers (unit-testable on any OS) -------------------------------------------

_VM_PAGE_SIZE_RE = re.compile(r"page size of (\d+) bytes")
_VM_LINE_RE = re.compile(r'^\s*"?([^":]+?)"?\s*:\s*(\d+)\.?\s*$')

# vm_stat label -> our field name
_VM_FIELDS = {
    "Pages free": "free",
    "Pages active": "active",
    "Pages inactive": "inactive",
    "Pages speculative": "speculative",
    "Pages throttled": "throttled",
    "Pages wired down": "wired",
    "Pages purgeable": "purgeable",
    "Pages occupied by compressor": "compressed",
    "Pages stored in compressor": "stored_in_compressor",
    "File-backed pages": "file_backed",
    "Anonymous pages": "anonymous",
}


def parse_vm_stat(text: str, page_size: Optional[int] = None) -> dict[str, int]:
    """Parse `vm_stat` output into bytes.

    Returns a dict with keys such as free, active, inactive, speculative, wired,
    purgeable, compressed (pages occupied by the compressor), plus `page_size` and
    `available` (= free + inactive + purgeable + speculative + active file-backed pages, which macOS can drop
    without swapping; see below). The page size printed in
    vm_stat's own header wins over `page_size`, because the counts are in that unit.
    """
    m = _VM_PAGE_SIZE_RE.search(text)
    if m:
        page_size = int(m.group(1))
    if not page_size:
        page_size = 4096
    out: dict[str, int] = {"page_size": page_size}
    for line in text.splitlines():
        lm = _VM_LINE_RE.match(line)
        if not lm:
            continue
        label, value = lm.group(1).strip(), int(lm.group(2))
        key = _VM_FIELDS.get(label)
        if key:
            out[key] = value * page_size
    # Active file-backed pages are clean copies of files: macOS drops them without swapping. vm_stat does not
    # split the inactive list into file and anonymous pages, so `file_backed - inactive` (never below 0) is a lower
    # bound of the active file pages: nothing is counted twice.
    out["reclaimable_file"] = max(0, out.get("file_backed", 0) - out.get("inactive", 0))
    out["available"] = sum(out.get(k, 0) for k in ("free", "inactive", "purgeable", "speculative",
                                                     "reclaimable_file"))
    return out


_SWAP_RE = re.compile(r"(total|used|free)\s*=\s*([\d.]+)\s*([KMGT]?)", re.IGNORECASE)
_UNIT = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def parse_swapusage(text: str) -> dict[str, int]:
    """Parse `sysctl vm.swapusage` ('total = 2048.00M  used = 1024.50M  free = ...').

    Returns {'total', 'used', 'free'} in bytes; missing values are 0.
    """
    out = {"total": 0, "used": 0, "free": 0}
    for key, num, unit in _SWAP_RE.findall(text):
        out[key.lower()] = int(float(num) * _UNIT[unit.upper()])
    return out


def parse_pressure_level(text: str) -> str:
    """Map kern.memorystatus_vm_pressure_level (1 normal, 2 warn, 4 critical).

    Unparseable output maps to "normal" so a broken sysctl does not block the engine
    forever; the swap and available-memory checks still apply.
    """
    try:
        level = int(text.strip().split()[-1])
    except (ValueError, IndexError):
        return PRESSURE_NORMAL
    if level >= 4:
        return PRESSURE_CRITICAL
    if level >= 2:
        return PRESSURE_WARN
    return PRESSURE_NORMAL


def parse_meminfo(text: str) -> dict[str, int]:
    """Parse /proc/meminfo into bytes (keys as in the file, e.g. 'MemAvailable')."""
    out: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.replace(":", " ").split()
        if len(parts) < 2:
            continue
        try:
            val = int(parts[1])
        except ValueError:
            continue
        if len(parts) >= 3 and parts[2].lower() == "kb":
            val *= 1024
        out[parts[0]] = val
    return out


# ---- probes --------------------------------------------------------------------------

Runner = Callable[[Sequence[str]], str]


def _run(cmd: Sequence[str]) -> str:
    return subprocess.run(
        list(cmd), capture_output=True, text=True, timeout=10, check=True
    ).stdout


class MacProbe:
    """Reads macOS memory state via sysctl and vm_stat. `runner` is injectable."""

    def __init__(self, runner: Runner = _run):
        self._run = runner
        self._total: Optional[int] = None
        self._page_size: Optional[int] = None

    def _static(self) -> tuple[int, int]:
        if self._total is None:
            lines = self._run(["sysctl", "-n", "hw.memsize", "hw.pagesize"]).split()
            self._total = int(lines[0])
            self._page_size = int(lines[1]) if len(lines) > 1 else 16384
        return self._total, self._page_size or 16384

    def sample(self) -> MemorySample:
        total, page_size = self._static()
        vm = parse_vm_stat(self._run(["vm_stat"]), page_size)
        try:
            swap = parse_swapusage(self._run(["sysctl", "-n", "vm.swapusage"]))
        except (subprocess.SubprocessError, OSError):
            swap = {"total": 0, "used": 0, "free": 0}
        try:
            pressure = parse_pressure_level(
                self._run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
            )
        except (subprocess.SubprocessError, OSError):
            pressure = PRESSURE_NORMAL
        return MemorySample(
            total=total,
            available=vm["available"],
            swap_used=swap["used"],
            pressure=pressure,
            ts=time.time(),
            free=vm.get("free"),
            inactive=vm.get("inactive"),
            purgeable=vm.get("purgeable"),
            speculative=vm.get("speculative"),
            wired=vm.get("wired"),
            compressed=vm.get("compressed"),
            swap_total=swap["total"],
        )


class LinuxProbe:
    """Reads /proc/meminfo so the engine and its tests run on Linux CI.

    Pressure is always "normal" (Linux PSI is not mapped; CI only needs a working probe).
    """

    def __init__(self, path: str = "/proc/meminfo"):
        self.path = path

    def sample(self) -> MemorySample:
        with open(self.path) as f:
            mi = parse_meminfo(f.read())
        total = mi.get("MemTotal", 0)
        available = mi.get("MemAvailable", mi.get("MemFree", 0) + mi.get("Cached", 0))
        swap_total = mi.get("SwapTotal", 0)
        return MemorySample(
            total=total,
            available=available,
            swap_used=max(0, swap_total - mi.get("SwapFree", 0)),
            pressure=PRESSURE_NORMAL,
            ts=time.time(),
            free=mi.get("MemFree"),
            inactive=mi.get("Inactive"),
            swap_total=swap_total,
        )


class FakeProbe:
    """Settable probe for tests.

    `sequence` is a script of readings returned in order by sample(): each item is a
    MemorySample (returned as is) or a dict of field overrides that are applied to the
    current values and persist. When the script runs out, the current values are
    returned. Use `set(**fields)` to change the current values directly.
    """

    def __init__(
        self,
        total: int = 64 * GB,
        available: int = 48 * GB,
        swap_used: int = 0,
        pressure: str = PRESSURE_NORMAL,
        sequence: Optional[Iterable[Union[MemorySample, dict[str, Any]]]] = None,
    ):
        self.values: dict[str, Any] = dict(
            total=total, available=available, swap_used=swap_used, pressure=pressure
        )
        self.sequence: list[Union[MemorySample, dict[str, Any]]] = list(sequence or [])
        self.calls = 0

    def set(self, **fields: Any) -> None:
        self.values.update(fields)

    def push(self, *items: Union[MemorySample, dict[str, Any]]) -> None:
        self.sequence.extend(items)

    def sample(self) -> MemorySample:
        self.calls += 1
        if self.sequence:
            item = self.sequence.pop(0)
            if isinstance(item, MemorySample):
                return item
            self.values.update(item)
        return MemorySample(ts=time.time(), **self.values)


def default_probe() -> SystemProbe:
    """MacProbe on macOS, LinuxProbe on Linux."""
    if sys.platform == "darwin":
        return MacProbe()
    if os.path.exists("/proc/meminfo"):
        return LinuxProbe()
    raise RuntimeError(f"no memory probe for platform {sys.platform!r}")


# --------------------------------------------------------------------------------------
# Protected services
# --------------------------------------------------------------------------------------


@runtime_checkable
class ProtectedService(Protocol):
    name: str

    def busy(self) -> Optional[str]:
        """A human reason when the service holds memory we must not compete with."""
        ...


# fetch(method, url, headers, timeout) -> (status, body). Raises OSError/URLError when
# the server cannot be reached. Only "GET" is ever passed.
Fetcher = Callable[[str, str, dict[str, str], float], tuple[int, bytes]]


def _urllib_get(method: str, url: str, headers: dict[str, str], timeout: float) -> tuple[int, bytes]:
    if method != "GET":  # hard guarantee: Shutter never changes anything in oMLX
        raise ValueError(f"OmlxService is read-only; refused {method} {url}")
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        try:
            body = e.read()
        except Exception:
            body = b""
        return e.code, body


_LOADED_STATUS = {"loaded", "ready", "active", "running", "serving"}


def _entry_name(e: dict[str, Any]) -> str:
    for k in ("id", "model", "name", "model_id"):
        if isinstance(e.get(k), str) and e[k]:
            return e[k]
    return "unnamed model"


def _entry_state(e: Any) -> Optional[bool]:
    """True/False when the entry says whether it is loaded, None when it has no state."""
    if not isinstance(e, dict):
        return None
    seen = False
    for k in ("loaded", "is_loaded"):
        if k in e:
            seen = True
            if bool(e[k]):
                return True
    for k in ("status", "state"):
        v = e.get(k)
        if isinstance(v, str):
            seen = True
            if v.strip().lower() in _LOADED_STATUS:
                return True
        elif isinstance(v, dict):  # e.g. {"status": {"value": "loaded"}}
            inner = v.get("value") or v.get("status") or v.get("state")
            if isinstance(inner, str):
                seen = True
                if inner.strip().lower() in _LOADED_STATUS:
                    return True
    return False if seen else None


def _entries(data: Any) -> Optional[list[Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("models", "data", "items", "loaded_models"):
            if isinstance(data.get(k), list):
                return data[k]
    return None


class OmlxService:
    """The user's chat model server (oMLX, port 8000). READ-ONLY: only GET requests.

    Strategy:
    1. GET {base}/v1/models/status (admin-style status listing). 404 or an answer
       without load-state fields falls through to step 2.
    2. GET {base}/v1/models (OpenAI-style). Entries with a truthy "loaded"/"is_loaded"
       or status "loaded" count as loaded.
    If neither answer carries load state, busy() returns None (unknown = not busy) and
    `last_error` / `last_raw` keep what was seen for diagnosis. An unreachable server or
    timeout also means not busy (nothing is loaded if oMLX is not running).
    """

    STATUS_PATH = "/v1/models/status"
    MODELS_PATH = "/v1/models"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 1.5,
        api_key: Optional[str] = None,
        name: str = "oMLX",
        fetch: Optional[Fetcher] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self.name = name
        self._fetch: Fetcher = fetch or _urllib_get
        self.last_error: Optional[str] = None
        self.last_raw: Optional[str] = None
        self.last_loaded: list[str] = []

    def _get(self, path: str) -> tuple[int, Any, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        status, body = self._fetch("GET", self.base_url + path, headers, self.timeout)
        raw = body.decode("utf-8", "replace") if isinstance(body, (bytes, bytearray)) else str(body)
        try:
            data = json.loads(raw) if raw.strip() else None
        except ValueError:
            data = None
        return status, data, raw

    @staticmethod
    def _loaded(data: Any) -> Optional[list[str]]:
        """Names of loaded models, [] when state is known and none are loaded, None when
        the answer carries no load state at all."""
        if isinstance(data, dict) and isinstance(data.get("loaded_models"), list):
            return [m if isinstance(m, str) else _entry_name(m) for m in data["loaded_models"]
                    if isinstance(m, (str, dict))]
        entries = _entries(data)
        if entries is None:
            return None
        known = False
        loaded: list[str] = []
        for e in entries:
            st = _entry_state(e)
            if st is None:
                continue
            known = True
            if st:
                loaded.append(_entry_name(e))
        return loaded if known else None

    def busy(self) -> Optional[str]:
        self.last_error = None
        self.last_loaded = []
        for path in (self.STATUS_PATH, self.MODELS_PATH):
            try:
                status, data, raw = self._get(path)
            except (OSError, urllib.error.URLError, TimeoutError, ValueError) as e:
                # Unreachable / timeout: oMLX is not running, so it holds no memory.
                self.last_error = f"{path}: unreachable ({e})"
                return None
            self.last_raw = raw[:2000]
            if status != 200:
                self.last_error = f"{path}: HTTP {status}"
                continue
            loaded = self._loaded(data)
            if loaded is None:
                self.last_error = f"{path}: no load-state fields in the answer"
                continue
            self.last_error = None
            self.last_loaded = loaded
            if not loaded:
                return None
            names = ", ".join(loaded[:3]) + (" and more" if len(loaded) > 3 else "")
            return (
                f"{self.name} has a model loaded ({names}); Shuttrr waits so it does not "
                f"compete with it for memory."
            )
        return None


PsRunner = Callable[[], str]


def _ps() -> str:
    return _run(["ps", "-axo", "pid=,rss=,command="])


def parse_ps(text: str) -> list[tuple[int, int, str]]:
    """Parse `ps -axo pid=,rss=,command=` into (pid, rss_bytes, command)."""
    out = []
    for line in text.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid, rss_kib = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        out.append((pid, rss_kib * 1024, parts[2]))
    return out


class ProcessRssService:
    """API-independent safety net: busy when any process whose command line matches
    `name_pattern` (regex, case-insensitive) has RSS above `min_bytes`.

    Example: ProcessRssService("omlx|splash|strata", 20 * GB). This process is ignored.
    Note: on Apple Silicon, Metal buffers are not always counted in RSS, so this is a
    second line of defence next to OmlxService, not a replacement.
    """

    def __init__(
        self,
        name_pattern: str,
        min_bytes: int,
        name: Optional[str] = None,
        ps: Optional[PsRunner] = None,
    ):
        self.pattern = re.compile(name_pattern, re.IGNORECASE)
        self.min_bytes = int(min_bytes)
        self.name = name or f"processes matching {name_pattern!r}"
        self._ps: PsRunner = ps or _ps
        self.last_error: Optional[str] = None

    def busy(self) -> Optional[str]:
        try:
            procs = parse_ps(self._ps())
            self.last_error = None
        except (OSError, subprocess.SubprocessError) as e:
            self.last_error = str(e)
            return None
        me = os.getpid()
        hits = [
            (rss, cmd) for pid, rss, cmd in procs
            if pid != me and rss > self.min_bytes and self.pattern.search(cmd)
        ]
        if not hits:
            return None
        rss, cmd = max(hits)
        prog = os.path.basename(cmd.split()[0]) if cmd.split() else cmd
        return (
            f"{prog} is using {fmt_gb(rss)} of memory (more than {fmt_gb(self.min_bytes)}); "
            f"Shuttrr waits so it does not compete with it."
        )


# --------------------------------------------------------------------------------------
# Guard
# --------------------------------------------------------------------------------------


@dataclass
class Decision:
    ok: bool
    reason: str
    sample: MemorySample

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reason": self.reason, "sample": self.sample.to_dict()}


ACTION_RELIEVE = "relieve"
ACTION_STOP = "stop"

PROTECT_PREFIX = "Stopped to protect your Mac: "
STOP_BODY = "it was running out of memory. Close other apps or pick a smaller size."
STOP_MESSAGE = PROTECT_PREFIX + STOP_BODY


class Watchdog:
    """Live protection while a request runs (pure logic, fed with samples; tested with FakeProbe).

    Trouble = macOS reports *critical* memory pressure, or swap grew by more than `swap_soft_bytes` since the
    run started. "warn" pressure is normal on a busy Mac and never acts.

    * First sign of trouble: ACTION_RELIEVE (the engine frees caches and idle models: a step down).
    * `grace_s` later it stops the request only if the Mac is still getting worse: pressure is still critical, or
      swap grew by another `swap_soft_bytes / 2` since the relief. Swap that grew once and then holds steady is
      not a reason (macOS does not give swap back quickly).
    * Swap grown past `swap_hard_bytes` (default 3x `swap_soft_bytes`, 768 MB with the 256 MB default) since the
      start: ACTION_STOP at once, even if it then holds steady.
    * Calm again (no critical pressure, swap no longer growing): back to normal; a later spike relieves first.
    """

    def __init__(self, swap_soft_bytes: int = 1 * GiB, swap_hard_bytes: Optional[int] = None,
                 grace_s: float = 10.0):
        self.swap_soft_bytes = int(swap_soft_bytes)
        self.swap_hard_bytes = int(swap_hard_bytes if swap_hard_bytes is not None
                                   else 3 * self.swap_soft_bytes)
        self.grace_s = float(grace_s)
        self.swap_baseline: Optional[int] = None
        self.relieved_at: Optional[float] = None
        self.relieved_swap = 0
        self.calm_swap: Optional[int] = None   # swap level that counts as "not growing" after a relief
        self.stopped: Optional[str] = None
        self.max_pressure = PRESSURE_NORMAL
        self.max_swap_growth = 0

    def start(self, sample: MemorySample) -> None:
        self.swap_baseline = int(sample.swap_used)
        self.relieved_at = None
        self.calm_swap = None
        self.stopped = None
        self.max_pressure = sample.pressure
        self.max_swap_growth = 0

    def trouble(self, s: MemorySample) -> Optional[str]:
        base = self.swap_baseline if self.swap_baseline is not None else s.swap_used
        if self.calm_swap is not None:
            base = max(base, self.calm_swap)   # growth already handled by a relief does not count again
        grown = s.swap_used - base
        if s.pressure == PRESSURE_CRITICAL:
            return "macOS reports critical memory pressure"
        if grown > self.swap_soft_bytes:
            return f"swap grew by {fmt_gb(grown)}"
        return None

    def step(self, s: MemorySample, now: Optional[float] = None) -> tuple[Optional[str], str]:
        """(None | ACTION_RELIEVE | ACTION_STOP, reason)."""
        now = time.time() if now is None else now
        if self.swap_baseline is None:
            self.start(s)
        grown = s.swap_used - int(self.swap_baseline or 0)
        self.max_swap_growth = max(self.max_swap_growth, grown)
        order = {PRESSURE_NORMAL: 0, PRESSURE_WARN: 1, PRESSURE_CRITICAL: 2}
        if order.get(s.pressure, 0) > order.get(self.max_pressure, 0):
            self.max_pressure = s.pressure
        if self.stopped:
            return ACTION_STOP, self.stopped
        if grown > self.swap_hard_bytes:
            self.stopped = f"swap grew by {fmt_gb(grown)} since the request started (limit {fmt_gb(self.swap_hard_bytes)})"
            return ACTION_STOP, self.stopped
        if self.relieved_at is not None:
            if now - self.relieved_at < self.grace_s:
                return None, "freeing caches"
            worse = []
            if s.pressure == PRESSURE_CRITICAL:
                worse.append("memory pressure is still critical")
            more = s.swap_used - self.relieved_swap
            if more > self.swap_soft_bytes // 2:
                worse.append(f"swap kept growing (another {fmt_gb(more)})")
            if worse:
                self.stopped = " and ".join(worse) + f" {self.grace_s:.0f} s after freeing caches"
                return ACTION_STOP, self.stopped
            # it eased: calm again, and the swap level now is the new floor
            self.relieved_at = None
            self.calm_swap = int(s.swap_used)
            return None, ""
        why = self.trouble(s)
        if why is None:
            return None, ""
        self.relieved_at = now
        self.relieved_swap = int(s.swap_used)
        return ACTION_RELIEVE, f"{why}; freeing caches"


def admission_budget(sample: MemorySample, headroom_bytes: int = 0) -> int:
    """What a request may use right now: the memory macOS can hand over without swapping (free + inactive +
    purgeable + speculative, see parse_vm_stat), minus an optional user-set headroom (0 by default)."""
    return max(0, int(sample.available) - max(0, int(headroom_bytes or 0)))


class MemoryGuard:
    """Admission control. check() refuses when (in this order):

    1. a protected service is busy;
    2. macOS reports *critical* memory pressure ("warn" is fine: a busy Mac sits there often);
    3. `max_fraction` is set and need > max_fraction * total;
    4. need > available (what macOS can give now: free + reclaimable). There is no fixed reserve:
       `headroom_bytes` is 0 unless the user sets `headroom_gb` themselves.

    While a request runs, a `Watchdog` (watch_step) decides when to free caches and when to stop.
    """

    def __init__(
        self,
        probe: SystemProbe,
        protected: Iterable[ProtectedService] = (),
        headroom_bytes: int = 0,
        swap_growth_bytes: int = 1 * GiB,
        max_fraction: Optional[float] = None,
        grace_s: float = 10.0,
    ):
        self.probe = probe
        self.protected = list(protected)
        self.headroom_bytes = int(headroom_bytes)
        self.swap_growth_bytes = int(swap_growth_bytes)
        self.max_fraction = max_fraction
        self.grace_s = float(grace_s)
        # Test hook only (proves the watchdog on a real Mac): admit any request whatever its estimate.
        self.admit_any = os.environ.get("SHUTTER_TEST_ADMIT_ANY") == "1"
        self.watchdog = Watchdog(self.swap_growth_bytes, grace_s=self.grace_s)
        self.swap_baseline: Optional[int] = None
        self.run_started_at: Optional[float] = None
        self.last_sample: Optional[MemorySample] = None
        self.last_decision: Optional[Decision] = None

    def _sample(self) -> MemorySample:
        s = self.probe.sample()
        self.last_sample = s
        return s

    def _protected_reason(self) -> Optional[str]:
        for svc in self.protected:
            try:
                reason = svc.busy()
            except Exception as e:  # a broken checker must not crash the engine
                reason = None
                setattr(svc, "last_error", f"busy() raised {e!r}")
            if reason:
                return reason
        return None

    def budget(self, sample: Optional[MemorySample] = None) -> int:
        return admission_budget(sample or self._sample(), self.headroom_bytes)

    def check(self, need_bytes: int, what: str = "") -> Decision:
        label = what or "this job"
        s = self._sample()

        def done(ok: bool, reason: str) -> Decision:
            d = Decision(ok, reason, s)
            self.last_decision = d
            return d

        reason = self._protected_reason()
        if reason:
            return done(False, reason)
        if s.pressure == PRESSURE_CRITICAL:
            return done(
                False,
                f"macOS reports critical memory pressure; waiting until it eases before starting {label}.",
            )
        if self.max_fraction is not None and need_bytes > self.max_fraction * s.total:
            cap = self.max_fraction * s.total
            return done(
                False,
                f"{label[:1].upper() + label[1:]} needs {fmt_gb(need_bytes)}, more than the "
                f"{self.max_fraction:.0%} limit of this Mac's {fmt_gb(s.total)} "
                f"({fmt_gb(cap)}).",
            )
        budget = admission_budget(s, self.headroom_bytes)
        extra = f" (after the {fmt_gb(self.headroom_bytes)} headroom you set)" if self.headroom_bytes else ""
        if need_bytes > budget and not self.admit_any:
            n, b = fmt_gb(need_bytes), fmt_gb(budget)
            if n == b:
                n, b = f"{need_bytes / GB:.2f} GB", f"{budget / GB:.2f} GB"
            return done(
                False,
                f"Not enough free memory for {label}: it needs {n}, but macOS can give only {b} right now{extra}.",
            )
        return done(
            True,
            f"{fmt_gb(budget)} free right now{extra} covers {fmt_gb(need_bytes)} for {label}.",
        )

    def admit(self, need_bytes: int, what: str = "") -> Decision:
        d = self.check(need_bytes, what)
        if not d.ok:
            raise MemoryRefused(d.reason)
        return d

    def begin_run(self) -> MemorySample:
        s = self._sample()
        self.swap_baseline = s.swap_used
        self.run_started_at = time.time()
        self.watchdog = Watchdog(self.swap_growth_bytes, grace_s=self.grace_s)
        self.watchdog.start(s)
        return s

    def end_run(self) -> None:
        self.swap_baseline = None
        self.run_started_at = None

    def watch_step(self, now: Optional[float] = None) -> tuple[Optional[str], str]:
        """One watchdog reading: (None | "relieve" | "stop", reason)."""
        s = self._sample()
        if self.swap_baseline is None:  # not inside begin_run/end_run: judge pressure only
            if s.pressure == PRESSURE_CRITICAL:
                return ACTION_STOP, "macOS reports critical memory pressure"
            return None, ""
        return self.watchdog.step(s, now)

    def watch(self) -> Optional[str]:
        """The stop reason (as a plain sentence) when the watchdog says stop, else None."""
        action, why = self.watch_step()
        if action == ACTION_STOP:
            return f"{STOP_BODY} ({why})"
        return None

    def snapshot(self) -> dict[str, Any]:
        s = self._sample()
        services = []
        for svc in self.protected:
            try:
                reason = svc.busy()
            except Exception as e:
                reason = None
                setattr(svc, "last_error", f"busy() raised {e!r}")
            services.append({
                "name": getattr(svc, "name", type(svc).__name__),
                "busy": reason,
                "last_error": getattr(svc, "last_error", None),
            })
        return {
            "sample": s.to_dict(),
            "headroom_bytes": self.headroom_bytes,
            "swap_growth_bytes": self.swap_growth_bytes,
            "max_fraction": self.max_fraction,
            "swap_baseline": self.swap_baseline,
            "run_started_at": self.run_started_at,
            "protected": services,
            "last_decision": self.last_decision.to_dict() if self.last_decision else None,
        }


# --------------------------------------------------------------------------------------
# GPU lock
# --------------------------------------------------------------------------------------


class GpuLock:
    """Exclusive cross-process lock (fcntl.flock on a lock file).

    The lock path is configurable so it can be pointed at a lock file other tools on the
    machine already use (rule 1: one heavy GPU job at a time across tools). The holder
    writes {"pid", "label", "time"} JSON into the file; the file is truncated on release
    but never deleted (deleting lock files races with other waiters).

    `with GpuLock(path):` blocks until acquired, or raises TimeoutError after
    `timeout_s` seconds when given.
    """

    def __init__(self, path: str, label: str = "shutter", timeout_s: Optional[float] = None,
                 poll_s: float = 1.0):
        if fcntl is None:  # pragma: no cover
            raise RuntimeError("GpuLock needs fcntl (macOS or Linux)")
        self.path = os.path.expanduser(path)
        self.label = label
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self._fd: Optional[int] = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def _try_once(self) -> bool:
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(fd)
            if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                return False
            raise
        info = json.dumps({"pid": os.getpid(), "label": self.label, "time": time.time()})
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, info.encode())
        os.fsync(fd)
        self._fd = fd
        return True

    def try_acquire(self, timeout_s: Optional[float] = 0, poll_s: Optional[float] = None) -> bool:
        """Try to take the lock. timeout_s=0: one attempt; None: wait forever."""
        if self._fd is not None:
            return True
        poll = self.poll_s if poll_s is None else poll_s
        deadline = None if timeout_s is None else time.monotonic() + max(0.0, timeout_s)
        while True:
            if self._try_once():
                return True
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                time.sleep(min(poll, remaining))
            else:
                time.sleep(poll)

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            os.ftruncate(fd, 0)
        except OSError:
            pass
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def holder(self) -> Optional[dict[str, Any]]:
        """Who holds the lock: the JSON the holder wrote, or None when nobody holds it.

        A held lock whose file is not our JSON (junk, or another tool's format) gives
        {"pid": None, "label": None, "time": None, "raw": <first 200 chars>}.
        """
        if self._fd is not None:
            locked = True
        else:
            try:
                fd = os.open(self.path, os.O_RDONLY)
            except FileNotFoundError:
                return None
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    locked = False
                except OSError:
                    locked = True
            finally:
                os.close(fd)
        if not locked:
            return None
        try:
            with open(self.path, "rb") as f:
                raw = f.read(4096).decode("utf-8", "replace")
        except OSError:
            raw = ""
        try:
            data = json.loads(raw)
        except ValueError:
            data = None
        if isinstance(data, dict):
            return {"pid": data.get("pid"), "label": data.get("label"), "time": data.get("time")}
        return {"pid": None, "label": None, "time": None, "raw": raw[:200]}

    def __enter__(self) -> "GpuLock":
        if not self.try_acquire(timeout_s=self.timeout_s):
            h = self.holder() or {}
            who = h.get("label") or "another process"
            pid = f" (pid {h['pid']})" if h.get("pid") else ""
            raise TimeoutError(
                f"The GPU lock {self.path} is held by {who}{pid}; gave up after {self.timeout_s}s."
            )
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.release()
        except Exception:
            pass


__all__ = [
    "MemorySample", "SystemProbe", "MacProbe", "LinuxProbe", "FakeProbe", "default_probe",
    "parse_vm_stat", "parse_swapusage", "parse_pressure_level", "parse_meminfo", "parse_ps",
    "ProtectedService", "OmlxService", "ProcessRssService",
    "Decision", "MemoryGuard", "Watchdog", "admission_budget", "ACTION_RELIEVE", "ACTION_STOP", "STOP_MESSAGE", "STOP_BODY",
    "PROTECT_PREFIX",
    "GpuLock", "fmt_gb",
]
