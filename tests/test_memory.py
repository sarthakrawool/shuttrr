import json
import multiprocessing as mp
import os
import subprocess
import sys
import textwrap
import time
import urllib.error

import pytest

from shutter.core import memory as m
from shutter.core.memory import (
    Decision,
    FakeProbe,
    GpuLock,
    LinuxProbe,
    MacProbe,
    MemoryGuard,
    MemorySample,
    OmlxService,
    ProcessRssService,
    ProtectedService,
    SystemProbe,
    default_probe,
    parse_meminfo,
    parse_pressure_level,
    parse_swapusage,
    parse_vm_stat,
)
from shutter.types import GB, MemoryRefused

MiB = 1024**2

VM_STAT = textwrap.dedent(
    """\
    Mach Virtual Memory Statistics: (page size of 16384 bytes)
    Pages free:                               81234.
    Pages active:                           2345678.
    Pages inactive:                         1234567.
    Pages speculative:                        12345.
    Pages throttled:                              0.
    Pages wired down:                        456789.
    Pages purgeable:                          23456.
    "Translation faults":                1234567890.
    Pages copy-on-write:                   12345678.
    Pages zero filled:                    987654321.
    Pages reactivated:                      1234567.
    Pages purged:                            123456.
    File-backed pages:                      1111111.
    Anonymous pages:                        2222222.
    Pages stored in compressor:              333333.
    Pages occupied by compressor:            111111.
    Decompressions:                          444444.
    Compressions:                            555555.
    Pageins:                                6666666.
    Pageouts:                                  7777.
    Swapins:                                      0.
    Swapouts:                                     0.
    """
)

SWAP = "vm.swapusage: total = 2048.00M  used = 1024.50M  free = 1023.50M  (encrypted)"

MEMINFO = textwrap.dedent(
    """\
    MemTotal:       16314424 kB
    MemFree:         1234567 kB
    MemAvailable:    9876543 kB
    Buffers:          123456 kB
    Cached:          5432100 kB
    Inactive:        3333333 kB
    SwapTotal:       2097148 kB
    SwapFree:        1048574 kB
    HugePages_Total:       0
    """
)


# ---------------------------------------------------------------- parsers

def test_parse_vm_stat_bytes_and_available():
    vm = parse_vm_stat(VM_STAT, 4096)  # header page size wins over the argument
    ps = 16384
    assert vm["page_size"] == ps
    assert vm["free"] == 81234 * ps
    assert vm["inactive"] == 1234567 * ps
    assert vm["speculative"] == 12345 * ps
    assert vm["purgeable"] == 23456 * ps
    assert vm["wired"] == 456789 * ps
    assert vm["compressed"] == 111111 * ps
    assert vm["available"] == (81234 + 1234567 + 23456 + 12345) * ps


def test_parse_vm_stat_without_header_uses_page_size():
    vm = parse_vm_stat("Pages free: 10.\nPages inactive: 5.\n", 4096)
    assert vm["available"] == 15 * 4096
    assert "purgeable" not in vm


def test_parse_vm_stat_junk():
    assert parse_vm_stat("nonsense", 16384)["available"] == 0


def test_parse_swapusage():
    s = parse_swapusage(SWAP)
    assert s["total"] == 2048 * MiB
    assert s["used"] == int(1024.5 * MiB)
    assert s["free"] == int(1023.5 * MiB)
    # -n form, no prefix, and gigabyte units
    s2 = parse_swapusage("total = 0.00M  used = 0.00M  free = 0.00M  (encrypted)")
    assert s2 == {"total": 0, "used": 0, "free": 0}
    assert parse_swapusage("total = 4.00G  used = 1.50G  free = 2.50G")["used"] == int(1.5 * GB)
    assert parse_swapusage("") == {"total": 0, "used": 0, "free": 0}


@pytest.mark.parametrize("text,expected", [
    ("1\n", "normal"), ("2", "warn"), ("4\n", "critical"),
    ("kern.memorystatus_vm_pressure_level: 2", "warn"), ("", "normal"), ("abc", "normal"),
])
def test_parse_pressure(text, expected):
    assert parse_pressure_level(text) == expected


def test_parse_meminfo():
    mi = parse_meminfo(MEMINFO)
    assert mi["MemTotal"] == 16314424 * 1024
    assert mi["HugePages_Total"] == 0


# ---------------------------------------------------------------- probes

def fake_mac_runner(pressure="1\n", swap=SWAP, fail_swap=False):
    calls = []

    def run(cmd):
        calls.append(list(cmd))
        if cmd[:2] == ["sysctl", "-n"] and cmd[2] == "hw.memsize":
            return "137438953472\n16384\n"
        if cmd == ["vm_stat"]:
            return VM_STAT
        if cmd[-1] == "vm.swapusage":
            if fail_swap:
                raise subprocess.CalledProcessError(1, cmd)
            return swap
        if cmd[-1] == "kern.memorystatus_vm_pressure_level":
            return pressure
        raise AssertionError(cmd)

    run.calls = calls
    return run


def test_mac_probe():
    run = fake_mac_runner(pressure="2\n")
    p = MacProbe(runner=run)
    s = p.sample()
    assert s.total == 128 * GB
    assert s.available == (81234 + 1234567 + 23456 + 12345) * 16384
    assert s.swap_used == int(1024.5 * MiB)
    assert s.swap_total == 2048 * MiB
    assert s.pressure == "warn"
    assert s.wired == 456789 * 16384 and s.compressed == 111111 * 16384
    p.sample()
    # memsize/pagesize read once
    assert sum(1 for c in run.calls if "hw.memsize" in c) == 1
    assert isinstance(p, SystemProbe)


def test_mac_probe_swap_failure_is_tolerated():
    s = MacProbe(runner=fake_mac_runner(fail_swap=True)).sample()
    assert s.swap_used == 0 and s.pressure == "normal"


def test_linux_probe(tmp_path):
    f = tmp_path / "meminfo"
    f.write_text(MEMINFO)
    s = LinuxProbe(str(f)).sample()
    assert s.total == 16314424 * 1024
    assert s.available == 9876543 * 1024
    assert s.swap_used == (2097148 - 1048574) * 1024
    assert s.pressure == "normal"


@pytest.mark.skipif(not os.path.exists("/proc/meminfo"), reason="linux only")
def test_default_probe_real():
    s = default_probe().sample()
    assert s.total > 0 and 0 < s.available <= s.total


def test_sample_to_dict():
    s = MemorySample(total=64 * GB, available=40 * GB, swap_used=0, pressure="normal", ts=1.0)
    d = s.to_dict()
    assert d["total"] == 64 * GB and d["used"] == 24 * GB and d["available_gb"] == 40.0
    assert d["free"] is None and d["ts"] == 1.0
    json.dumps(d)


def test_fake_probe_sequence():
    p = FakeProbe(available=40 * GB, sequence=[{"swap_used": 1 * GB}, {"pressure": "warn"}])
    assert p.sample().swap_used == 1 * GB
    s = p.sample()
    assert s.pressure == "warn" and s.swap_used == 1 * GB  # overrides persist
    p.set(pressure="normal")
    assert p.sample().pressure == "normal"
    explicit = MemorySample(total=1, available=1, swap_used=0)
    p.push(explicit)
    assert p.sample() is explicit
    assert p.calls == 4


# ---------------------------------------------------------------- guard

class StubService:
    def __init__(self, reason=None, name="stub"):
        self.reason, self.name = reason, name

    def busy(self):
        return self.reason


class RaisingService:
    name = "broken"

    def busy(self):
        raise RuntimeError("boom")


def test_guard_ok():
    g = MemoryGuard(FakeProbe(available=48 * GB))
    d = g.check(30 * GB, "LongCat DiT")
    assert isinstance(d, Decision) and d.ok
    assert d.sample.available == 48 * GB
    assert g.admit(30 * GB).ok


def test_guard_refuses_not_enough_memory():
    g = MemoryGuard(FakeProbe(available=30.25 * GB), headroom_bytes=8 * GB)
    d = g.check(25 * GB, "the LongCat DiT")
    assert not d.ok
    assert d.reason == (
        "Not enough free memory for the LongCat DiT: it needs 25.0 GB plus 8.0 GB "
        f"headroom, but only {m.fmt_gb(30.25 * GB)} is available."
    )
    with pytest.raises(MemoryRefused) as ei:
        g.admit(25 * GB, "the LongCat DiT")
    assert ei.value.reason == d.reason
    # exactly fits
    assert g.check(int(22.25 * GB)).ok


def test_guard_refuses_pressure():
    g = MemoryGuard(FakeProbe(available=100 * GB, pressure="warn"))
    d = g.check(1 * GB, "the VAE")
    assert not d.ok
    assert "warn memory pressure" in d.reason and "the VAE" in d.reason


def test_guard_refuses_protected_service_first():
    svc = StubService("oMLX has a model loaded (qwen); Shutter waits.")
    g = MemoryGuard(FakeProbe(available=100 * GB, pressure="critical"), protected=[svc])
    d = g.check(1 * GB)
    assert not d.ok and d.reason == svc.reason
    assert isinstance(svc, ProtectedService)


def test_guard_broken_service_is_ignored():
    g = MemoryGuard(FakeProbe(), protected=[RaisingService()])
    assert g.check(1 * GB).ok


def test_guard_max_fraction():
    g = MemoryGuard(FakeProbe(total=128 * GB, available=120 * GB), headroom_bytes=0,
                    max_fraction=0.75)
    d = g.check(100 * GB, "this model")
    assert not d.ok
    assert "100.0 GB" in d.reason and "75%" in d.reason and "96.0 GB" in d.reason
    assert d.reason.startswith("This model")
    assert g.check(90 * GB).ok


def test_guard_default_what():
    d = MemoryGuard(FakeProbe(available=1 * GB)).check(1 * GB)
    assert "this job" in d.reason


def test_watch_swap_growth():
    p = FakeProbe(swap_used=1 * GB)
    g = MemoryGuard(p, swap_growth_bytes=256 * MiB)
    assert g.watch() is None  # no run yet
    g.begin_run()
    assert g.swap_baseline == 1 * GB
    p.set(swap_used=1 * GB + 200 * MiB)
    assert g.watch() is None
    p.set(swap_used=1 * GB + 600 * MiB)
    r = g.watch()
    assert r and "Swap grew by 0.6 GB" in r and f"limit {m.fmt_gb(256 * MiB)}" in r
    g.end_run()
    assert g.watch() is None


def test_watch_swap_shrinking_is_fine():
    p = FakeProbe(swap_used=2 * GB)
    g = MemoryGuard(p)
    g.begin_run()
    p.set(swap_used=0)
    assert g.watch() is None


def test_watch_critical_pressure():
    p = FakeProbe()
    g = MemoryGuard(p)
    g.begin_run()
    p.set(pressure="warn")
    assert g.watch() is None
    p.set(pressure="critical")
    assert "critical" in g.watch()


def test_snapshot():
    g = MemoryGuard(FakeProbe(), protected=[StubService(None, "omlx")])
    g.check(1 * GB, "x")
    snap = g.snapshot()
    json.dumps(snap)
    assert snap["protected"] == [{"name": "omlx", "busy": None, "last_error": None}]
    assert snap["last_decision"]["ok"] is True
    assert snap["sample"]["total"] == 64 * GB


# ---------------------------------------------------------------- OmlxService

class FakeFetch:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, method, url, headers, timeout):
        self.calls.append((method, url, dict(headers), timeout))
        r = self.routes.get(url.split("8000", 1)[1])
        if r is None:
            return 404, b'{"detail":"Not Found"}'
        if isinstance(r, Exception):
            raise r
        status, body = r
        return status, body if isinstance(body, bytes) else json.dumps(body).encode()


def _assert_get_only(fetch):
    assert fetch.calls and all(c[0] == "GET" for c in fetch.calls)


def test_omlx_status_loaded():
    fetch = FakeFetch({"/v1/models/status": (200, {"models": [
        {"id": "qwen3-32b", "loaded": True, "memory": 20 * GB},
        {"id": "other", "loaded": False},
    ]})})
    svc = OmlxService(fetch=fetch, api_key="k")
    r = svc.busy()
    assert r and "qwen3-32b" in r and "other" not in r
    assert len(fetch.calls) == 1
    assert fetch.calls[0][2]["Authorization"] == "Bearer k"
    assert fetch.calls[0][3] == 1.5
    _assert_get_only(fetch)


def test_omlx_status_loaded_models_list():
    fetch = FakeFetch({"/v1/models/status": (200, {"loaded_models": ["a", {"id": "b"}]})})
    r = OmlxService(fetch=fetch).busy()
    assert "a, b" in r


def test_omlx_status_none_loaded():
    fetch = FakeFetch({"/v1/models/status": (200, {"models": [{"id": "x", "status": "unloaded"}]})})
    svc = OmlxService(fetch=fetch)
    assert svc.busy() is None
    assert svc.last_error is None and svc.last_loaded == []
    assert len(fetch.calls) == 1


def test_omlx_fallback_to_models_loaded():
    fetch = FakeFetch({"/v1/models": (200, {"object": "list", "data": [
        {"id": "m1", "object": "model", "status": "loaded"},
        {"id": "m2", "object": "model", "is_loaded": False},
    ]})})
    svc = OmlxService("http://127.0.0.1:8000/", fetch=fetch)
    r = svc.busy()
    assert r and "m1" in r
    assert [c[1] for c in fetch.calls] == [
        "http://127.0.0.1:8000/v1/models/status", "http://127.0.0.1:8000/v1/models"]
    _assert_get_only(fetch)


def test_omlx_fallback_not_loaded():
    fetch = FakeFetch({"/v1/models": (200, {"data": [{"id": "m1", "loaded": False}]})})
    assert OmlxService(fetch=fetch).busy() is None


def test_omlx_no_state_fields_is_unknown_not_busy():
    body = {"object": "list", "data": [{"id": "m1", "object": "model", "owned_by": "omlx"}]}
    fetch = FakeFetch({"/v1/models": (200, body)})
    svc = OmlxService(fetch=fetch)
    assert svc.busy() is None
    assert "no load-state" in svc.last_error
    assert "owned_by" in svc.last_raw


def test_omlx_status_without_state_falls_back():
    fetch = FakeFetch({
        "/v1/models/status": (200, {"ok": True}),
        "/v1/models": (200, {"data": [{"id": "m", "loaded": True}]}),
    })
    assert "m" in OmlxService(fetch=fetch).busy()


def test_omlx_unreachable():
    fetch = FakeFetch({"/v1/models/status": urllib.error.URLError("refused")})
    svc = OmlxService(fetch=fetch)
    assert svc.busy() is None
    assert "unreachable" in svc.last_error
    assert len(fetch.calls) == 1


def test_omlx_timeout():
    fetch = FakeFetch({"/v1/models/status": TimeoutError("timed out")})
    assert OmlxService(fetch=fetch).busy() is None


def test_omlx_bad_json():
    fetch = FakeFetch({"/v1/models/status": (200, b"<html>"), "/v1/models": (500, b"")})
    svc = OmlxService(fetch=fetch)
    assert svc.busy() is None and "HTTP 500" in svc.last_error


def test_default_fetcher_refuses_non_get():
    with pytest.raises(ValueError):
        m._urllib_get("POST", "http://127.0.0.1:1/v1/models/unload", {}, 0.1)


def test_omlx_real_unreachable_port():
    # Real urllib path against a closed port: must be "not busy", fast.
    svc = OmlxService("http://127.0.0.1:9", timeout=0.5)
    assert svc.busy() is None
    assert svc.last_error


# ---------------------------------------------------------------- ProcessRssService

PS_OUT = textwrap.dedent(
    """\
        1   12000 /sbin/launchd
      501 31457280 /Applications/oMLX.app/Contents/MacOS/omlx serve --port 8000
      777  2097152 /usr/local/bin/strata-helper
      888 99999999 /usr/bin/python3 -m shutter.server
    garbage line
    """
)


def test_process_rss_busy():
    svc = ProcessRssService("omlx|splash|strata", 20 * GB, ps=lambda: PS_OUT)
    r = svc.busy()
    assert r and "omlx" in r and "30.0 GB" in r and "20.0 GB" in r


def test_process_rss_below_threshold():
    svc = ProcessRssService("omlx|splash|strata", 40 * GB, ps=lambda: PS_OUT)
    assert svc.busy() is None


def test_process_rss_ignores_self():
    me = os.getpid()
    out = f"{me} 99999999 python omlx-test\n"
    assert ProcessRssService("omlx", 1 * GB, ps=lambda: out).busy() is None


def test_process_rss_ps_failure():
    def boom():
        raise OSError("no ps")
    svc = ProcessRssService("omlx", 1, ps=boom)
    assert svc.busy() is None and svc.last_error


def test_parse_ps():
    procs = m.parse_ps(PS_OUT)
    assert procs[1] == (501, 31457280 * 1024,
                        "/Applications/oMLX.app/Contents/MacOS/omlx serve --port 8000")
    assert len(procs) == 4


# ---------------------------------------------------------------- GpuLock

def _hold_lock(path, ready, release):
    lock = GpuLock(path, label="child")
    assert lock.try_acquire()
    ready.set()
    release.wait(10)
    lock.release()


def test_gpulock_basic(tmp_path):
    path = str(tmp_path / "deep" / "dir" / "gpu.lock")
    a = GpuLock(path, label="a")
    assert a.holder() is None  # file does not exist yet
    assert a.try_acquire()
    assert a.held and a.try_acquire()  # idempotent
    h = a.holder()
    assert h["pid"] == os.getpid() and h["label"] == "a"
    b = GpuLock(path, label="b")
    assert b.holder()["label"] == "a"
    assert not b.try_acquire()
    t0 = time.monotonic()
    assert not b.try_acquire(timeout_s=0.3, poll_s=0.05)
    assert time.monotonic() - t0 >= 0.25
    a.release()
    assert a.holder() is None
    assert b.try_acquire()
    b.release()
    b.release()  # double release is harmless


def test_gpulock_across_processes(tmp_path):
    path = str(tmp_path / "gpu.lock")
    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    p = ctx.Process(target=_hold_lock, args=(path, ready, release))
    p.start()
    try:
        assert ready.wait(20)
        lock = GpuLock(path, label="parent")
        assert not lock.try_acquire()
        h = lock.holder()
        assert h["label"] == "child" and h["pid"] == p.pid
        with pytest.raises(TimeoutError) as ei:
            with GpuLock(path, timeout_s=0.2, poll_s=0.05):
                pass
        assert "child" in str(ei.value)
        release.set()
        assert lock.try_acquire(timeout_s=10, poll_s=0.05)
        lock.release()
    finally:
        release.set()
        p.join(10)
    assert p.exitcode == 0


def test_gpulock_released_when_holder_dies(tmp_path):
    path = str(tmp_path / "gpu.lock")
    code = (
        "import sys, time; sys.path.insert(0, %r)\n"
        "from shutter.core.memory import GpuLock\n"
        "l = GpuLock(%r, label='dier'); assert l.try_acquire(); print('ok', flush=True)\n"
        "time.sleep(30)\n"
    ) % (os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(m.__file__)))), path)
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "ok"
        lock = GpuLock(path)
        assert not lock.try_acquire()
        proc.kill()
        proc.wait(10)
        assert lock.try_acquire(timeout_s=5, poll_s=0.05)
        lock.release()
    finally:
        if proc.poll() is None:
            proc.kill()


def test_gpulock_context_manager(tmp_path):
    path = str(tmp_path / "gpu.lock")
    with GpuLock(path, label="ctx") as lock:
        assert lock.held
        assert GpuLock(path).holder()["label"] == "ctx"
    assert not lock.held
    assert GpuLock(path).holder() is None


def test_gpulock_junk_file(tmp_path):
    path = tmp_path / "gpu.lock"
    path.write_bytes(b"\x00\xffnot json{{{")
    lock = GpuLock(str(path))
    assert lock.holder() is None  # junk but unlocked
    assert lock.try_acquire()
    assert json.loads(path.read_text())["label"] == "shutter"
    lock.release()


def test_gpulock_holder_with_foreign_format(tmp_path):
    # Another tool holds the lock and wrote its own (non-JSON) content.
    import fcntl
    path = tmp_path / "gpu.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    try:
        os.write(fd, b"12345 strata\n")
        fcntl.flock(fd, fcntl.LOCK_EX)
        h = GpuLock(str(path)).holder()
        assert h["pid"] is None and "strata" in h["raw"]
        assert not GpuLock(str(path)).try_acquire()
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
