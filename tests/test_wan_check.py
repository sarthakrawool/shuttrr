"""scripts/wan_check.py on the CPU: the step loop in fresh processes, records, checked_on, and the GPU lock refusals."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tests.test_wan_pipeline import make_build, vae_file  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent


def load_script():
    spec = importlib.util.spec_from_file_location("wan_check", ROOT / "scripts" / "wan_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def wc():
    return load_script()


def embeds(path: Path) -> Path:
    r = np.random.default_rng(5)
    np.savez(path, prompt=r.standard_normal((7, 4096)).astype(np.float32),
             negative=r.standard_normal((5, 4096)).astype(np.float32))
    return path


def test_checked_on_entry_and_append(tmp_path, wc):
    build = make_build(tmp_path, None)
    results = [{"rung": "resident", "ok": True, "peak_bytes": 3 * wc.GB, "seconds": 10.0, "hash": "a"},
               {"rung": "stream_tight", "ok": True, "peak_bytes": wc.GB, "seconds": 20.0, "hash": "a"},
               {"rung": "staged", "ok": False, "error": "exit code 1"}]
    cmp = wc.compare(results)
    assert cmp["reference"] == "resident" and cmp["same"] == {"resident": True, "stream_tight": True}
    assert not cmp["identical"]                                    # a step failed
    e = wc.checked_on_entry("16x16", 5, 2, results, cmp)
    assert e["peaks_gb"] == {"resident": 3.0, "stream_tight": 1.0} and e["failed"] == ["staged"]
    assert e["machine"]["arch"] and e["machine"]["memory_gb"] > 0 and e["mlx"] != "?"
    wc.append_checked_on(build, e)
    wc.append_checked_on(build, dict(e, size="32x32"))
    man = json.loads((build / "shuttrr.json").read_text())
    assert [x["size"] for x in man["checked_on"]] == ["16x16", "32x32"]
    assert man["parts"]["dit"]["n_blocks"] == 3                    # the rest of the manifest untouched
    assert wc.compare([{"rung": "a", "ok": True, "hash": "x"}, {"rung": "b", "ok": True, "hash": "y"}])["same"] \
        == {"a": True, "b": False}


def test_two_steps_in_fresh_processes(tmp_path, wc, vae_file):  # noqa: F811
    """main(): takes the locks, runs two steps each in its own process, finds them identical, records them."""
    build = make_build(tmp_path, vae_file)
    measured = tmp_path / "measured.json"
    lock_dir = tmp_path / "studio-gpu.lock.d"
    out = tmp_path / "out"
    rc = wc.main([str(build), "--model-id", "tiny", "--sizes", "16x16", "--frames", "5", "--steps", "2",
                  "--rungs", "resident,stream_tight", "--out", str(out), "--measured", str(measured),
                  "--lock-file", str(tmp_path / "gpu.lock"), "--lock-dirs", str(lock_dir), "--cpu",
                  "--prompt-embeds", str(embeds(tmp_path / "e.npz"))])
    report = json.loads((out / "report.json").read_text())
    assert rc == 0, json.dumps(report, indent=1)[:3000]
    assert not lock_dir.exists()                                  # released
    res = report["sizes"]["16x16"]["results"]
    assert [r["rung"] for r in res] == ["resident", "stream_tight"] and all(r["ok"] for r in res)
    assert res[0]["hash"] == res[1]["hash"] and res[0]["peak_bytes"] > 0
    a, b = (np.load(out / "16x16" / r / "frames.npy") for r in ("resident", "stream_tight"))
    assert a.shape == (5, 16, 16, 3) and np.array_equal(a, b)
    m = json.loads(measured.read_text())["tiny"]["16x16"]
    assert set(m) == {"resident", "stream_tight"} and m["stream_tight"]["identical"] is True
    assert m["resident"]["frames"] == 5 and m["resident"]["steps"] == 2
    co = json.loads((build / "shuttrr.json").read_text())["checked_on"]
    assert len(co) == 1 and co[0]["identical"] is True and co[0]["size"] == "16x16"
    assert set(co[0]["peaks_gb"]) == {"resident", "stream_tight"} and co[0]["frames"] == 5 and co[0]["steps"] == 2
    # the adapter now uses the measured peak
    from shutter.adapters.wan import WanAdapter
    from shutter.types import JobRequest
    ad = WanAdapter("tiny", build, {"measured_path": str(measured)})
    est = ad.rung_estimates(JobRequest(model="tiny", prompt="x", width=16, height=16, frames=5))
    assert est["stream_tight"].peak == m["stream_tight"]["peak_bytes"] and "measured" in est["stream_tight"].note


def test_refuses_when_gpu_is_held(tmp_path, wc, capsys):
    from shutter.core.memory import GpuLock
    build = make_build(tmp_path, None)
    base = [str(build), "--model-id", "tiny", "--rungs", "resident", "--out", str(tmp_path / "o"), "--cpu"]
    held = GpuLock(str(tmp_path / "gpu.lock"), label="another render")
    assert held.try_acquire(0)
    try:
        rc = wc.main(base + ["--lock-file", str(tmp_path / "gpu.lock"), "--lock-dirs", ""])
    finally:
        held.release()
    assert rc == 3 and "another render" in capsys.readouterr().err
    # a shared lock folder held by another tool: refused, and the folder is left alone
    d = tmp_path / "studio-gpu.lock.d"
    d.mkdir()
    (d / "owner").write_text("fasth3\n")
    rc = wc.main(base + ["--lock-file", str(tmp_path / "gpu.lock"), "--lock-dirs", str(d)])
    assert rc == 3 and "fasth3" in capsys.readouterr().err and (d / "owner").exists()
    assert not (tmp_path / "o" / "report.json").exists()


def test_gpulock_reexecs_under_the_script(tmp_path, wc, monkeypatch):
    build = make_build(tmp_path, None)
    rec = tmp_path / "args.txt"
    script = tmp_path / "gpulock.sh"
    script.write_text(f'#!/bin/sh\necho "$@" > {rec}\necho "$SHUTTER_WAN_CHECK_LOCKED" >> {rec}\nexit 0\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.delenv(wc.LOCKED_ENV, raising=False)
    rc = wc.main([str(build), "--model-id", "tiny", "--gpulock", str(script), "--rungs", "staged"])
    assert rc == 0
    args, env = rec.read_text().splitlines()
    assert args.startswith("shuttr-wan-check -- ") and "wan_check.py" in args and "--rungs staged" in args
    assert env == "1"
    assert os.environ.get(wc.LOCKED_ENV) is None


def test_bad_arguments(tmp_path, wc):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError, match="HEIGHTxWIDTH"):
        wc.parse_size("480by832")
    assert wc.parse_size("480x832") == (480, 832)
    assert wc.main([str(tmp_path / "nope"), "--model-id", "x"]) == 2
    build = make_build(tmp_path, None)
    assert wc.main([str(build), "--model-id", "x", "--sizes", "480", "--no-lock"]) == 2
    assert wc.main([str(build), "--model-id", "x", "--rungs", "resident,fastest", "--no-lock"]) == 2


def test_memory_level_floor(wc, monkeypatch):
    monkeypatch.setattr(wc.sys, "platform", "darwin")
    monkeypatch.setattr(wc, "_sysctl", lambda name: "59")
    assert "59 (below 78)" in wc.memory_level_problem()
    assert wc.memory_level_problem(40) is None and wc.memory_level_problem(0) is None
    assert wc.build_parser().parse_args(["b", "--model-id", "x", "--min-memory-level", "40"]).min_memory_level == 40


def test_encode_once_runs_first_and_feeds_every_step(tmp_path, wc):
    build = make_build(tmp_path, None)
    seen = []

    def runner(spec, timeout):
        seen.append(spec)
        if spec.get("encode_only"):
            return {"rung": "encode", "ok": True, "peak_bytes": wc.GB, "seconds": 1.0, "file": spec["out_file"]}
        return {"rung": spec["rung"], "ok": True, "peak_bytes": wc.GB, "seconds": 2.0, "hash": "h", "frames": 5,
                "steps": 2, "size": [16, 16]}
    a = wc.build_parser().parse_args([str(build), "--model-id", "tiny", "--sizes", "16x16", "--rungs",
                                      "resident,stream_tight", "--encode-once", "--no-record",
                                      "--out", str(tmp_path / "o")])
    rep = wc.check(a, runner=runner, log=lambda s: None)
    assert seen[0]["encode_only"] and [s["rung"] for s in seen[1:]] == ["resident", "stream_tight"]
    assert all(s["prompt_embeds"] == str(tmp_path / "o" / "prompt_embeds.npz") for s in seen[1:])
    assert rep["prompt_encode"]["seconds"] == 1.0 and rep["sizes"]["16x16"]["compare"]["identical"]
