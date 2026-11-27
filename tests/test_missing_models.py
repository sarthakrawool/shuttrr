"""Models whose folders were deleted disappear from GET /v1/models and refuse requests; they come back by themselves
when the folder is back (re-downloaded, drive plugged in). The config entry (and the person's saved defaults) stays."""

from __future__ import annotations

import shutil

import pytest

from shutter.adapters.dummy import DummyAdapter
from shutter.core.config import Config
from shutter.core.engine import Engine, SubmitError
from shutter.core.registry import Registry
from shutter.types import JobRequest

from tests.test_engine import FakeProbe


class FolderAdapter(DummyAdapter):
    family = "folder"
    needs_model_dir = True


def make(tmp_path):
    d = tmp_path / "models" / "My-Model"
    d.mkdir(parents=True)
    cfg = Config(home=str(tmp_path / "home"), memory_poll_s=0.01, busy_process_pattern=None, dummy_model=False,
                 models={"mine": {"adapter": "folder", "dir": str(d), "options": {"user_keys": ["steps"], "steps": 9}}})
    reg = Registry(cfg, families={"folder": FolderAdapter})
    return Engine(cfg, probe=FakeProbe(), registry=reg, watch_interval_s=0.0, sample_interval_s=0.01), d, cfg


def test_deleted_model_is_hidden_and_refused_then_comes_back(tmp_path):
    eng, d, cfg = make(tmp_path)
    assert [m["id"] for m in eng.models()] == ["mine"]
    shutil.rmtree(d)
    assert eng.models() == []                                   # re-checked on every call, no restart needed
    with pytest.raises(SubmitError, match="no longer on this Mac"):
        eng.submit(JobRequest(model="mine", params={"steps": 1, "step_s": 0}))
    assert cfg.models["mine"]["options"]["steps"] == 9          # saved defaults kept for a re-download
    d.mkdir(parents=True)
    assert [m["id"] for m in eng.models()] == ["mine"]


def test_registry_flags_missing_folders(tmp_path):
    eng, d, _ = make(tmp_path)
    assert not eng.registry.files_missing("mine")
    shutil.rmtree(d)
    rows = {r["id"]: r for r in eng.registry.list()}
    assert rows["mine"]["missing"] and not rows["mine"]["installed"] and eng.registry.files_missing("mine")
