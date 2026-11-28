"""Models already on this Mac (unregistered, or kept by another Shuttrr home) are offered as 'Use Files on This
Mac' and registered without a download."""
import json
import os
from pathlib import Path

import pytest

from shutter.core import catalog as cat
from shutter.core.config import Config

MID = "z-image-turbo-8bit"
FILES = [{"path": "transformer/model.safetensors", "size": 64}, {"path": "vae/vae.safetensors", "size": 16}]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHUTTER_HOME", str(tmp_path / ".shutter"))
    monkeypatch.delenv("SHUTTER_MODEL_SEARCH", raising=False)
    e = dict(cat.CATALOG[MID], files=FILES, check_files=["transformer", "vae"], legacy=[])
    monkeypatch.setitem(cat.CATALOG, MID, e)
    monkeypatch.setattr(cat, "remote_ready", lambda mid: False)   # never touches the network
    c = Config.load()
    c.ensure_dirs()
    c.save()
    return c, e, tmp_path


def write_copy(folder: Path, sizes=(64, 16)):
    for f, n in zip(FILES, sizes):
        p = folder / f["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"0" * n)


def row(cfg):
    return {r["id"]: r for r in cat.catalog_rows(cfg)}[MID]


def test_unregistered_copy_in_own_folder_is_offered_and_registered(env):
    cfg, e, _ = env
    write_copy(cat.models_home(cfg) / e["folder"])
    r = row(cfg)
    assert not r["installed"] and r["on_disk"] == {"dir": str(cat.models_home(cfg) / e["folder"]), "elsewhere": False}
    assert r["available"] and r["disk"]["ok"]
    done = cat.use_local(MID, cfg)
    assert done["state"] == "done" and done["local"]
    fresh = Config.load()
    assert MID in fresh.models and row(fresh)["installed"] and row(fresh)["on_disk"] is None


def test_copy_in_another_shuttrr_home_is_hard_linked(env):
    cfg, e, home = env
    other = home / ".shutter-h3test" / "models" / e["folder"]
    write_copy(other)
    r = row(cfg)
    assert r["on_disk"] == {"dir": str(other), "elsewhere": True}
    cat.use_local(MID, cfg)
    mine = cat.models_home(cfg) / e["folder"] / FILES[0]["path"]
    assert mine.is_file() and os.stat(mine).st_ino == os.stat(other / FILES[0]["path"]).st_ino
    assert Config.load().models[MID]["dir"].endswith(e["folder"])


def test_folder_registered_by_another_home_config_is_found(env):
    cfg, e, home = env
    elsewhere = home / "big-disk" / "zimage"
    write_copy(elsewhere)
    (home / ".shutter-test").mkdir()
    (home / ".shutter-test" / "config.json").write_text(json.dumps({"models": {MID: {"dir": str(elsewhere)}}}))
    assert row(cfg)["on_disk"]["dir"] == str(elsewhere)


def test_partial_copy_is_not_offered(env):
    cfg, e, home = env
    write_copy(home / ".shutter-x" / "models" / e["folder"], sizes=(10, 16))
    r = row(cfg)
    assert r["on_disk"] is None and not r["available"]
    with pytest.raises(cat.DownloadError):
        cat.use_local(MID, cfg)


def test_docs_from_another_revision_do_not_block(env, monkeypatch):
    cfg, e, home = env
    files = FILES + [{"path": "README.md", "size": 999}]
    monkeypatch.setitem(cat.CATALOG, MID, dict(e, files=files))
    other = home / ".shutter-old" / "models" / e["folder"]
    write_copy(other)
    (other / "README.md").write_text("older readme")
    assert row(cfg)["on_disk"]["dir"] == str(other)
