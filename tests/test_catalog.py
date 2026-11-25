import json
import os
from pathlib import Path

import pytest

from shutter.core import catalog as cat
from shutter.core.config import Config

GB = 1024**3
MID = "longcat-avatar-1.6-8bit"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("SHUTTER_HOME", str(tmp_path / "home"))
    c = Config.load()
    c.ensure_dirs()
    c.save()
    return c


def fake_fetch(repo, dest):
    for f in cat.CATALOG[MID]["check_files"]:
        p = Path(dest) / f
        if "." in p.name:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x")
        else:
            p.mkdir(parents=True, exist_ok=True)
            (p / "w.bin").write_bytes(b"0" * 10)


def test_download_registers_model_with_small_mac_preset(cfg, monkeypatch):
    monkeypatch.setattr(cat, "total_memory", lambda: 16 * GB)
    seen = []
    done = cat.download(MID, cfg, fetch=fake_fetch, total_of=lambda r: 31, on_progress=seen.append, poll_s=0.01)
    assert done["state"] == "done" and seen[0]["state"] == "downloading"
    fresh = Config.load()
    spec = fresh.models[MID]
    assert spec["adapter"] == "longcat-avatar" and spec["options"] == {"vae_bf16": True}
    assert (fresh.default_mode, fresh.default_efficiency) == ("low_memory", 1.0)
    row = cat.catalog_rows(fresh)[0]
    assert row["installed"] and row["download"]["state"] == "done" and row["notes"]


def test_big_mac_gets_no_preset_and_user_options_survive(cfg, monkeypatch):
    monkeypatch.setattr(cat, "total_memory", lambda: 128 * GB)
    cfg.models[MID] = {"adapter": "longcat-avatar", "dir": "/nowhere", "options": {"vae_tile": 32}}
    cfg.save()
    cat.download(MID, cfg, fetch=fake_fetch, total_of=lambda r: 31, poll_s=0.01)
    fresh = Config.load()
    assert fresh.models[MID]["options"] == {"vae_tile": 32} and fresh.default_mode == "fast"
    assert fresh.models[MID]["dir"].endswith(cat.CATALOG[MID]["folder"])


def test_failure_is_recorded_in_plain_words(cfg):
    def boom(repo, dest):
        raise ConnectionError("Connection reset by peer")

    with pytest.raises(cat.DownloadError, match="connection dropped"):
        cat.download(MID, cfg, fetch=boom, total_of=lambda r: 31, poll_s=0.01)
    st = cat.read_state(cfg, MID)
    assert st["state"] == "failed" and "carry on" in st["error"]
    assert MID not in Config.load().models


def test_incomplete_download_is_not_registered(cfg):
    with pytest.raises(cat.DownloadError, match="files are missing"):
        cat.download(MID, cfg, fetch=lambda r, d: None, total_of=lambda r: 31, poll_s=0.01)
    assert not cat.catalog_rows(cfg)[0]["installed"]


def test_not_enough_disk(cfg):
    with pytest.raises(cat.DownloadError, match="not enough free disk"):
        cat.download(MID, cfg, fetch=fake_fetch, total_of=lambda r: 10**18, poll_s=0.01)


def test_dead_download_reads_as_stopped(cfg):
    p = cat.state_path(cfg, MID)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"state": "downloading", "pid": 2**22 + 12345, "done_bytes": 5}))
    assert cat.read_state(cfg, MID)["state"] == "stopped"
    assert cat.cancel(MID, cfg) is False


def test_unknown_model(cfg):
    with pytest.raises(cat.DownloadError, match="unknown model"):
        cat.download("nope", cfg)


def test_doctor_rows(cfg):
    rows = cat.doctor(cfg)
    names = [r["check"] for r in rows]
    assert "Apple silicon Mac" in names and "LongCat Video Avatar 1.6" in names
    assert all(set(r) == {"check", "ok", "detail"} for r in rows)


def test_cli_catalog_and_doctor(cfg, capsys):
    from shutter import cli

    assert cli.main(["catalog"]) == 0
    assert "not downloaded" in capsys.readouterr().out
    cli.main(["--json", "doctor"])
    assert "checks" in json.loads(capsys.readouterr().out)


# ---- checks after a download, and the disk-space rule


def _listing(dest):
    import hashlib
    out = []
    for root, _d, files in os.walk(dest):
        for n in files:
            p = Path(root) / n
            rel = str(p.relative_to(dest))
            if rel.startswith(".cache"):
                continue
            out.append({"path": rel, "size": p.stat().st_size,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest() if p.suffix == ".bin" else None})
    return out


def test_download_verifies_checksums(cfg, tmp_path):
    ref = tmp_path / "ref"
    fake_fetch(None, ref)
    done = cat.download(MID, cfg, fetch=fake_fetch, files_of=lambda r: _listing(ref), poll_s=0.01)
    assert done["verified"] == "sha256"


def test_download_with_a_bad_file_fails_and_removes_it(cfg, tmp_path):
    ref = tmp_path / "ref"
    fake_fetch(None, ref)
    files = _listing(ref)
    bad = next(f for f in files if f["sha256"])
    bad["sha256"] = "0" * 64
    with pytest.raises(cat.DownloadError, match="did not match"):
        cat.download(MID, cfg, fetch=fake_fetch, files_of=lambda r: files, poll_s=0.01)
    dest = cat.models_home(cfg) / cat.CATALOG[MID]["folder"]
    assert not (dest / bad["path"]).exists()
    assert MID not in Config.load(cfg.path).models


def test_offline_download_says_how_it_was_checked(cfg):
    def offline(repo):
        raise OSError("offline")
    done = cat.download(MID, cfg, fetch=fake_fetch, files_of=offline, total_of=lambda r: 31, poll_s=0.01)
    assert done["verified"] == "files" and "could not be reached" in done["verified_detail"]


def test_disk_check_keeps_five_gb_spare(cfg, monkeypatch):
    import collections
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(cat.shutil, "disk_usage", lambda p: usage(0, 0, 4 * GB))
    with pytest.raises(cat.DownloadError, match="5 GB to spare"):
        cat.download(MID, cfg, fetch=fake_fetch, total_of=lambda r: 31, poll_s=0.01)
