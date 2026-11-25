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


# ---------------------------------------------------------------- Wan: download the source, then convert

WAN = "wan2.1-t2v-1.3b"
A14B = "wan2.2-t2v-a14b"


def tiny_wan_repo(root: Path) -> Path:
    """A tiny Wan repo in the official layout, with every part in safetensors (so no PyTorch is needed)."""
    import numpy as np

    from tests import wan_fixtures as F
    F.save_st(root / "diffusion_pytorch_model.safetensors", F.official_dit_weights())
    F.save_st(root / "text_encoder" / "model.safetensors", F.t5_weights())
    F.write_tokenizer_stub(root / "google" / "umt5-xxl")
    F.save_st(root / "Wan2.1_VAE.safetensors", {"conv2.weight": np.zeros((16, 16, 1, 1, 1), np.float32),
                                                "decoder.conv1.weight": np.ones((8, 16, 3, 3, 3), np.float32)})
    return root


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """huggingface_hub replaced by a module whose snapshot_download copies a tiny local repo."""
    import shutil
    import sys
    import types

    pytest.importorskip("mlx.core")
    src = tiny_wan_repo(tmp_path / "hub-repo")
    class Calls(list):
        src: Path

    calls = Calls()

    def snapshot_download(repo_id, local_dir, **kw):
        calls.append(repo_id)
        shutil.copytree(src, local_dir, dirs_exist_ok=True)
        return local_dir

    class HfApi:
        def model_info(self, repo, files_metadata=True):
            raise ConnectionError("offline")

    mod = types.SimpleNamespace(snapshot_download=snapshot_download, HfApi=HfApi)
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    calls.src = src
    return calls


def test_wan_rows(cfg, monkeypatch):
    monkeypatch.setattr(cat, "total_memory", lambda: 16 * GB)
    rows = {r["id"]: r for r in cat.catalog_rows(cfg)}
    assert rows[MID]["needs_conversion"] is False and list(rows)[0] == MID
    for mid, repo in (("wan2.1-t2v-1.3b", "Wan-AI/Wan2.1-T2V-1.3B"), ("wan2.2-ti2v-5b", "Wan-AI/Wan2.2-TI2V-5B"),
                      ("wan2.2-t2v-a14b", "Wan-AI/Wan2.2-T2V-A14B"), ("wan2.2-i2v-a14b", "Wan-AI/Wan2.2-I2V-A14B")):
        r = rows[mid]
        e = cat.CATALOG[mid]
        assert r["kind"] == "video" and r["source"] == repo and e["source_repo"] == repo and e["adapter"] == "wan"
        assert r["needs_conversion"] and r["checked_on"] == [] and r["size_approximate"] and not r["installed"]
        assert r["built_size_bytes"] < r["size_bytes"]
        assert r["dir"].endswith(f"{mid}-shuttrr-8bit")
    # the smallest Mac comes from the family memory check: one GB less fails it
    from shutter.families import family
    for mid in ("wan2.1-t2v-1.3b", "wan2.2-i2v-a14b"):
        ck = cat.wan_checkpoint(mid)
        need = cat.CATALOG[mid]["min_memory_gb"]
        assert not family("wan").check_memory(ck, need * GB)
        assert family("wan").check_memory(ck, (need - 1) * GB)
    # a 14B expert in 8-bit is ~15 GB: two of them plus the text encoder
    assert 35 < cat.CATALOG[A14B]["built_size_gb"] < 45


def test_wan_download_then_convert(cfg, monkeypatch, fake_hub):
    from shutter.families.wan import sources

    monkeypatch.setattr(cat, "total_memory", lambda: 64 * GB)
    monkeypatch.setattr(sources, "torch_available", lambda: True)
    seen = []
    done = cat.download(WAN, cfg, on_progress=seen.append, poll_s=0.01)
    assert fake_hub == ["Wan-AI/Wan2.1-T2V-1.3B"]
    states = [s["state"] for s in seen]
    assert states[0] == "downloading" and "converting" in states and states[-1] == "done"
    assert states.index("converting") > states.index("downloading")
    build = Path(done["dir"])
    assert build == cat.models_home(cfg) / f"{WAN}-shuttrr-8bit" and (build / "shuttrr.json").exists()
    assert (cat.models_home(cfg) / "sources" / "Wan2.1-T2V-1.3B" / "diffusion_pytorch_model.safetensors").exists()
    assert any("original download" in n for n in done["notes"])
    fresh = Config.load()
    assert fresh.models[WAN] == {"adapter": "wan", "dir": str(build), "options": {}}
    row = {r["id"]: r for r in cat.catalog_rows(fresh)}[WAN]
    assert row["installed"] and row["checked_on"] == [] and row["download"]["state"] == "done"
    # the checked-on list is read from the installed build
    man = json.loads((build / "shuttrr.json").read_text())
    man["checked_on"] = [{"mac": "M4 Max 128 GB", "date": "2026-11-01"}]
    (build / "shuttrr.json").write_text(json.dumps(man))
    row = {r["id"]: r for r in cat.catalog_rows(fresh)}[WAN]
    assert row["checked_on"][0]["mac"] == "M4 Max 128 GB"


def test_wan_download_refused_when_too_big_for_this_mac(cfg, monkeypatch, fake_hub):
    monkeypatch.setattr(cat, "total_memory", lambda: (cat.CATALOG[A14B]["min_memory_gb"] - 1) * GB)
    with pytest.raises(cat.DownloadError, match="cannot run on this Mac"):
        cat.download(A14B, cfg, poll_s=0.01)
    assert fake_hub == []                                     # nothing was fetched
    st = cat.read_state(cfg, A14B)
    assert st["state"] == "failed" and "Lowest memory" in st["error"]
    assert A14B not in Config.load().models


def test_wan_download_refused_without_pytorch_or_a_donor(cfg, monkeypatch, fake_hub):
    from shutter.families.wan import sources

    monkeypatch.setattr(cat, "total_memory", lambda: 64 * GB)
    monkeypatch.setattr(sources, "torch_available", lambda: False)
    with pytest.raises(cat.DownloadError, match="PyTorch"):
        cat.download(WAN, cfg, poll_s=0.01)
    assert fake_hub == []


def test_wan_download_borrows_parts_from_longcat(cfg, monkeypatch, fake_hub):
    """No PyTorch, but an installed LongCat-style model has a umT5 text encoder, tokenizer and a Wan 2.1 VAE."""
    import shutil

    from shutter.families.wan import sources

    monkeypatch.setattr(cat, "total_memory", lambda: 64 * GB)
    monkeypatch.setattr(sources, "torch_available", lambda: False)
    # the hub repo looks like the official one: text encoder and VAE as PyTorch files only
    repo = fake_hub.src
    shutil.rmtree(repo / "text_encoder")
    (repo / "Wan2.1_VAE.safetensors").unlink()
    (repo / "models_t5_umt5-xxl-enc-bf16.pth").write_bytes(b"pickle")
    (repo / "Wan2.1_VAE.pth").write_bytes(b"pickle")
    lc = cat.models_home(cfg) / "LongCat-Video-Avatar-1.6-MLX-8bit-LowMem"
    donor = tiny_wan_repo(Path(cfg.home) / "donor")
    shutil.copytree(donor / "text_encoder", lc / "components" / "text_encoder")
    shutil.copytree(donor / "google" / "umt5-xxl", lc / "components" / "tokenizer")
    (lc / "vae").mkdir(parents=True)
    shutil.copy(donor / "Wan2.1_VAE.safetensors", lc / "vae" / "model.safetensors")
    done = cat.download(WAN, cfg, poll_s=0.01)
    assert done["state"] == "done" and fake_hub == ["Wan-AI/Wan2.1-T2V-1.3B"]
    man = json.loads((Path(done["dir"]) / "shuttrr.json").read_text())
    assert set(man["parts"]) == {"dit", "text_encoder", "tokenizer", "vae"}


def test_converting_state_reads_as_stopped_when_the_process_died(cfg):
    p = cat.state_path(cfg, WAN)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"state": "converting", "pid": 2**22 + 12345, "done_bytes": 5}))
    assert cat.read_state(cfg, WAN)["state"] == "stopped"


def test_doctor_skips_wan_until_installed(cfg):
    names = [r["check"] for r in cat.doctor(cfg)]
    assert not any(n.startswith("Wan") for n in names)
