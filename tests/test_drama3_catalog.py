"""Fish Audio Drama 3 in the catalog: audio fit from measured peaks, unpublished entries hidden, licence gate."""

from __future__ import annotations

from shutter.core import catalog as cat
from shutter.core.config import Config

GB = cat.GB


def test_audio_fit_uses_measured_peaks():
    lv = lambda m, g: cat.classify_fit(m, g * GB, baseline=3 * GB)["level"]
    assert "fish-drama3-5bit" not in cat.CATALOG          # ships later (public recalibration first)
    assert [lv("fish-drama3-8bit", g) for g in (16, 32, 48, 64, 96, 128)] == \
        ["no_fit", "no_fit", "barely", "barely", "barely", "recommended"]
    r = cat.classify_fit("fish-drama3-8bit", 64 * GB, baseline=3 * GB)["reason"]
    assert "Not yet tested on this Mac size" in r and "47.5 GB" in r
    assert "Needs a Mac with 48 GB" in cat.classify_fit("fish-drama3-8bit", 40 * GB, baseline=3 * GB)["reason"]
    assert "47.5 GB" in cat.memory_line("fish-drama3-8bit", 128 * GB)
    assert "faster than real time" in cat.makes_line("fish-drama3-8bit", 128 * GB)
    assert cat.CATALOG["fish-drama3-8bit"]["beta"] and "Built with Fish Audio" in cat.CATALOG["fish-drama3-8bit"]["summary"]


def test_fish_is_listed_with_its_licence(tmp_path):
    cfg = Config(home=str(tmp_path))
    rows = {r["id"]: r for r in cat.catalog_rows(cfg)}
    assert rows["fish-drama3-8bit"]["licence"]["name"].startswith("Fish Audio Community License")
    assert rows["fish-drama3-8bit"]["licence_accepted"] is False


def test_licence_acceptance_is_saved(tmp_path):
    cfg = Config(home=str(tmp_path), path=str(tmp_path / "config.json"))
    cfg.save()
    assert cat.licence_accepted(cfg, "wan2.1-t2v-1.3b-8bit")          # no licence: always fine
    assert not cat.licence_accepted(cfg, "fish-drama3-8bit")
    cat.accept_licence(cfg, "fish-drama3-8bit")
    assert cat.licence_accepted(Config.load(str(tmp_path / "config.json")), "fish-drama3-8bit")


def test_a14b_does_not_fit_under_64_gb_until_measured():
    for mid in ("wan2.2-t2v-a14b", "wan2.2-i2v-a14b"):
        for g in (16, 32, 48):
            f = cat.classify_fit(mid, g * GB, baseline=3 * GB)
            assert f["level"] == "no_fit" and "64 GB" in f["reason"], (mid, g, f)
        assert cat.classify_fit(mid, 64 * GB, baseline=3 * GB)["level"] == "barely"


def test_fish_downloads_only_once_every_pinned_file_is_on_hugging_face(tmp_path):
    files = cat.CATALOG["fish-drama3-8bit"]["files"]
    assert len(files) == 22 and all(f["sha256"] for f in files if f["path"].endswith(".safetensors"))
    assert not cat.remote_ready("fish-drama3-8bit", files_of=lambda r: files[:-1], now=0)        # half uploaded
    assert not cat.remote_ready("fish-drama3-8bit", files_of=lambda r: files, now=60)           # cached for 3 h
    assert cat.remote_ready("fish-drama3-8bit", files_of=lambda r: files, now=cat.REMOTE_CHECK_S + 1)
    assert cat.remote_ready("fish-drama3-8bit", files_of=lambda r: [], now=10**9)               # ready stays ready
    bad = [dict(f, sha256="0" * 64) if f["path"] == "tower.safetensors" else f for f in files]
    cat._remote_cache.clear()
    assert not cat.remote_ready("fish-drama3-8bit", files_of=lambda r: bad, now=0)
    row = {r["id"]: r for r in cat.catalog_rows(Config(home=str(tmp_path)))}["fish-drama3-8bit"]
    assert row["available"] is False and "still uploading" in row["availability_note"] and row["beta"]


def test_48_gb_fits_by_streaming_from_the_ssd():
    """48 GB: the whole model does not fit with room for apps, the SSD streaming steps do (same audio, ~3x slower)."""
    f = cat.classify_fit("fish-drama3-8bit", 48 * GB, baseline=3 * GB)
    assert f["level"] == "barely" and f["label"] == "Fits, about 3x slower" and f.get("streams")
    assert "SSD" in f["reason"] and "same audio" in f["reason"] and "Beta" in f["reason"]
    assert f["peak_bytes"] == int(39.4 * GB)                    # the 24 GB cache step fits with 4 GB for apps
    tight = cat.classify_fit("fish-drama3-8bit", 48 * GB, baseline=6 * GB)
    assert tight["peak_bytes"] == int(33.0 * GB)                # more kept by macOS: the 18 GB cache step
    assert "real time speed" in cat.makes_line("fish-drama3-8bit", 48 * GB)
    assert "faster than real time" in cat.makes_line("fish-drama3-8bit", 64 * GB)
    assert not cat.classify_fit("fish-drama3-8bit", 64 * GB, baseline=3 * GB).get("streams")
