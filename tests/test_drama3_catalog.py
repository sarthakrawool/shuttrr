"""Fish Audio Drama 3 in the catalog: audio fit from measured peaks, unpublished entries hidden, licence gate."""

from __future__ import annotations

from shutter.core import catalog as cat
from shutter.core.config import Config

GB = cat.GB


def test_audio_fit_uses_measured_peaks():
    lv = lambda m, g: cat.classify_fit(m, g * GB, baseline=3 * GB)["level"]
    assert "fish-drama3-5bit" not in cat.CATALOG          # ships later (public recalibration first)
    assert [lv("fish-drama3-8bit", g) for g in (16, 32, 48, 64, 96, 128)] == \
        ["no_fit", "no_fit", "no_fit", "barely", "barely", "recommended"]
    r = cat.classify_fit("fish-drama3-8bit", 64 * GB, baseline=3 * GB)["reason"]
    assert "Not yet tested on this Mac size" in r and "47.5 GB" in r
    assert "Needs a Mac with 64 GB" in cat.classify_fit("fish-drama3-8bit", 48 * GB, baseline=3 * GB)["reason"]
    assert "47.5 GB" in cat.memory_line("fish-drama3-8bit", 128 * GB)
    assert "faster than real time" in cat.makes_line("fish-drama3-8bit", 128 * GB)
    assert cat.CATALOG["fish-drama3-8bit"]["beta"] and "Built with Fish Audio" in cat.CATALOG["fish-drama3-8bit"]["summary"]


def test_unpublished_is_hidden_unless_asked(tmp_path, monkeypatch):
    cfg = Config(home=str(tmp_path))
    monkeypatch.delenv("SHUTTER_SHOW_UNPUBLISHED", raising=False)
    assert "fish-drama3-8bit" not in {r["id"] for r in cat.catalog_rows(cfg)}
    monkeypatch.setenv("SHUTTER_SHOW_UNPUBLISHED", "1")
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
