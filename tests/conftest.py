import pytest


@pytest.fixture(autouse=True)
def _known_chip(monkeypatch):
    """Tests do not depend on the Mac they run on: the catalog sees a Pro chip (the reference-recipe defaults)
    unless a test stands in another one with monkeypatch.setattr(catalog, "chip_info", ...)."""
    from shutter.core import catalog
    monkeypatch.setattr(catalog, "chip_info", lambda: {"name": "Apple M5 Pro", "tier": "pro", "gpu_cores": 20})


@pytest.fixture(autouse=True)
def _no_remote_checks(monkeypatch):
    """No network in tests: a build whose upload may still be running (catalog check_remote) looks not ready
    unless a test passes its own file list to catalog.remote_ready."""
    from shutter.core import catalog
    monkeypatch.setattr(catalog, "_remote_cache", {})
    monkeypatch.setattr(catalog, "_hf_files", lambda repo: [])
