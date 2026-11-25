"""The dashboard page and its static assets: served, offline, and wired to real API routes."""

import re

import pytest
from starlette.routing import Mount, Route
from starlette.testclient import TestClient

from shutter.core.config import Config
from shutter.core.engine import Engine
from shutter.core.memory import FakeProbe
from shutter.server.api import DASHBOARD, STATIC, create_app

PAGE = DASHBOARD.read_text(encoding="utf-8")


@pytest.fixture
def app(tmp_path):
    cfg = Config(home=str(tmp_path / "home"), memory_poll_s=0.01, busy_process_pattern=None)
    eng = Engine(cfg, probe=FakeProbe(), protected=[], watch_interval_s=0, sample_interval_s=0.01)
    return create_app(eng)


@pytest.fixture
def client(app):
    return TestClient(app)


def linked_assets():
    return re.findall(r'(?:href|src)="(/static/[^"]+)"', PAGE)


def css_urls(css: str):
    return [u for u in re.findall(r'url\(\s*["\']?([^"\')]+)["\']?\s*\)', css) if not u.startswith("data:")]


def test_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "mac-sidebar" in r.text and "/v1/events" in r.text


def test_linked_stylesheets_served_as_css(client):
    assets = linked_assets()
    assert {"/static/kit/tokens.css", "/static/kit/base.css", "/static/kit/macui.css", "/static/shutter.css"} <= set(assets)
    for path in assets:
        r = client.get(path)
        assert r.status_code == 200, path
        assert r.headers["content-type"].startswith("text/css"), path


def test_fonts_the_page_uses_are_served(client):
    for name in ("Inter-Variable.woff2", "GeistMono-Variable.woff2"):
        r = client.get(f"/static/fonts/{name}")
        assert r.status_code == 200
        assert r.headers["content-type"] == "font/woff2"
        assert r.content[:4] == b"wOF2"


def test_missing_static_file_is_404(client):
    assert client.get("/static/nope.css").status_code == 404


def test_static_needs_no_api_key_but_api_does(tmp_path):
    cfg = Config(home=str(tmp_path / "home"), memory_poll_s=0.01, busy_process_pattern=None)
    eng = Engine(cfg, probe=FakeProbe(), protected=[], watch_interval_s=0, sample_interval_s=0.01)
    c = TestClient(create_app(eng, api_key="s3cret"), client=("10.0.0.5", 5000))
    assert c.get("/static/shutter.css").status_code == 200
    assert c.get("/v1/system").status_code == 401


def test_no_external_hosts():
    texts = {"dashboard.html": PAGE}
    for p in STATIC.rglob("*.css"):
        texts[str(p.relative_to(STATIC))] = p.read_text(encoding="utf-8")
    for name, text in texts.items():
        # strip SVG namespace URIs inside data: icons; they are identifiers, never fetched
        cleaned = text.replace("http://www.w3.org/2000/svg", "")
        assert not re.search(r"(?:https?:)?//[a-z0-9.-]+\.[a-z]{2,}", cleaned, re.I), name
        assert "@import" not in cleaned, name


def test_css_urls_the_page_needs_resolve(client):
    # Every font face that the page can actually use must exist (tokens.css also declares unused faces).
    css = client.get("/static/kit/tokens.css").text
    used = [u for u in css_urls(css) if "Inter" in u or "GeistMono" in u]
    assert used
    for u in used:
        assert client.get("/static/kit/" + u).status_code == 200, u


def test_every_api_path_the_page_calls_exists(app):
    calls = set(re.findall(r'api\("(GET|POST|DELETE)",\s*"(/v1/[^"?]*)', PAGE))
    calls |= {("GET", m) for m in re.findall(r'EventSource\("(/v1/[^"?]+)', PAGE)}
    # the job cancel call builds its path: "/v1/jobs/" + id
    if re.search(r'api\("DELETE",\s*"/v1/jobs/"\s*\+', PAGE):
        calls.add(("DELETE", "/v1/jobs/{job_id}"))
    # pause/resume chooses the path with a ternary
    calls |= {("POST", p) for p in re.findall(r'"(/v1/queue/(?:pause|resume))"', PAGE)}
    # model Load / Unload build their path: "/v1/models/" + id + ("/load" or "/unload")
    if re.search(r'api\("POST",\s*"/v1/models/"\s*\+', PAGE):
        calls |= {("POST", "/v1/models/{model_id}/load"), ("POST", "/v1/models/{model_id}/unload")}
    calls = {(m, p.rstrip("/")) for m, p in calls
             if (p.rstrip("/") != "/v1/jobs" or m != "DELETE") and (p.rstrip("/") != "/v1/models" or m != "POST")}

    routes = {}
    for r in app.routes:
        if isinstance(r, Route):
            routes.setdefault(r.path, set()).update(r.methods or {"GET"})
    assert any(isinstance(r, Mount) and r.path == "/static" for r in app.routes)

    expected = {("GET", "/v1/system"), ("GET", "/v1/jobs"), ("POST", "/v1/jobs"), ("GET", "/v1/models"),
                ("DELETE", "/v1/jobs/{job_id}"), ("POST", "/v1/queue/pause"), ("POST", "/v1/queue/resume"),
                ("POST", "/v1/pool/release"), ("GET", "/v1/events"),
                ("GET", "/v1/settings"), ("POST", "/v1/settings"),
                ("POST", "/v1/models/{model_id}/load"), ("POST", "/v1/models/{model_id}/unload")}
    assert expected <= calls, expected - calls
    for method, path in calls:
        assert path in routes, path
        assert method in routes[path], (method, path)


def test_mode_control_is_accessible_and_static():
    # the slider is a real ARIA slider that names the rung, and the segmented switch is a radiogroup
    assert 'role="slider"' in PAGE and "aria-valuetext" in PAGE and 'id="mode-seg" role="radiogroup"' in PAGE
    for key in ("ArrowRight", "ArrowLeft", "Home", "End"):
        assert key in PAGE, key
    # follows other clients, honours reduced motion, never invents numbers
    assert '"settings.changed"' in PAGE and "prefers-reduced-motion" in PAGE
    assert "est_peak_bytes" in PAGE and 'textContent = "–"' in PAGE
    # the per-job override starts at "Use default"
    assert 'data-jmode="" class="is-on" aria-checked="true">Use default' in PAGE
    css = (STATIC / "shutter.css").read_text(encoding="utf-8")
    assert "cubic-bezier(0.22, 1, 0.36, 1)" in css and "prefers-reduced-motion" in css
    assert "gradient" not in css and "blur(" not in css
