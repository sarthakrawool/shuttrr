import json
import socket
import threading
import time
import urllib.request

import pytest

from shutter import cli
from shutter.core.config import Config
from shutter.core.engine import Engine
from shutter.core.memory import FakeProbe


def test_parse_submit_args():
    p = cli.build_parser()
    a = p.parse_args(["submit", "--model", "m", "--prompt", "hi", "--param", "framing=small",
                      "--seed", "7", "--duration", "3.5", "--param", "steps=4", "--width", "64", "--steps", "3",
                      "--image", "/tmp/a.png", "--input", "mask=/tmp/m.png",
                      "--param", "name=abc", "--param", "flags={\"a\": 1}", "--wait"])
    body = cli.build_request(a)
    assert body["model"] == "m" and body["seed"] == 7 and "framing" not in body
    assert body["width"] == 64 and body["steps"] == 3 and body["client"].startswith("shutter-cli/")
    assert body["inputs"] == {"image": "/tmp/a.png", "mask": "/tmp/m.png"}
    assert body["duration_s"] == 3.5 and a.wait
    assert body["params"] == {"framing": "small", "steps": 4, "name": "abc", "flags": {"a": 1}}


def test_parse_errors(capsys):
    p = cli.build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["submit", "--model", "m", "--param", "novalue"])
    with pytest.raises(SystemExit):
        p.parse_args(["submit", "--model", "m", "--framing", "small"])   # gone: use --param framing=small
    with pytest.raises(SystemExit):
        p.parse_args(["submit"])


def test_other_commands_parse():
    p = cli.build_parser()
    assert p.parse_args(["serve", "--host", "127.0.0.1", "--port", "9"]).port == 9
    assert p.parse_args(["status", "abc"]).id == "abc"
    assert p.parse_args(["status"]).id is None
    assert p.parse_args(["cancel", "x"]).id == "x"
    assert p.parse_args(["models"]).func is cli.cmd_models
    assert p.parse_args(["system"]).func is cli.cmd_system
    assert p.parse_args(["init", "--force"]).force


def test_init_writes_config(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SHUTTER_HOME", str(tmp_path / "h"))
    assert cli.main(["init"]) == 0
    data = json.loads((tmp_path / "h" / "config.json").read_text())
    assert data["port"] == 8484 and data["models"] == {}
    assert cli.main(["init"]) == 1


def test_unreachable_daemon(capsys):
    assert cli.main(["--url", "http://127.0.0.1:1", "system"]) == 1
    assert "cannot reach" in capsys.readouterr().err


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live(tmp_path):
    uvicorn = pytest.importorskip("uvicorn")
    from shutter.server.api import create_app

    cfg = Config(home=str(tmp_path / "home"), memory_poll_s=0.01, busy_process_pattern=None)
    eng = Engine(cfg, probe=FakeProbe(), protected=[], watch_interval_s=0, sample_interval_s=0.01)
    eng.start()
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(eng), host="127.0.0.1", port=port,
                                           log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    t0 = time.monotonic()
    while not server.started:
        assert time.monotonic() - t0 < 10
        time.sleep(0.01)
    yield f"http://127.0.0.1:{port}", eng
    server.should_exit = True
    t.join(5)
    eng.stop(5)


def test_submit_wait_end_to_end(live, capsys):
    url, eng = live
    rc = cli.main(["--url", url, "submit", "--model", "dummy", "--prompt", "hello",
                   "--param", "steps=3", "--param", "step_s=0", "--wait", "--poll", "0.02"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "done in" in out
    assert cli.main(["--url", url, "status"]) == 0
    assert cli.main(["--url", url, "system"]) == 0
    assert cli.main(["--url", url, "models"]) == 0
    out = capsys.readouterr().out
    assert "dummy" in out and "pool" in out
    # failing job -> exit code 1
    rc = cli.main(["--url", url, "submit", "--model", "dummy", "--param", "fail=nope",
                   "--param", "step_s=0", "--wait", "--poll", "0.02"])
    assert rc == 1
    assert cli.main(["--url", url, "submit", "--model", "missing"]) == 1


def test_sse_disconnect_cleans_up(live):
    url, eng = live
    resp = urllib.request.urlopen(url + "/v1/events", timeout=5)
    assert resp.readline().startswith(b":")
    assert eng.bus.subscriber_count == 1
    resp.close()
    t0 = time.monotonic()
    while eng.bus.subscriber_count:
        assert time.monotonic() - t0 < 5, "SSE subscription not closed after disconnect"
        time.sleep(0.05)


def test_mode_options_parse():
    p = cli.build_parser()
    a = p.parse_args(["submit", "--model", "m", "--mode", "low-memory", "--efficiency", "0.75"])
    assert cli.build_request(a)["params"] == {"mode": "low_memory", "efficiency": 0.75}
    assert "params" not in cli.build_request(p.parse_args(["submit", "--model", "m"]))
    for bad in (["--mode", "turbo"], ["--efficiency", "2"], ["--efficiency", "x"]):
        with pytest.raises(SystemExit):
            p.parse_args(["submit", "--model", "m"] + bad)
    assert p.parse_args(["settings"]).func is cli.cmd_settings


def test_settings_end_to_end(live, capsys):
    url, eng = live
    assert cli.main(["--url", url, "settings"]) == 0
    out = capsys.readouterr().out
    assert "mode        fast" in out and "* 0.00  resident" in out and "stream_tight" in out
    assert cli.main(["--url", url, "settings", "--mode", "low_memory", "--efficiency", "0.5"]) == 0
    out = capsys.readouterr().out
    assert "mode        low memory" in out and "* 0.50  staged" in out
    assert cli.main(["--url", url, "--json", "settings", "--efficiency", "1"]) == 0
    v = json.loads(capsys.readouterr().out)
    assert v["mode"] == "low_memory" and v["rung"] == "stream_tight"
    eng.pause()
    assert cli.main(["--url", url, "--json", "submit", "--model", "dummy", "--mode", "fast"]) == 0
    v = json.loads(capsys.readouterr().out)
    assert v["request"]["params"] == {"mode": "fast", "efficiency": 1.0}


def test_uninstall_removes_home_and_can_keep_models(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "stop_engine", lambda port, timeout=10.0: [])
    home = tmp_path / "shutter-home"
    (home / "models" / "m").mkdir(parents=True)
    (home / "models" / "m" / "w.bin").write_bytes(b"1")
    (home / "venv" / "bin").mkdir(parents=True)
    (home / "config.json").write_text("{}")
    res = cli.uninstall(str(home), keep_models=True)
    assert sorted(p.name for p in home.iterdir()) == ["models"] and res["kept"]
    res = cli.uninstall(str(home))
    assert not home.exists() and res["removed"]


def test_uninstall_refuses_the_home_folder(monkeypatch):
    import os
    monkeypatch.setattr(cli, "stop_engine", lambda port, timeout=10.0: [])
    with pytest.raises(ValueError):
        cli.uninstall(os.path.expanduser("~"))


def test_uninstall_needs_yes_without_a_terminal(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SHUTTER_HOME", str(tmp_path / "h"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert cli.main(["uninstall"]) == 2


# ---------------------------------------------------------------- inspect / convert (no daemon)


def _tiny_repo(root):
    import numpy as np

    from tests import wan_fixtures as F
    F.save_st(root / "diffusion_pytorch_model.safetensors", F.official_dit_weights())
    F.save_st(root / "text_encoder" / "model.safetensors", F.t5_weights())
    F.write_tokenizer_stub(root / "google" / "umt5-xxl")
    F.save_st(root / "Wan2.1_VAE.safetensors", {"conv2.weight": np.zeros((16, 16, 1, 1, 1), np.float32),
                                                "decoder.conv1.weight": np.ones((8, 16, 3, 3, 3), np.float32)})
    return root


@pytest.fixture
def home(tmp_path, monkeypatch):
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from shutter.core import catalog
    monkeypatch.setenv("SHUTTER_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(catalog, "total_memory", lambda: 16 * 1024**3)
    return tmp_path / "home"


def test_parse_lora_spec(tmp_path):
    from shutter.families.wan.sources import parse_lora
    assert parse_lora("a.safetensors:0.8") == ("a.safetensors", 0.8)
    assert parse_lora("a.safetensors") == ("a.safetensors", 1.0)
    assert parse_lora("/x/odd:name/a.safetensors") == ("/x/odd:name/a.safetensors", 1.0)
    assert parse_lora("/x/odd:name/a.safetensors:-0.5") == ("/x/odd:name/a.safetensors", -0.5)
    assert parse_lora("a.safetensors:nan") == ("a.safetensors:nan", 1.0)
    f = tmp_path / "l.safetensors"
    f.write_bytes(b"x")
    a = cli.build_parser().parse_args(["convert", "src", "out", "--lora", f"{f}:0.8", "--lora", str(f),
                                       "--bits", "4", "--add", "my-wan"])
    assert a.lora == [(str(f), 0.8), (str(f), 1.0)] and a.bits == 4 and a.add == "my-wan"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["convert", "src", "out", "--lora", str(tmp_path / "missing.safetensors")])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["convert", "src", "out", "--bits", "6"])


def test_inspect_tiny_repo(home, tmp_path, capsys):
    repo = _tiny_repo(tmp_path / "repo")
    assert cli.main(["inspect", str(repo)]) == 0
    out = capsys.readouterr().out
    assert "Wan-architecture model (3 blocks, width 256)" in out and "text to video" in out
    assert "convert it first" in out and "video model" in out and "text encoder" in out and "VAE" in out
    assert "this Mac  can run it (16.0 GB" in out and "verdict   ready to convert" in out
    assert cli.main(["--json", "inspect", str(repo)]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["ok"] and d["family"] == "wan" and d["this_mac"]["fits"] is True and "dit" in d["parts"]


def test_inspect_bare_file_and_problems(home, tmp_path, capsys):
    import numpy as np

    from tests import wan_fixtures as F
    f = tmp_path / "ft.safetensors"
    F.save_st(f, F.to_comfy(F.official_dit_weights()))
    assert cli.main(["inspect", str(f)]) == 0
    out = capsys.readouterr().out
    assert "missing   text encoder, VAE, tokenizer" in out
    bad = tmp_path / "clip.safetensors"
    F.save_st(bad, F.official_dit_weights(extra={"img_emb.proj.1.weight": np.zeros((256, 1280), np.float32)}))
    assert cli.main(["inspect", str(bad)]) == 1
    out = capsys.readouterr().out
    assert "problem   Not supported" in out and "CLIP" in out and "verdict   cannot run" in out
    other = tmp_path / "other.safetensors"
    F.save_st(other, {"foo.weight": np.zeros((2, 2), np.float32)})
    assert cli.main(["inspect", str(other)]) == 1
    assert "not a model Shuttrr recognises" in capsys.readouterr().out
    assert cli.main(["inspect", str(tmp_path / "nope")]) == 1


def test_convert_add_registers_build(home, tmp_path, capsys):
    from shutter.families import inspect

    repo = _tiny_repo(tmp_path / "repo")
    out = tmp_path / "build"
    assert cli.main(["convert", str(repo), str(out), "--add", "my-wan"]) == 0
    assert "Added my-wan" in capsys.readouterr().out
    cfg = Config.load()
    assert cfg.models["my-wan"] == {"adapter": "wan", "dir": str(out.resolve())}
    ck = inspect(out)
    assert ck.ok and not ck.needs_conversion
    # a build folder is refused the second time, with the reason
    assert cli.main(["convert", str(repo), str(out)]) == 2
    assert "already holds a build" in capsys.readouterr().err


def test_convert_finds_missing_parts_in_installed_models(home, tmp_path, capsys):
    """A bare fine-tune takes the text encoder and tokenizer from an installed build, the VAE from LongCat."""
    import shutil

    from tests import wan_fixtures as F
    repo = _tiny_repo(tmp_path / "repo")
    models = home / "models"
    assert cli.main(["convert", str(repo), str(models / "wan-built")]) == 0
    lc = models / "LongCat-Video-Avatar"
    (lc / "vae").mkdir(parents=True)
    shutil.copy(repo / "Wan2.1_VAE.safetensors", lc / "vae" / "model.safetensors")
    capsys.readouterr()
    f = tmp_path / "ft.safetensors"
    F.save_st(f, F.to_comfy(F.official_dit_weights(seed=5)))
    lora = tmp_path / "style.safetensors"
    F.save_st(lora, F.lora_weights())
    assert cli.main(["convert", str(f), str(tmp_path / "ft-build"), "--lora", f"{lora}:0.5"]) == 0
    out = capsys.readouterr().out
    assert "Using the text encoder from" in out and "wan-built" in out
    assert f"Using the VAE from {lc}" in out
    man = json.loads((tmp_path / "ft-build" / "shuttrr.json").read_text())
    assert man["loras_merged"] == [{"file": "style.safetensors", "strength": 0.5}]


def test_convert_main_lora_strength(tmp_path, monkeypatch):
    from shutter.families.wan import convert as C
    seen = {}
    monkeypatch.setattr(C, "convert", lambda src, out, **kw: seen.update(kw))
    assert C.main(["s", "o", "--lora", "a.safetensors:0.8", "--lora", "b.safetensors"]) == 0
    assert seen["loras"] == [("a.safetensors", 0.8), ("b.safetensors", 1.0)]
