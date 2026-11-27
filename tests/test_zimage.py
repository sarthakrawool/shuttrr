"""Z-Image family: a tiny random model in the mflux layout runs through the real pipeline.

Checks that every memory step gives the same image bit for bit, that LoRAs change the image only while active (and
the same on every step), LoRA name parsing, the adapter's estimates and refusals, and the catalog / request options.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("tokenizers")

from mlx.utils import tree_flatten  # noqa: E402

from shutter.families.zimage import lora as L  # noqa: E402
from shutter.families.zimage import model as Mo  # noqa: E402
from shutter.families.zimage.pipeline import Settings, ZImagePipeline  # noqa: E402

GB = 1024**3
DIT = dict(dim=192, n_heads=3, n_layers=4, n_refiner_layers=1, cap_feat_dim=64, axes_dims=[16, 24, 24])
TE = dict(vocab_size=320, hidden_size=64, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
          intermediate_size=128, head_dim=32)


def _save(folder: Path, params: dict, shard_bytes: int = 1 << 20) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    shards, cur, size = [], {}, 0
    for k, v in params.items():
        if size + v.nbytes > shard_bytes and cur:
            shards.append(cur)
            cur, size = {}, 0
        cur[k] = v
        size += v.nbytes
    if cur:
        shards.append(cur)
    wm = {}
    for i, sh in enumerate(shards):
        mx.save_safetensors(str(folder / f"{i}.safetensors"), sh, {"quantization_level": "8"})
        wm.update({k: f"{i}.safetensors" for k in sh})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"quantization_level": "8"},
                                                                      "weight_map": wm}))


def _randomize(module, seed: int) -> None:
    key = mx.random.key(seed)
    flat = []
    for i, (k, v) in enumerate(tree_flatten(module.parameters())):
        r = mx.random.normal(v.shape, key=mx.random.key(seed * 100003 + i)) * (0.5 if "norm" in k else 0.08)
        flat.append((k, (r + (1.0 if "norm" in k and k.endswith("weight") else 0.0)).astype(mx.bfloat16)))
    from mlx.utils import tree_unflatten
    module.update(tree_unflatten(flat))


def _tokenizer(folder: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    words = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "user", "assistant", "\n"] + \
            [chr(c) for c in range(33, 127)] + [f"w{i}" for i in range(150)]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<|endoftext|>"))
    tok.pre_tokenizer = pre_tokenizers.Split(pattern=" ", behavior="removed")
    tok.add_special_tokens(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    folder.mkdir(parents=True, exist_ok=True)
    tok.save(str(folder / "tokenizer.json"))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("zimage-tiny")
    dit = Mo.ZImageTransformer(**DIT)
    _randomize(dit, 1)
    nn.quantize(dit, group_size=64, bits=8, class_predicate=lambda p, m: isinstance(m, nn.Linear) and m.weight.shape[1] % 64 == 0)
    _save(d / "transformer", dict(tree_flatten(dit.parameters())))
    te = Mo.TextEncoder(n_layers=TE["num_hidden_layers"], **TE)
    _randomize(te, 2)
    nn.quantize(te, group_size=64, bits=8)
    _save(d / "text_encoder", dict(tree_flatten(te.parameters())))
    vae = Mo.VAE()
    _randomize(vae, 3)
    _save(d / "vae", dict(tree_flatten(vae.parameters())), shard_bytes=1 << 30)
    _tokenizer(d / "tokenizer")
    (d / "zimage.json").write_text(json.dumps({"dit": DIT, "te": TE}))
    return d


class Pool:
    def __init__(self):
        self.d = {}

    def get(self, key, loader, size, keep=True, **kw):
        if key not in self.d:
            self.d[key] = loader()
        return self.d[key]

    def release(self, key):
        self.d.pop(key, None)


def _run(d: Path, rung: str, pool=None, **kw) -> list:
    s = Settings(prompt="w1 w2 w3 hello", width=256, height=256, steps=3, seed=5, **kw)
    return ZImagePipeline(d, "tiny", pool=pool).generate(s, rung)


def _lora_file(path: Path, comfy_qkv: bool = False, seed: int = 9) -> Path:
    key = mx.random.key(seed)
    t = {}
    r = 4
    for i in range(DIT["n_layers"]):
        names = ["attention.to_q", "attention.to_k", "attention.to_v", "attention.to_out.0", "feed_forward.w1",
                 "feed_forward.w2", "feed_forward.w3", "adaLN_modulation.0"]
        dims = {"attention.to_q": (192, 192), "attention.to_k": (192, 192), "attention.to_v": (192, 192),
                "attention.to_out.0": (192, 192), "feed_forward.w1": (192, 512), "feed_forward.w2": (512, 192),
                "feed_forward.w3": (192, 512), "adaLN_modulation.0": (192, 768)}
        for j, n in enumerate(names):
            din, dout = dims[n]
            # q, k and v share one down matrix, as a fused qkv LoRA has
            jj = 0 if n in ("attention.to_k", "attention.to_v") else j
            a = mx.random.normal((r, din), key=mx.random.key(seed * 100003 + 100 * i + jj)) * 0.1
            b = mx.random.normal((dout, r), key=mx.random.key(seed * 100003 + 1000 + 100 * i + j)) * 0.1
            t[f"diffusion_model.layers.{i}.{n}.lora_A.weight"] = a
            t[f"diffusion_model.layers.{i}.{n}.lora_B.weight"] = b
        if comfy_qkv:
            a = t.pop(f"diffusion_model.layers.{i}.attention.to_q.lora_A.weight")
            t.pop(f"diffusion_model.layers.{i}.attention.to_k.lora_A.weight")
            t.pop(f"diffusion_model.layers.{i}.attention.to_v.lora_A.weight")
            bs = [t.pop(f"diffusion_model.layers.{i}.attention.to_{p}.lora_B.weight") for p in "qkv"]
            t[f"diffusion_model.layers.{i}.attention.qkv.lora_down.weight"] = a
            t[f"diffusion_model.layers.{i}.attention.qkv.lora_up.weight"] = mx.concatenate(bs, axis=0)
    mx.save_safetensors(str(path), t)
    return path


def test_every_memory_step_gives_the_same_image(tiny):
    import numpy as np
    ref = _run(tiny, "resident", Pool())
    assert ref[0].shape == (256, 256, 3) and ref[0].dtype == np.uint8
    for rung in ("split", "staged", "stream_wide", "stream_tight"):
        out = _run(tiny, rung, Pool())
        assert np.array_equal(out[0], ref[0]), rung


def test_count_uses_seed_plus_i(tiny):
    import numpy as np
    two = _run(tiny, "staged", count=2)
    one = ZImagePipeline(tiny, "tiny").generate(Settings(prompt="w1 w2 w3 hello", width=256, height=256, steps=3,
                                                         seed=6), "staged")
    assert np.array_equal(two[1], one[0]) and not np.array_equal(two[0], two[1])


def test_warm_model_is_reused_and_lora_does_not_stick(tiny, tmp_path):
    import numpy as np
    pool = Pool()
    base = _run(tiny, "resident", pool)
    lf = _lora_file(tmp_path / "l.safetensors")
    with_lora = _run(tiny, "resident", pool, loras=[{"path": str(lf), "strength": 1.0}])
    assert len(pool.d) == 1
    assert not np.array_equal(base[0], with_lora[0])
    again = _run(tiny, "resident", pool)            # same warm model, LoRA turned off
    assert np.array_equal(again[0], base[0])
    zero = _run(tiny, "staged", loras=[{"path": str(lf), "strength": 0.0}])
    assert np.array_equal(zero[0], base[0])


def test_lora_is_the_same_on_every_step(tiny, tmp_path):
    import numpy as np
    lf = str(_lora_file(tmp_path / "l.safetensors"))
    ref = _run(tiny, "resident", Pool(), loras=[lf])
    for rung in ("staged", "stream_tight"):
        assert np.array_equal(_run(tiny, rung, loras=[lf])[0], ref[0]), rung


def test_fused_qkv_lora_equals_split_names(tiny, tmp_path):
    import numpy as np
    a = _run(tiny, "staged", loras=[str(_lora_file(tmp_path / "a.safetensors"))])
    b = _run(tiny, "staged", loras=[str(_lora_file(tmp_path / "b.safetensors", comfy_qkv=True))])
    assert np.array_equal(a[0], b[0])


def test_lora_names():
    assert L.target("diffusion_model.layers.3.attention.to_q.lora_A.weight") == ("layers.3.attention.to_q", "down")
    assert L.target("transformer.layers.3.feed_forward.w2.lora_B.default.weight") == \
        ("layers.3.feed_forward.w2", "up")
    assert L.target("lora_unet_layers_12_attention_to_out_0.lora_down.weight") == \
        ("layers.12.attention.to_out.0", "down")
    assert L.target("lora_unet_noise_refiner_1_feed_forward_w1.alpha") == ("noise_refiner.1.feed_forward.w1", "alpha")
    assert L.target("diffusion_model.layers.0.attention.out.lora_up.weight") == ("layers.0.attention.to_out.0", "up")
    assert L.target("diffusion_model.final_layer.adaLN_modulation.1.lora_down.weight") == \
        ("all_final_layer.2-1.adaLN_modulation.0", "down")
    assert L.target("diffusion_model.layers.0.attention_norm1.diff") is None
    assert L.target("lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight") is None


def test_lora_for_another_model_is_refused(tmp_path):
    p = tmp_path / "wan.safetensors"
    mx.save_safetensors(str(p), {"diffusion_model.blocks.0.self_attn.q.lora_down.weight": mx.zeros((4, 8))})
    with pytest.raises(ValueError, match="no LoRA tensors for Z-Image"):
        L.read_lora(p)


def test_adapter_estimates_and_refusals(tiny, tmp_path):
    from shutter.adapters.zimage import ZImageAdapter
    from shutter.core import modes as M
    from shutter.types import JobRequest
    a = ZImageAdapter("tiny", tiny, {})
    assert a.can_load() == (True, "")
    req = JobRequest(model="tiny", prompt="x", width=512, height=512, params={"budget_bytes": 100 * GB})
    est = a.rung_estimates(req)
    peaks = [est[r].peak for r in M.RUNG_IDS]
    assert peaks == sorted(peaks, reverse=True)
    lf = _lora_file(tmp_path / "l.safetensors")
    req2 = JobRequest(model="tiny", prompt="x", width=512, height=512, params={"loras": [str(lf)]})
    assert a.rung_estimates(req2)["staged"].peak == est["staged"].peak + lf.stat().st_size
    bad = ZImageAdapter("x", tmp_path, {})
    ok, why = bad.can_load()
    assert not ok and "not a Z-Image build" in why
    with pytest.raises(ValueError, match="count"):
        a.settings_for(JobRequest(model="tiny", prompt="x", params={"count": 9}))
    with pytest.raises(ValueError, match="unknown style"):
        a.settings_for(JobRequest(model="tiny", prompt="x", params={"style": "studio-x"}))
    s = a.settings_for(JobRequest(model="tiny", prompt="a dog", width=1000, height=777,
                                  params={"style": "3d-animated"}))
    assert (s.width, s.height) == (992, 768) and s.prompt.startswith("3D animated film still: a dog")


def test_adapter_runs_and_writes_pngs(tiny, tmp_path):
    from shutter.adapters.base import RunContext
    from shutter.adapters.zimage import ZImageAdapter
    from shutter.types import Job, JobRequest
    a = ZImageAdapter("tiny", tiny, {})
    req = JobRequest(model="tiny", prompt="w1", width=256, height=256, steps=2, seed=3,
                     output=str(tmp_path / "out.png"), params={"count": 2, "budget_bytes": 100 * GB})
    job = Job(request=req)
    ctx = RunContext(models=Pool(), cache=None, work_dir=tmp_path, report=lambda p, s: None,
                     check_cancelled=lambda: None, memory_sample=lambda: {})
    res = a.run(job, ctx)
    assert res["outputs"] == [str(tmp_path / "out.png"), str(tmp_path / "out.s4.png")]
    assert all(Path(p).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for p in res["outputs"])
    assert res["kind"] == "image" and res["identical"] is True


def test_catalog_entry_and_fit():
    from shutter.core import catalog as C
    e = C.CATALOG["z-image-turbo-8bit"]
    assert e["kind"] == "image" and e["adapter"] == "z-image" and e["revision"]
    assert all(f["sha256"] for f in e["files"] if f["path"].endswith(".safetensors"))
    assert C.default_request_any("z-image-turbo-8bit", 16 * GB) == (1024, 1024, 1)
    fit = C.classify_fit("z-image-turbo-8bit", 16 * GB, baseline=3 * GB)
    assert fit["level"] == "recommended", fit
    assert C.classify_fit("z-image-turbo-8bit", 8 * GB, baseline=3 * GB)["level"] == "no_fit"   # not tested yet
    assert "images" in C.makes_line("z-image-turbo-8bit", 16 * GB)


def test_request_options_for_image(tmp_path, monkeypatch):
    from shutter.core import request_options as R
    from shutter.core.config import Config
    cfg = Config.load(str(tmp_path / "config.json"))
    cfg.models = {"z-image-turbo-8bit": {"adapter": "z-image", "dir": str(tmp_path / "zi"), "options": {}}}
    o = R.request_options(cfg, "z-image-turbo-8bit", total_bytes=16 * GB)
    assert o["kind"] == "image" and o["editable"] and o["lora"]
    assert set(o["defaults"]) == {"width", "height", "steps", "guidance"}
    assert {"id": "3d-animated", "label": "3D animated"} in o["styles"]
    assert any(s["width"] == 896 and s["height"] == 1152 and "portrait" in s["label"] for s in o["sizes"])
    e = R.estimate(cfg, "z-image-turbo-8bit", {"width": 1000, "height": 1024, "count": 2}, total_bytes=16 * GB)
    assert e["issues"] and e["issues"][0]["field"] == "width" and e["issues"][0]["fix"] == 992
    assert "2 images" in e["label"]


def test_registry_knows_the_family():
    from shutter.core.registry import BUILTIN_FAMILIES
    assert BUILTIN_FAMILIES["z-image"] == "shutter.adapters.zimage:ZImageAdapter"


def _req(**params):
    from shutter.types import JobRequest
    return JobRequest(model="tiny", prompt="x", width=1024, height=1024, params=params)


def test_small_mac_fast_mode_uses_the_lowest_step(tiny):
    from shutter.adapters.zimage import ZImageAdapter
    a = ZImageAdapter("tiny", tiny, {})
    pick = a.choose_rung(_req(mode="fast", total_bytes=16 * GB, budget_bytes=40 * GB))
    assert pick["rung"] == "stream_tight" and pick["why"] == "small Mac" and pick["fits"]
    # a pinned step or the Low memory slider is the person's choice and is kept
    assert a.choose_rung(_req(rung="staged", total_bytes=16 * GB, budget_bytes=40 * GB))["rung"] == "staged"
    assert a.choose_rung(_req(mode="low_memory", efficiency=0.5, total_bytes=16 * GB,
                              budget_bytes=40 * GB))["rung"] == "staged"


def test_fast_mode_keeps_a_margin_on_big_macs(tiny):
    from shutter.adapters.zimage import FAST_MARGIN_BYTES, ZImageAdapter
    a = ZImageAdapter("tiny", tiny, {})
    ests = a.rung_estimates(_req())
    wide = ests["stream_wide"].peak
    # the budget just covers stream_wide: without room for the margin, Fast takes the next step down
    pick = a.choose_rung(_req(mode="fast", total_bytes=64 * GB, budget_bytes=wide + FAST_MARGIN_BYTES // 2))
    assert pick["rung"] == "stream_tight" and pick["fits"]
    from shutter.core import modes as M
    budget = wide + FAST_MARGIN_BYTES
    pick = a.choose_rung(_req(mode="fast", total_bytes=64 * GB, budget_bytes=budget))
    assert pick["rung"] == next(r for r in M.RUNG_IDS if ests[r].peak + FAST_MARGIN_BYTES <= budget)
    assert ests[pick["rung"]].peak <= wide
    pick = a.choose_rung(_req(mode="fast", total_bytes=64 * GB, budget_bytes=100 * GB))
    assert pick["rung"] == "resident"
    # nothing fits with the margin: the lowest step, and it may still start if it fits without the margin
    tight = ests["stream_tight"].peak
    pick = a.choose_rung(_req(mode="fast", total_bytes=64 * GB, budget_bytes=tight + 1))
    assert pick["rung"] == "stream_tight" and pick["fits"]


def test_watchdog_stops_a_request_that_grows_swap():
    """0.2.14 let a run through that grew swap by 1.1 GB once and then held (16 GB M5, stream_wide at 1024)."""
    from shutter.core import memory as m
    from shutter.core.config import Config
    from shutter.core.memory import FakeProbe, MemoryGuard
    cfg = Config()
    assert cfg.swap_growth_bytes == 256 * 1024**2
    cfg.swap_growth_mb = 1024.0                 # the default 0.2.3 - 0.2.14 wrote into config.json
    assert cfg.swap_growth_bytes == 256 * 1024**2
    p = FakeProbe(swap_used=3 * GB)
    g = MemoryGuard(p, swap_growth_bytes=Config().swap_growth_bytes, grace_s=10)
    g.begin_run()
    p.set(swap_used=3 * GB + 300 * 1024**2)
    a, _ = g.watch_step(now=0)
    assert a == m.ACTION_RELIEVE
    p.set(swap_used=int(4.1 * GB))              # +1.1 GB, then holding: stopped at once (over 3 x 256 MB)
    a, why = g.watch_step(now=1)
    assert a == m.ACTION_STOP and "swap grew by" in why
