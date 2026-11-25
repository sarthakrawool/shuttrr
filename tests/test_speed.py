"""Opt-in speed techniques (NOT bit-identical): block-sparse attention, TeaCache step caching, their LongCat hooks
(real patched tree, read-only, tiny random DiT on CPU) and the adapter's `params.speed` plumbing."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from shutter.core.step_cache import TeaCache, rel_l1

ROOT = Path(__file__).resolve().parent.parent
REAL_CODE = Path("/home/claude/ref/shutter-ref/longcat-mlx-code")


# ============================================================================ TeaCache (pure, numpy)


def _toy_run(cache, steps=10, drift=0.01, seed=0):
    """A fake sampler: the 'block stack' is a fixed nonlinear map, the indicator moves by `drift` per step."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((8, 8)) * 0.3
    x = rng.standard_normal((4, 8))
    x0 = x.copy()
    outs = []
    stack = lambda z: np.tanh(z @ w) + z            # noqa: E731
    for i in range(steps):
        ind = x0 * (1.0 + drift * i)
        if cache is None:
            y = stack(x)
        else:
            cache.new_step(i)
            y = cache.run(x, ind, stack)
        outs.append(y)
        x = 0.9 * x + 0.1 * y
    if cache is not None:
        cache.finish()
    return outs


def test_teacache_threshold_zero_is_identical():
    ref = _toy_run(None)
    c = TeaCache(threshold=0.0, num_steps=10)
    got = _toy_run(c)
    assert all(np.array_equal(a, b) for a, b in zip(ref, got))
    st = c.stats()
    assert st["skipped"] == 0 and st["computed"] == 10 and st["skipped_steps"] == []


def test_teacache_skips_reuse_residual_and_respect_warmup_and_last():
    c = TeaCache(threshold=0.05, num_steps=10, warmup=2)
    _toy_run(c, drift=0.01)
    st = c.stats()
    skipped = [s for _, s, _ in st["skipped_steps"]]
    assert st["skipped"] > 0 and st["computed"] + st["skipped"] == 10
    assert all(s >= 2 for s in skipped) and 9 not in skipped        # warm-up and last step computed
    # accumulation: rel_l1 of step i vs i-1 is ~0.01/(1+0.01 i): 0.05 allows ~4 skips in a row at most
    runs = max(len(list(g)) for k, g in __import__("itertools").groupby(e[3] for e in c.events) if k == "skip")
    assert runs <= 5
    # skipped output = input + stored residual, exactly
    c2 = TeaCache(threshold=1e9, num_steps=3, warmup=0, keep_last=False)
    x0, x1 = np.ones((2, 2)), np.full((2, 2), 2.0)
    c2.new_step(0)
    y0 = c2.run(x0, x0, lambda z: z * 3)
    c2.new_step(1)
    y1 = c2.run(x1, x1 * 1.001, lambda z: z * 100)
    assert np.array_equal(y0, x0 * 3) and np.array_equal(y1, x1 + (x0 * 3 - x0))


def test_teacache_slots_runs_invalidate_cap_and_shape():
    c = TeaCache(threshold=1e9, warmup=0, keep_last=False, max_consecutive=2)
    f = lambda z: z + 1                                           # noqa: E731
    x = np.ones(3)
    for i in range(6):
        c.new_step(i)
        c.run(x, x, f)                                            # slot 0
        c.run(x * 2, x * 2, f)                                    # slot 1 (e.g. the guidance pass)
    ev = [(e[1], e[2], e[3]) for e in c.events]
    for slot in (0, 1):
        assert [a for s, sl, a in ev if sl == slot] == ["compute", "skip", "skip", "compute", "skip", "skip"]
    c.new_step(6)
    c.invalidate()
    _, skip = c.begin(x, (3,))
    assert not skip                                               # invalidated: must compute
    c.new_step(0)                                                 # step index went back: new run, caches empty
    assert c.runs == 2 and c.slots == {}
    c.new_step(1)
    c.run(np.ones(4), np.ones(4), f)                               # (first call of the run computes anyway)
    c.new_step(2)
    _, skip = c.begin(np.ones(5), (5,))
    assert not skip                                               # shape changed: compute


def test_teacache_counts_steps_without_cached_calls_and_rel_l1():
    c = TeaCache(threshold=0.1, num_steps=4)
    c.new_step(0)
    c.run(np.ones(2), np.ones(2), lambda z: z)
    c.new_step(1)                 # e.g. a KV-cache step: no cached call
    c.new_step(2)
    c.finish()
    assert c.stats()["steps_without_cached_calls"] == 2
    assert rel_l1(np.array([2.0, 2.0]), np.array([1.0, 1.0])) == 1.0
    assert rel_l1(np.zeros(2), np.zeros(2)) == 0.0
    with pytest.raises(ValueError):
        TeaCache(threshold=-1)


def test_teacache_rescale_polynomial():
    c = TeaCache(threshold=1.0, coefficients=[2.0, 0.5])          # 2 x + 0.5
    assert c._rescale(0.25) == 1.0


# ============================================================================ block-sparse attention (MLX CPU)

mx = pytest.importorskip("mlx.core")
from shutter.kernels import block_sparse as BS  # noqa: E402


@pytest.mark.parametrize("grid,prefix,dtype", [((4, 8, 8), 0, "f32"), ((5, 6, 10), 1, "f32"),
                                               ((3, 7, 9), 2, "bf16")])
def test_sparsity_zero_equals_dense_exactly(grid, prefix, dtype):
    q, k, v = BS.make_inputs(grid, 2, 32, "random", 3, prefix_frames=prefix,
                             dtype=mx.bfloat16 if dtype == "bf16" else None)
    ref = BS.dense_reference(q, k, v)
    out, st = BS.block_sparse_attention(q, k, v, grid, sparsity=0.0, k_prefix=prefix * grid[1] * grid[2],
                                        return_stats=True)
    assert mx.array_equal(out, ref) and st["path"].startswith("dense") and st["pairs_skipped"] == 0
    forced = BS.block_sparse_attention(q, k, v, grid, sparsity=0.0, k_prefix=prefix * grid[1] * grid[2],
                                       force_gather=True)
    tol = 2e-2 if dtype == "bf16" else 2e-6
    assert float(mx.max(mx.abs(forced.astype(mx.float32) - ref.astype(mx.float32)))) < tol


def test_chunk_roundtrip_with_padding():
    x = mx.arange(2 * 3 * 5 * 6 * 7 * 4, dtype=mx.float32).reshape(2, 3, 5 * 6 * 7, 4)
    xc = BS.to_chunks(x, (5, 6, 7), (4, 4, 4))
    assert xc.shape == (2, 3, 2 * 2 * 2, 64, 4)
    assert mx.array_equal(BS.from_chunks(xc, (5, 6, 7), (4, 4, 4)), x)
    lay = BS.chunk_layout((5, 6, 7), (4, 4, 4))
    assert lay["valid"].sum() == 5 * 6 * 7 and lay["padded"]
    # chunk 0 holds tokens (t<4, h<4, w<4) in t, h, w order
    tok = np.array(xc[0, 0, 0, :, 0]).astype(int) // 4 % (5 * 6 * 7)
    t, h, w = np.unravel_index(tok, (5, 6, 7))
    assert t.max() == 3 and h.max() == 3 and w.max() == 3 and tok[1] == 1 and tok[4] == 7


def _numpy_block_sparse(q, k, v, grid, prefix, sparsity, chunk=(4, 4, 4), keep_prefix=True):
    """Brute force: build the token-level mask implied by the chunk selection, then masked dense softmax."""
    q, k, v = (np.asarray(a, np.float64)[0] for a in (q, k, v))       # [H, N, D]
    Hd, Nq, D = q.shape
    ql = BS.chunk_layout(grid, chunk)
    cs = ql["chunk_size"]
    T, H, W = grid
    ids = np.full(ql["valid"].shape, -1)
    tok = np.arange(Nq).reshape(T, H, W)
    (nt, nh, nw), (ct, ch, cw), (pt, ph, pw) = ql["n"], ql["chunk"], ql["pad"]
    tp = np.pad(tok, [(0, pt), (0, ph), (0, pw)], constant_values=-1)
    ids = tp.reshape(nt, ct, nh, ch, nw, cw).transpose(0, 2, 4, 1, 3, 5).reshape(nt * nh * nw, cs)
    npc = -(-prefix // cs) if prefix else 0
    pids = np.full((npc * cs,), -1)
    pids[:prefix] = np.arange(prefix)
    kids = np.concatenate([pids.reshape(npc, cs), np.where(ids >= 0, ids + prefix, -1)], 0)
    out = np.zeros_like(q)
    for h in range(Hd):
        def pool(x, idm):
            return np.stack([x[r[r >= 0]].mean(0) for r in idm])
        qp, kp = pool(q[h], ids), pool(k[h], kids)
        s = qp @ kp[npc:].T if keep_prefix else qp @ kp.T
        kk = BS.keep_count(s.shape[1], sparsity)
        for c in range(len(ids)):
            sel = np.argsort(-s[c])[:kk] + (npc if keep_prefix else 0)
            if keep_prefix:
                sel = np.concatenate([np.arange(npc), sel])
            keys = np.concatenate([kids[j][kids[j] >= 0] for j in sel])
            rows = ids[c][ids[c] >= 0]
            lg = q[h][rows] @ k[h][keys].T / math.sqrt(D)
            p = np.exp(lg - lg.max(1, keepdims=True))
            out[h][rows] = (p / p.sum(1, keepdims=True)) @ v[h][keys]
    return out[None]


@pytest.mark.parametrize("grid,prefix,sparsity,keep", [((4, 8, 8), 0, 0.75, True), ((5, 6, 10), 60, 0.5, True),
                                                       ((5, 6, 10), 60, 0.6, False), ((8, 8, 8), 0, 0.9375, True)])
def test_block_sparse_matches_bruteforce_masked_attention(grid, prefix, sparsity, keep):
    q, k, v = BS.make_inputs(grid, 2, 16, "random", 5, prefix_frames=prefix // (grid[1] * grid[2]))
    got = BS.block_sparse_attention(q, k, v, grid, sparsity=sparsity, k_prefix=prefix, keep_prefix=keep)
    ref = _numpy_block_sparse(q, k, v, grid, prefix, sparsity, keep_prefix=keep)
    assert np.max(np.abs(np.asarray(got) - ref)) < 1e-5


def test_grouping_does_not_change_the_result_and_compile_matches_eager():
    grid = (5, 6, 10)
    q, k, v = BS.make_inputs(grid, 4, 16, "structured", 2, prefix_frames=1)
    kw = dict(sparsity=0.75, k_prefix=60)
    a = BS.block_sparse_attention(q, k, v, grid, **kw)
    b = BS.block_sparse_attention(q, k, v, grid, head_chunk=1, q_chunk_group=3, **kw)
    assert float(mx.max(mx.abs(a - b))) < 1e-6
    f = mx.compile(lambda q_, k_, v_: BS.block_sparse_attention(q_, k_, v_, grid, **kw))
    assert float(mx.max(mx.abs(f(q, k, v) - a))) < 1e-5


def test_pattern_and_keep_count():
    assert BS.keep_count(16, 0.9375) == 1 and BS.keep_count(1386, 0.9375) == 87 and BS.keep_count(10, 0.0) == 10
    with pytest.raises(ValueError):
        BS.keep_count(10, 1.0)
    p = BS.pattern((23, 44, 82), k_prefix=3608, sparsity=0.9375)     # 704x1312 noise queries vs cond + noise keys
    assert p["q_chunks"] == p["k_chunks"] == 6 * 11 * 21 and p["prefix_chunks"] == 57
    assert p["kept_per_q_chunk"] == 57 + 87 and abs(p["score_fraction"] - 0.1138) < 1e-3
    q, k, v = BS.make_inputs((5, 6, 10), 1, 8, "random", 0, prefix_frames=1)
    _, st = BS.block_sparse_attention(q, k, v, (5, 6, 10), sparsity=0.5, k_prefix=60, return_stats=True)
    pat = BS.pattern((5, 6, 10), k_prefix=60, sparsity=0.5)
    assert all(st[kk] == pat[kk] for kk in pat)
    assert st["pairs_kept"] + st["pairs_skipped"] == st["pairs_total"]


def test_quality_numbers_are_measured_and_ordered():
    """Records PSNR / cosine vs dense at sparsity 0.5 / 0.75 / 0.9375 (also printed by `-s`); the values in
    docs/KERNELS.md come from BS.quality(). Synthetic data: random = worst case, structured / peaked = local."""
    rows = BS.quality()
    print("\n" + BS.quality_table(rows))
    by = {(r["kind"], r["sparsity"]): r for r in rows}
    for kind in ("random", "structured", "peaked"):
        ps = [by[(kind, s)]["psnr_db"] for s in (0.5, 0.75, 0.9375)]
        assert ps[0] > ps[1] > ps[2]
    assert by[("structured", 0.5)]["cosine"] > 0.99 and by[("peaked", 0.9375)]["cosine"] > 0.99
    assert by[("random", 0.9375)]["cosine"] < 0.5          # i.i.d. attention is not sparse: honest worst case


def test_selftest():
    r = BS.selftest()
    assert r["sparsity0_bit_identical"] and r["gather_all_max_abs"] < 1e-5


# ============================================================================ bench script


def test_bench_speed_refuses_without_yes_and_smoke_runs(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "bench_speed.py")], capture_output=True, text=True)
    assert r.returncode == 2 and "--yes" in r.stdout and "gpulock.sh" in r.stdout
    out = tmp_path / "speed.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "bench_speed.py"), "--smoke", "--out", str(out)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.loads(out.read_text())
    assert res["smoke"] is True and res["rows"]
    row = res["rows"][0]
    assert row["dense_s"] > 0 and all(s["s"] > 0 for s in row["sparse"])
    assert {s["sparsity"] for s in row["sparse"]} == {0.5, 0.75, 0.9375}
    assert all("cosine" in s for s in row["sparse"])


# ============================================================================ LongCat hooks on the real tree


TINY = {
    "in_channels": 4, "out_channels": 4, "hidden_size": 64, "depth": 3, "num_heads": 4, "caption_channels": 32,
    "mlp_ratio": 4, "adaln_tembed_dim": 32, "frequency_embedding_size": 32, "patch_size": [1, 2, 2],
    "text_tokens_zero_pad": True, "audio_window": 5, "audio_block": 5, "audio_channel": 32,
    "intermediate_dim": 32, "output_dim": 32, "context_tokens": 8, "vae_scale": 4, "audio_prenorm": False,
    "class_range": 24, "class_interval": 4,
}
T_LAT, H_LAT, W_LAT = 9, 16, 16           # 1 cond + 8 noise latents, 8x8 patches: noise grid 8x8x8 = 8 chunks


@pytest.fixture
def tree():
    if not REAL_CODE.is_dir():
        pytest.skip("patched LongCat tree not available")
    from shutter.adapters import longcat as L

    gen = L.import_generation(REAL_CODE)
    import importlib

    mods = {n: importlib.import_module(f"longcat_video_avatar.{n}") for n in ("fast", "kv_cache")}
    from longcat_video_avatar.models.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel

    yield {"gen": gen, "L": L, "Model": LongCatVideoAvatarTransformer3DModel, **mods}
    for name in [n for n in sys.modules if n.startswith("longcat_video_avatar")]:
        del sys.modules[name]
    if str(REAL_CODE.resolve()) in sys.path:
        sys.path.remove(str(REAL_CODE.resolve()))


def _rand(shape, seed):
    return mx.array(np.random.default_rng(seed).standard_normal(shape).astype(np.float32))


def _tiny(tree):
    mx.random.seed(0)
    dit = tree["Model"].from_config(TINY)
    mx.eval(dit.parameters())
    return dit


def _inputs(T=T_LAT, seed=1):
    lat = _rand((1, 4, T, H_LAT, W_LAT), seed)
    ts = mx.full((1, T), 600.0)
    ts[:, :1] = 0
    text, mask = _rand((1, 1, 6, 32), seed + 1), mx.array([[1, 1, 1, 1, 0, 0]])
    audio = _rand((1, 1 + 4 * (T - 1), 5, 5, 32), seed + 2)
    return lat, ts, text, mask, audio


def _speed(**kw):
    from shutter.adapters import longcat_speed as LSP

    sp = LSP.parse_speed(kw)
    sp.hw = (H_LAT // 2, W_LAT // 2)
    return sp


@pytest.mark.parametrize("compile_", [False, True])
def test_bsa_hook_sparsity_zero_is_bit_identical_and_uninstalls(tree, compile_):
    from shutter.adapters import longcat_speed as LSP

    fast = tree["fast"]
    dit = _tiny(tree)
    lat, ts, text, mask, audio = _inputs()
    fast.apply(dit, compile=compile_, fast_norms=compile_)
    ref = dit(lat, ts, text, audio, encoder_attention_mask=mask, num_cond_latents=1)
    fast.remove(dit)
    sp = _speed(block_sparse={"sparsity": 0.0, "min_q_tokens": 64})
    assert LSP.install_attention(dit, sp) == TINY["depth"]
    fast.apply(dit, compile=compile_, fast_norms=compile_)
    got = dit(lat, ts, text, audio, encoder_attention_mask=mask, num_cond_latents=1)
    assert mx.array_equal(got, ref)
    pats = list(sp.attn["sparse"].values())
    assert pats and all(p["nothing_skipped"] for p in pats)
    fast.remove(dit)
    assert LSP.uninstall(dit) == TINY["depth"]
    assert all(type(b.attn).__name__ == "Attention" for b in dit.blocks)
    fast.apply(dit, compile=compile_, fast_norms=compile_)
    assert mx.array_equal(dit(lat, ts, text, audio, encoder_attention_mask=mask, num_cond_latents=1), ref)


def test_bsa_hook_sparse_first_segment_and_kv_cache_path(tree):
    from shutter.adapters import longcat_speed as LSP

    fast, kv = tree["fast"], tree["kv_cache"]
    dit = _tiny(tree)
    lat, ts, text, mask, audio = _inputs()
    ref = dit(lat, ts, text, audio, encoder_attention_mask=mask, num_cond_latents=1)
    sp = _speed(block_sparse={"sparsity": 0.75, "min_q_tokens": 128})
    LSP.install_attention(dit, sp)
    fast.apply(dit, compile=True, fast_norms=True)
    got = dit(lat, ts, text, audio, encoder_attention_mask=mask, num_cond_latents=1)
    mx.eval(got)
    assert bool(mx.all(mx.isfinite(got))) and not mx.array_equal(got, ref)
    (sig, p), = sp.attn["sparse"].items()
    assert sig[0][2] == 8 * 64 and sig[1][2] == 9 * 64                   # noise queries vs cond + noise keys
    assert p["prefix_tokens"] == 64 and p["prefix_kept"] and p["q_chunks"] == 8 and p["selected_per_q_chunk"] == 2
    dense = {k[2] for k in sp.attn["dense"]}
    assert any("min_q_tokens" in r for r in dense)                       # cond frame attends to itself densely
    # continuation through the KV cache: ref + 4 cond latents cached, noise queries attend to them as a dense prefix
    sp2 = _speed(block_sparse={"sparsity": 0.5, "min_q_tokens": 300})
    fast.remove(dit)
    LSP.uninstall(dit)
    LSP.install_attention(dit, sp2)
    fast.apply(dit, compile=True, fast_norms=True)
    T, nc = 13, 5
    lat2, _, text, mask, _ = _inputs(T, seed=4)
    audio2 = _rand((1, 1 + 4 * (T - 2), 5, 5, 32), 9)
    cache = kv.build_cache(dit, lat2[:, :, :nc], H_LAT, W_LAT, T, 4, 1)
    pred = kv.forward_noise(dit, lat2[:, :, nc:], 600.0, text, mask, audio2, cache, T, 4, 1)
    mx.eval(pred)
    (sig2, p2), = sp2.attn["sparse"].items()
    assert sig2[0][2] == 8 * 64 and p2["prefix_tokens"] == 5 * 64 and p2["prefix_chunks"] == 5
    assert bool(mx.all(mx.isfinite(pred)))


def test_bsa_hook_reference_skip_runs_dense(tree):
    from shutter.adapters import longcat_speed as LSP

    dit = _tiny(tree)
    sp = _speed(block_sparse={"sparsity": 0.5, "min_q_tokens": 64})
    LSP.install_attention(dit, sp)
    attn = dit.blocks[0].attn
    T = 13
    x = _rand((1, T * 64, 64), 3)
    attn(x, (T, 8, 8), num_cond_latents=5, num_ref_latents=1, ref_img_index=10, mask_frame_range=3)
    assert not sp.attn["sparse"]
    assert any("reference-skip" in k[2] for k in sp.attn["dense"])


def _bare_gen(tree, dit, compile_=False):
    L, gen_mod = tree["L"], tree["gen"]
    cls = L.make_generator_class(gen_mod)
    g = object.__new__(cls)
    g.paths = gen_mod.ModelPaths(components=Path("."), dit=Path("."), loras={})
    g.lora_keys = []
    g.compile = compile_
    g.dit = dit
    g._anchor = None
    g._shutter_hook = None
    tree["fast"].apply(dit, compile=compile_, fast_norms=compile_)   # what _load_dit's _mount_fast did
    return g


def _denoise(tree, g, steps=8):
    lat, _, text, mask, audio = _inputs()
    st = tree["gen"].Settings(steps=steps)
    return g._denoise(lat, 1, text, mask, None, None, audio, st, log=lambda m: None)


@pytest.mark.parametrize("compile_", [False, True])
def test_step_cache_threshold_zero_identical_in_real_denoise_loop(tree, compile_):
    ref = _denoise(tree, _bare_gen(tree, _tiny(tree), compile_))
    g = _bare_gen(tree, _tiny(tree), compile_)
    sp = _speed(step_cache={"threshold": 0.0})
    g._shutter_set_speed(sp)
    got = _denoise(tree, g)
    sp.cache.finish()
    assert mx.array_equal(got, ref)
    st = sp.report()["step_cache"]
    assert st["applied"] and st["computed"] == 8 and st["skipped"] == 0 and len(st["rel_l1"]) == 8
    # clearing the speed options restores the tree's classes; output identical again
    g._shutter_set_speed(None)
    assert not any(hasattr(type(b), "_shutter_speed_base") for b in g.dit.blocks)
    assert mx.array_equal(_denoise(tree, g), ref)


def test_step_cache_skips_and_reports_in_real_denoise_loop(tree):
    g = _bare_gen(tree, _tiny(tree), True)
    ref = _denoise(tree, _bare_gen(tree, _tiny(tree), True))
    sp = _speed(step_cache={"threshold": 10.0, "warmup": 1})
    g._shutter_set_speed(sp)
    got = _denoise(tree, g)
    sp.cache.finish()
    rep = sp.report(steps_run=8)["step_cache"]
    assert rep["skipped"] > 0 and rep["computed"] + rep["skipped"] == 8
    skipped = [s for _, s, _ in rep["skipped_steps"]]
    assert 0 not in skipped and 7 not in skipped                       # warm-up and the last step always run
    assert bool(mx.all(mx.isfinite(got))) and not mx.array_equal(got, ref)


def test_both_hooks_together_and_remount_on_warm_generator(tree):
    g = _bare_gen(tree, _tiny(tree), True)
    sp = _speed(block_sparse={"sparsity": 0.5, "min_q_tokens": 64}, step_cache={"threshold": 0.0})
    g._shutter_set_speed(sp)
    out = _denoise(tree, g, steps=3)
    assert bool(mx.all(mx.isfinite(out)))
    rep = sp.report(steps_run=3)
    assert rep["identical"] is False and rep["block_sparse"]["applied"] and rep["step_cache"]["applied"]
    assert any(p["skipped_chunk_pairs_per_dit_call"] > 0 for p in rep["block_sparse"]["sparse_patterns"])
    assert type(g.dit.blocks[0].attn).__name__.startswith("BlockSparse")
    assert type(g.dit.blocks[0]).__name__.startswith("StepCached")


# ============================================================================ adapter plumbing (no model needed)


def test_parse_speed_defaults_and_errors():
    from shutter.adapters import longcat_speed as LSP

    assert LSP.parse_speed(None) is None and LSP.parse_speed({}) is None
    assert LSP.parse_speed({"block_sparse": False, "step_cache": {"enabled": False}}) is None
    sp = LSP.parse_speed({"block_sparse": True})
    assert sp.block_sparse.sparsity == 0.9375 and sp.block_sparse.chunk == (4, 4, 4) and sp.step_cache is None
    sp = LSP.parse_speed({"step_cache": {"threshold": 0.2}, "block_sparse": {"sparsity": 0.5, "chunk": [2, 4, 4]}})
    assert sp.step_cache.threshold == 0.2 and sp.block_sparse.chunk == (2, 4, 4)
    for bad in ({"turbo": True}, {"block_sparse": {"sparsity": 1.0}}, {"block_sparse": {"foo": 1}},
                {"step_cache": {"threshold": -1}}, {"block_sparse": {"chunk": [4, 4]}}, "fast"):
        with pytest.raises(ValueError):
            LSP.parse_speed(bad)
    st = LSP.parse_speed({"block_sparse": True, "strict": True})
    LSP.refusals_for(LSP.parse_speed({"block_sparse": True}), stream=True)
    with pytest.raises(ValueError):
        LSP.refusals_for(st, stream=True)


def test_adapter_rejects_bad_speed_at_estimate(tmp_path):
    from shutter.adapters import longcat as L
    from shutter.types import JobRequest

    ad = L.LongCatAvatarAdapter("lc", tmp_path, {})
    r = JobRequest(model="lc", prompt="p", image="i.png", audio="a.wav", duration_s=3.0,
                   params={"framing": "full", "image_size": "1920x1080", "speed": {"block_sparse": {"sparsity": 2}}})
    with pytest.raises(ValueError):
        ad.estimate(r)


def test_adapter_speed_for_notes_and_refusals(tmp_path):
    from shutter.adapters import longcat as L
    from shutter.types import JobRequest

    ad = L.LongCatAvatarAdapter("lc", tmp_path, {})
    r = JobRequest(model="lc", prompt="p", image="i.png", audio="a.wav",
                   params={"framing": "full", "image_size": "1920x1080",
                           "speed": {"block_sparse": True, "step_cache": True}})
    plan = ad.resolve(r)
    sp = ad.speed_for(r, plan)
    assert sp.hw == (44, 82) and sp.active
    assert any("NOT bit-identical" in n and "block_sparse" in n for n in plan.notes)
    plan2 = ad.resolve(r)
    sp2 = ad.speed_for(r, plan2, stream=True)
    assert not sp2.active and set(sp2.refused) == {"block_sparse", "step_cache"}
    rep = sp2.report()
    assert rep["identical"] is True and not rep["block_sparse"]["applied"]
    assert any("not applied" in n for n in plan2.notes)
    assert ad.speed_for(JobRequest(model="lc", prompt="p", params={}), plan) is None


# ============================================================================ adapter run() with the fake tree


def _adapter_helpers():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_lc_adapter_tests", ROOT / "tests" / "test_longcat_adapter.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_run_reports_speed_and_refuses_on_a_tree_without_mount_point(tmp_path):
    import textwrap

    H = _adapter_helpers()
    from shutter.adapters import longcat as L

    code = tmp_path / "code"
    pkg = code / "longcat_video_avatar"
    (pkg / "models").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "models" / "__init__.py").write_text("")
    (pkg / "models" / "attention.py").write_text("SDPA_HEAD_CHUNK = 1\n")
    (pkg / "fast.py").write_text("FFN_CHUNKS = 1\n")
    (pkg / "generation.py").write_text(textwrap.dedent(H.FAKE_GENERATION))
    (pkg / "roi.py").write_text(textwrap.dedent(H.FAKE_ROI))
    try:
        img, wav = tmp_path / "img.png", tmp_path / "vo.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=320x180",
                        "-frames:v", "1", str(img)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=220:r=48000:d=2",
                        str(wav)], check=True)
        ad = L.LongCatAvatarAdapter("lc", tmp_path / "model", {"code_dir": str(code)})
        pool = H.FakePool()
        base = dict(size="64x128", upscale_to="off", zoom_cancel="off")
        ctx, _ = H.make_ctx(tmp_path, pool)
        res = ad.run(H.job_for((img, wav), tmp_path, **base, speed={"block_sparse": True, "step_cache": True}), ctx)
        sp = res["speed"]
        assert set(sp["refused"]) == {"block_sparse", "step_cache"} and "_mount_fast" in sp["refused"]["block_sparse"]
        assert sp["identical"] is True and res["identical"] is True          # nothing was applied
        assert not sp["block_sparse"]["applied"] and sp["requested"]["block_sparse"]["sparsity"] == 0.9375
        assert any("speed.block_sparse not applied" in n for n in res["plan"]["notes"])
        ctx, _ = H.make_ctx(tmp_path, pool)
        res2 = ad.run(H.job_for((img, wav), tmp_path, **base), ctx)
        assert "speed" not in res2 and res2["identical"] is True
        with pytest.raises(ValueError):
            ctx, _ = H.make_ctx(tmp_path, pool)
            ad.run(H.job_for((img, wav), tmp_path, **base, speed={"block_sparse": True, "strict": True}), ctx)
    finally:
        for name in [n for n in sys.modules if n.startswith("longcat_video_avatar")]:
            del sys.modules[name]
        if str(code.resolve()) in sys.path:
            sys.path.remove(str(code.resolve()))
