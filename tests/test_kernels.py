"""Reference implementations vs plain MLX / NumPy on CPU; Metal kernels only where Metal exists."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from shutter.kernels import adaln_mod, flash_attn, norm_rope, qlinear8  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HAS_METAL = bool(mx.metal.is_available())
metal = pytest.mark.skipif(not HAS_METAL, reason="Metal is not available (CPU-only MLX)")


def f32(a):
    return np.array(a.astype(mx.float32))


def bf16_ulp(x):
    """One bf16 ulp at |x| (8 significant bits)."""
    e = np.floor(np.log2(np.maximum(np.abs(x), 1e-30)))
    return 2.0 ** (e - 7)


# ------------------------------------------------------------------------------------------ adaln_mod
def test_adaln_reference_matches_numpy():
    x, sh, sc = adaln_mod.make_inputs(1, 3, 64, 512)
    ref = f32(adaln_mod.reference(x, sh, sc))
    xs = np.array(x.astype(mx.float32), dtype=np.float64)
    mean = xs.mean(-1, keepdims=True)
    var = ((xs - mean) ** 2).mean(-1, keepdims=True)
    y = (xs - mean) / np.sqrt(var + 1e-6) * (np.array(sc) + 1) + np.array(sh)
    assert np.all(np.abs(ref - y) <= bf16_ulp(y) + 1e-6)


def test_adaln_fast_path_is_close_to_port():
    x, sh, sc = adaln_mod.make_inputs(1, 3, 64, 512)
    a, b = f32(adaln_mod.reference(x, sh, sc)), f32(adaln_mod.reference_fast(x, sh, sc))
    assert np.all(np.abs(a - b) <= bf16_ulp(a) + 1e-6)


def test_adaln_dispatch_uses_reference_without_metal():
    x, sh, sc = adaln_mod.make_inputs(1, 2, 16, 256)
    if not HAS_METAL:
        assert mx.array_equal(adaln_mod.adaln_mod(x, sh, sc), adaln_mod.reference(x, sh, sc))


# ------------------------------------------------------------------------------------------ norm_rope
def test_rope_split_for_head_dim_128():
    f = norm_rope.rope_freqs((2, 3, 4), 128)
    assert f.shape == (24, 128)
    f = np.array(f)
    # t, h, w sub-dims 44 / 42 / 42; token (t=1, h=0, w=0) has only temporal phases
    tok = 1 * 12
    assert np.all(f[tok, 44:] == 0) and np.all(f[tok, :44:2] > 0)
    assert np.all(f[tok, 0::2] == f[tok, 1::2])  # each frequency used for one (even, odd) pair


def test_norm_rope_reference_matches_numpy():
    grid, H = (2, 4, 6), 4
    qkv, wq, wk = norm_rope.make_inputs(grid, heads=H)
    q, k, v = norm_rope.reference(qkv, wq, wk, grid, heads=H)
    N, D = 48, 128
    x = np.array(qkv.astype(mx.float32), dtype=np.float64).reshape(1, N, 3, H, D).transpose(2, 0, 3, 1, 4)
    ang = np.array(norm_rope.rope_freqs(grid, D), dtype=np.float64)
    for t, w, got in ((x[0], wq, q), (x[1], wk, k)):
        n = t / np.sqrt((t ** 2).mean(-1, keepdims=True) + 1e-6) * np.array(w.astype(mx.float32))
        z = n[..., 0::2] + 1j * n[..., 1::2]  # (even, odd) pair as a complex number, rotated by the angle
        r = z * np.exp(1j * ang[None, None, :, 0::2])
        exp = np.stack([r.real, r.imag], -1).reshape(n.shape)
        assert np.max(np.abs(f32(got) - exp)) < 0.05  # two bf16 roundings on values ~ O(3)
    assert mx.array_equal(v, qkv.reshape(1, N, 3, H, D).transpose(2, 0, 3, 1, 4)[2])


def test_signed_sin_rotation_is_bit_identical_to_rotate_half():
    grid, D = (2, 4, 6), 128
    x = mx.random.normal((1, 2, 48, D)).astype(mx.float32)
    f = norm_rope.rope_freqs(grid, D)
    a = x * mx.cos(f) + norm_rope._rotate_half(x) * mx.sin(f)
    cos, sin = norm_rope.rope_tables(grid, D)
    sw = x.reshape(1, 2, 48, D // 2, 2)[..., ::-1].reshape(1, 2, 48, D)
    assert mx.array_equal(a, x * cos + sw * sin)


def test_norm_rope_fast_path_is_close_to_port():
    grid = (2, 4, 6)
    qkv, wq, wk = norm_rope.make_inputs(grid, heads=4)
    a = norm_rope.reference(qkv, wq, wk, grid, heads=4)
    b = norm_rope.reference_fast(qkv, wq, wk, grid, heads=4)
    for x, y in zip(a, b):
        x, y = f32(x), f32(y)
        assert np.all(np.abs(x - y) <= 2 * bf16_ulp(x) + 1e-6)


# ------------------------------------------------------------------------------------------- qlinear8
def test_uint8_view_layout_matches_mx_dequantize():
    x, w, s, b, bias = qlinear8.make_inputs(8, 256, 64)
    assert w.dtype == mx.uint32 and w.shape == (64, 64) and s.shape == (64, 4)
    assert mx.array_equal(qlinear8.dequantize_ref(w, s, b), mx.dequantize(w, s, b, group_size=64, bits=8))
    # element j of a row is byte j%4 of word j/4, low bits first
    words = np.array(w)
    bytes_ = np.array(w.view(mx.uint8))
    assert np.array_equal(bytes_[:, 5], (words[:, 1] >> 8) & 0xFF)


def test_qlinear8_reference_matches_quantized_matmul_fp32():
    x, w, s, b, bias = qlinear8.make_inputs(64, 512, 96)
    ref = f32(qlinear8.reference(x.astype(mx.float32), w, s.astype(mx.float32), b.astype(mx.float32)))
    got = f32(mx.quantized_matmul(x.astype(mx.float32), w, s.astype(mx.float32), b.astype(mx.float32),
                                  transpose=True, group_size=64, bits=8))
    assert np.max(np.abs(ref - got)) <= 1e-3 * np.max(np.abs(ref))


def test_qlinear8_reference_matches_mlx_linear_bf16():
    x, w, s, b, bias = qlinear8.make_inputs(64, 512, 96)
    ref = f32(qlinear8.reference(x, w, s, b, bias))
    got = f32(qlinear8.mlx_path(x, w, s, b, bias))
    assert np.max(np.abs(ref - got)) <= 0.1 * np.max(np.abs(ref))  # CPU bf16 qmm accumulates coarsely


# ----------------------------------------------------------------------------------------- flash_attn
@pytest.mark.parametrize("nq,nk,bk", [(40, 72, 16), (33, 100, 32), (64, 64, 64)])
def test_online_softmax_tiled_matches_sdpa(nq, nk, bk):
    q, k, v = (mx.random.normal((1, 2, n, 128)) for n in (nq, nk, nk))
    a = f32(flash_attn.reference_tiled(q, k, v, bk=bk))
    b = f32(flash_attn.reference(q, k, v))
    c = f32(flash_attn.reference_naive(q, k, v))
    assert np.max(np.abs(a - b)) < 1e-5 and np.max(np.abs(c - b)) < 1e-5


def test_online_softmax_bf16_close_not_identical():
    q, k, v = flash_attn.make_inputs(1, 2, 100, 150)
    a, b = f32(flash_attn.reference_tiled(q, k, v, bk=32)), f32(flash_attn.reference(q, k, v))
    assert np.max(np.abs(a - b)) < 0.02


# ------------------------------------------------------------------------------------- Metal kernels
@metal
@pytest.mark.parametrize("mod,tol", [(adaln_mod, 0.07), (norm_rope, 0.07), (qlinear8, 0.02), (flash_attn, 0.02)])
def test_kernel_selftest(mod, tol):
    r = mod.selftest()
    assert r["ran_kernel"]
    key = "max_rel_diff" if "max_rel_diff" in r else "max_abs_diff"
    assert r[key] <= tol, r


def test_selftests_run_without_metal():
    for mod in (adaln_mod, norm_rope, qlinear8, flash_attn):
        r = mod.selftest() if mod is not qlinear8 else mod.selftest(64, 256, 64)
        assert r["ran_kernel"] == HAS_METAL and "max_abs_diff" in r


def test_kernel_sources_declare_their_inputs():
    """Each Metal source refers only to names the generated signature will provide."""
    for mod, names in ((adaln_mod, ["x", "shift", "scale", "eps", "out"]),
                       (norm_rope, ["qkv", "wq", "wk", "cosv", "sinv", "eps", "qo", "ko", "vo"]),
                       (qlinear8, ["x", "w", "scales", "biases", "y"]),
                       (flash_attn, ["q", "k", "v", "scale", "o"])):
        for n in names:
            assert n in mod._SOURCE


# ------------------------------------------------------------------------------------------ bench script
def test_bench_refuses_without_yes_and_prints_command():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "bench_kernels.py")], capture_output=True, text=True)
    assert r.returncode == 2 and "gpulock.sh shutter-bench" in r.stdout
    h = subprocess.run([sys.executable, str(ROOT / "scripts" / "bench_kernels.py"), "--help"], capture_output=True,
                       text=True, check=True).stdout
    assert "--yes" in h and "gpulock.sh" in h


def test_bench_is_self_contained():
    src = (ROOT / "scripts" / "bench_kernels.py").read_text()
    assert "import shutter" not in src and "from shutter" not in src
    for k in ("adaln_mod", "norm_rope", "qlinear8", "flash_attn"):
        ksrc = (ROOT / "shutter" / "kernels" / f"{k}.py").read_text()
        assert "from shutter" not in ksrc and "import shutter" not in ksrc
