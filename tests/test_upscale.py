"""post.upscale: the zoom cancel is folded into one Lanczos resample and really removes a known push-in."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from shutter.addons.reels import upscale as U

W, H, N, ZOOM = 512, 288, 50, 0.02
OUT = (1024, 576)


def _pattern(X, Y):
    rng = np.random.default_rng(0)
    fx, fy, ph = rng.uniform(0.03, 0.15, 12), rng.uniform(0.03, 0.15, 12), rng.uniform(0, 6.28, 12)
    return sum(np.cos(fx[i] * X + (1 if i % 2 else -1) * fy[i] * Y + ph[i]) for i in range(12))


def _zoom_frames(zoom=ZOOM, n=N, w=W, h=H):
    """Analytic linear push-in about the centre: frame k is the scene magnified by 1 + zoom * k / (n - 1)."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    out = []
    for k in range(n):
        m = 1 + zoom * k / (n - 1)
        g = np.clip(128 + 9 * _pattern(cx + (xs - cx) / m, cy + (ys - cy) / m), 0, 255).astype(np.uint8)
        out.append(np.stack([g, g // 2 + 60, 255 - g], -1))
    return np.stack(out)


def _decode(path, w, h):
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(-1, h, w).astype(np.float32)


@pytest.fixture(scope="module")
def zoom_clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("zoom")
    src = d / "src.mp4"
    frames = _zoom_frames()
    # lossless source, so the only push in it is the synthetic one (x264 at crf 10 measurably damps a slow zoom)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                    "-r", "25", "-i", "-", "-f", "lavfi", "-i", "sine=f=330:r=48000:d=2", "-map", "0:v", "-map", "1:a",
                    "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", "-c:a", "aac", "-shortest", str(src)],
                   input=frames.tobytes(), check=True)
    return src


def test_measure_zoom_on_known_push(zoom_clip):
    assert U.measure_zoom(zoom_clip) == pytest.approx(ZOOM, abs=2e-4)
    # ignoring a band of columns (the speaker) still measures the push from the rest
    assert U.measure_zoom(zoom_clip, person_columns=(200, 312)) == pytest.approx(ZOOM, abs=3e-4)


def test_fit_zoom_exact_frames():
    f = _zoom_frames(0.007, n=2)
    z, tx, ty = U.fit_zoom(f[0, :, :, 0], f[1, :, :, 0])
    assert z == pytest.approx(0.007, abs=3e-5) and abs(tx) < 0.02 and abs(ty) < 0.02


def test_zoom_cancel_removes_push(zoom_clip, tmp_path):
    # outputs encoded lossless (crf 0): lossy x264 damps slow zooms by a few %, which would hide a residual
    dst = tmp_path / "out.mp4"
    res = U.upscale(zoom_clip, dst, f"{OUT[0]}x{OUT[1]}", zoom_cancel=ZOOM, backend="numpy", crf=0)
    assert res["method"] == "warp-lanczos4-numpy" and res["frames"] == N
    info = U.probe_video(dst)
    assert (info["width"], info["height"]) == OUT and info["has_audio"]
    fr = _decode(dst, *OUT)
    assert len(fr) == N
    residual_last = U.fit_zoom(fr[0], fr[-1])[0]
    residual_mid = U.fit_zoom(fr[0], fr[N // 2])[0]
    assert abs(residual_last) < 1e-3, residual_last        # brief: < 0.1 %
    assert abs(residual_mid) < 1e-3, residual_mid
    # a plain upscale keeps the push (control)
    plain = tmp_path / "plain.mp4"
    U.upscale(zoom_clip, plain, OUT, crf=0)
    pf = _decode(plain, *OUT)
    assert U.fit_zoom(pf[0], pf[-1])[0] == pytest.approx(ZOOM, abs=3e-4)
    # anchor "last": the last frame is the plain upscale (same geometry), earlier frames are magnified
    assert abs(U.fit_zoom(pf[-1], fr[-1])[0]) < 2e-4
    assert U.fit_zoom(pf[0], fr[0])[0] == pytest.approx(ZOOM, abs=3e-4)


def test_anchor_first_matches_upwarp_geometry(zoom_clip, tmp_path):
    dst = tmp_path / "first.mp4"
    U.upscale(zoom_clip, dst, OUT, zoom_cancel=ZOOM, anchor="first", backend="numpy", crf=0)
    fr = _decode(dst, *OUT)
    plain = tmp_path / "plain.mp4"
    U.upscale(zoom_clip, plain, OUT, crf=0)
    pf = _decode(plain, *OUT)
    assert abs(U.fit_zoom(pf[0], fr[0])[0]) < 2e-4          # frame 0 untouched
    assert abs(U.fit_zoom(fr[0], fr[-1])[0]) < 1e-3


def test_plain_upscale_is_one_ffmpeg_pass(tmp_path):
    calls = []

    def runner(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    res = U.upscale("in.mp4", tmp_path / "o.mp4", "1920x1080", runner=runner)
    assert len(calls) == 1 and res["method"] == "ffmpeg-lanczos"
    vf = calls[0][calls[0].index("-vf") + 1]
    assert "lanczos" in vf and "scale=1920:1080:force_original_aspect_ratio=increase" in vf and "crop=1920:1080" in vf
    assert calls[0].count("-i") == 1                      # its own audio, copied
    U.upscale("in.mp4", tmp_path / "o.mp4", (1920, 1080), audio_from="vo.wav", runner=runner)
    assert "vo.wav" in calls[1]


def test_frame_warps_geometry():
    S = U.cover_transform((1312, 704), (1920, 1080))
    assert S[0, 0] == pytest.approx(1080 / 704) and S[0, 2] == pytest.approx(-(1312 * 1080 / 704 - 1920) / 2)
    ws = U.frame_warps(93, (1312, 704), (1920, 1080), zoom=0.007)
    assert np.allclose(ws[-1], S)                                      # last frame = plain upscale
    assert ws[0][0, 0] == pytest.approx(S[0, 0] * 1.007)               # first frame scaled by S * (1 + z)
    k = 40
    assert ws[k][0, 0] == pytest.approx(S[0, 0] * 1.007 / (1 + 0.007 * k / 92))
    # the frame centre stays the output centre
    c = np.array([(1312 - 1) / 2, (704 - 1) / 2, 1.0])
    for A in ws:
        assert np.allclose(A @ c, ws[-1] @ c)
    wf = U.frame_warps(93, (1312, 704), (1920, 1080), zoom=0.007, anchor="first")
    assert np.allclose(wf[0], S)
    # never samples outside the source with anchor="last"
    for A in ws:
        inv = U._invert(A)
        corners = np.array([[0, 0, 1], [1919, 0, 1], [0, 1079, 1], [1919, 1079, 1]], float).T
        src = inv @ corners
        assert src[0].min() >= -0.5 and src[0].max() <= 1311.5 and src[1].min() >= -0.5 and src[1].max() <= 703.5


def test_numpy_warp_identity_and_rotation_guard():
    rng = np.random.default_rng(3)
    img = (rng.random((20, 30, 3)) * 255).astype(np.uint8)
    out = U.warp_affine_lanczos(img, np.array([[1.0, 0, 0], [0, 1.0, 0]]), (30, 20))
    assert np.array_equal(out, img)
    with pytest.raises(ValueError):
        U.warp_affine_lanczos(img, np.array([[0.99, 0.1, 0], [-0.1, 0.99, 0]]), (30, 20))


def test_cv2_backend_missing_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(RuntimeError, match="opencv-python-headless"):
        U._warp_fn("cv2", False)
    f, used = U._warp_fn("auto", False)
    assert used == "numpy"
    with pytest.raises(RuntimeError, match="rotation"):
        U._warp_fn("numpy", True)


def test_numpy_backend_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur((rng.random((90, 160, 3)) * 255).astype(np.uint8), (0, 0), 1.2)
    for A in U.frame_warps(4, (160, 90), (240, 135), zoom=0.03) + \
            U.frame_warps(2, (160, 90), (240, 135), zoom=0.03, anchor="first"):
        a = cv2.warpAffine(img, A, (240, 135), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        b = U.warp_affine_lanczos(img, A, (240, 135))
        assert np.abs(a.astype(int) - b).max() <= 2
