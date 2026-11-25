"""Upscale a generated clip to the edit size, optionally cancelling the model's slow camera push-in inside the
same resample (one resample per pixel, never two).

Method (ported from the measured tool ``speed/exact/tools/upwarp.py``): the source -> output map is the cover
scale plus centre crop ``S = [[s, 0, -ox], [0, s, -oy]]``; for frame n the model's push is a similarity ``M_n``
(frame 0 -> frame n) and the frame is warped once, with Lanczos-4, by ``S . inv(M_n)``. upwarp measures ``M_n``
per frame from background points (OpenCV LK + RANSAC); here ``M_n`` is either given (``transforms=``, e.g. from
that same measurement) or comes from a linear model: ``zoom_cancel`` = total push by the last frame, so frame n is
magnified by ``m_n = 1 + z * n / (N - 1)`` about the frame centre.

``anchor``:
  - ``"last"`` (default, what the Shutter brief asks for): the warp is ``S . diag(1 + z) . inv(M_n)``, i.e. frame n
    is scaled by ``s * (1 + z) / m_n``. The last frame is a plain upscale and every earlier frame is magnified a
    little more, so no pixel outside the source is ever sampled (no reflected border strips).
  - ``"first"``: exactly upwarp: ``S . inv(M_n)``; frame 0 is the plain upscale and later frames are shrunk, so a
    few rows at the frame edge come from BORDER_REFLECT (2.5 px at 0.7 % push into 1080p).

Backends for the warped path:
  - ``"cv2"``: ``cv2.warpAffine(..., INTER_LANCZOS4, BORDER_REFLECT)``, the same call upwarp makes. OpenCV is an
    optional dependency (``pip install opencv-python-headless``); asking for it without OpenCV raises a clear error.
  - ``"numpy"``: the same operation for axis-aligned transforms (scale + shift, which is all the linear zoom model
    produces): separable 8-tap Lanczos (a = 4, weights normalised to 1 like OpenCV), BORDER_REFLECT indexing, in
    float32. Differs from OpenCV only by OpenCV's 1/32 px coefficient tables and its fixed-point rounding (tested to
    within a couple of 8-bit levels when OpenCV is available). Transforms with rotation need ``cv2``.
  - ``"auto"``: cv2 when importable, else numpy.

With ``zoom_cancel == 0`` and no transforms the clip goes through ONE ffmpeg pass (``scale`` with Lanczos to the
cover size, then ``crop``), which is the plain upscale.

Why not a pure ffmpeg expression for the zoom cancel: checked on ffmpeg 6.1, ``scale`` with ``eval=frame`` and the
``n`` variable works, but output sizes are integers, so the per-frame zoom is quantised to 1-2 px steps (up to
0.05 % of 1920 per frame, a visible 0.5-1 px edge stutter) plus a 0.5 px centring jitter; ``zoompan`` crops on
integer pixels as well; ``pad`` + ``perspective`` (``eval=frame``, ``in`` variable) is sub-pixel exact but only
offers bilinear or bicubic, not Lanczos. None is equivalent to upwarp's single Lanczos warp, so the warp is done
here and ffmpeg only decodes and encodes.

``measure_zoom(path)`` estimates the push between the first and last frame from a static background, numpy only
(Gauss-Newton fit of scale + shift on a smoothed image, optionally restricted to the background columns). It is
reliable on static backgrounds (tested to < 0.01 % on synthetic clips); on real clips pass the person columns so
moving parts are ignored. upwarp's own OpenCV measurement remains the reference.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

__all__ = ["upscale", "measure_zoom", "parse_size", "cover_transform", "linear_push_transforms",
           "warp_affine_lanczos", "probe_video"]

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


# ----------------------------------------------------------------------------- small helpers


def parse_size(size) -> tuple[int, int]:
    """'1920x1080' | (1920, 1080) -> (W, H)."""
    if isinstance(size, str):
        w, h = size.lower().split("x")
        return int(w), int(h)
    w, h = size
    return int(w), int(h)


def probe_video(path, runner: Callable = subprocess.run) -> dict:
    """{'width', 'height', 'fps', 'frames', 'has_audio'} via ffprobe."""
    r = runner([FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(path)],
               capture_output=True, text=True, check=True)
    streams = json.loads(r.stdout)["streams"]
    v = next(s for s in streams if s.get("codec_type") == "video")
    num, den = (int(x) for x in v.get("r_frame_rate", "25/1").split("/"))
    frames = int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else None
    return {"width": int(v["width"]), "height": int(v["height"]), "fps": num / den if den else 25.0,
            "frames": frames, "has_audio": any(s.get("codec_type") == "audio" for s in streams)}


def _compose(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A after B for 2x3 affines."""
    A3 = np.vstack([A, [0, 0, 1]])
    B3 = np.vstack([B, [0, 0, 1]])
    return (A3 @ B3)[:2]


def _invert(M: np.ndarray) -> np.ndarray:
    M3 = np.vstack([M, [0, 0, 1]])
    return np.linalg.inv(M3)[:2]


def cover_transform(src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> np.ndarray:
    """upwarp's S: scale to cover the target, centre crop (OpenCV pixel convention: integer = pixel centre)."""
    W, H = src_wh
    OW, OH = dst_wh
    s = max(OW / W, OH / H)
    ox, oy = (W * s - OW) / 2, (H * s - OH) / 2
    return np.array([[s, 0, -ox], [0, s, -oy]], np.float64)


def linear_push_transforms(n_frames: int, size_wh: tuple[int, int], zoom: float) -> list[np.ndarray]:
    """M_n (frame 0 -> frame n) for a linear push-in of `zoom` by the last frame, about the frame centre."""
    W, H = size_wh
    cx, cy = (W - 1) / 2, (H - 1) / 2
    out = []
    for n in range(n_frames):
        m = 1.0 + zoom * (n / (n_frames - 1) if n_frames > 1 else 1.0)
        out.append(np.array([[m, 0, (1 - m) * cx], [0, m, (1 - m) * cy]], np.float64))
    return out


def frame_warps(n_frames: int, src_wh, dst_wh, zoom: float = 0.0,
                transforms: Optional[Sequence[np.ndarray]] = None, anchor: str = "last") -> list[np.ndarray]:
    """The single source->output affine for every frame."""
    S = cover_transform(src_wh, dst_wh)
    Ms = list(transforms) if transforms is not None else linear_push_transforms(n_frames, src_wh, zoom)
    if len(Ms) != n_frames:
        raise ValueError(f"{len(Ms)} transforms for {n_frames} frames")
    if anchor == "last":
        # reference = the last frame: magnify everything by its push so nothing outside the source is sampled
        A = _compose(S, np.asarray(Ms[-1], np.float64))
    elif anchor == "first":
        A = S
    else:
        raise ValueError("anchor must be 'last' or 'first'")
    return [_compose(A, _invert(np.asarray(M, np.float64))) for M in Ms]


# ----------------------------------------------------------------------------- numpy Lanczos-4 warp


def _reflect(idx: np.ndarray, n: int) -> np.ndarray:
    """OpenCV BORDER_REFLECT: fedcba|abcdefgh|hgfedcb."""
    if n == 1:
        return np.zeros_like(idx)
    period = 2 * n
    i = np.mod(idx, period)
    return np.where(i < n, i, period - 1 - i)


def _lanczos_taps(u: np.ndarray, n_src: int, a: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """For source coordinates u (pixel centres at integers): 2a tap indices and normalised weights."""
    i0 = np.floor(u).astype(np.int64)
    k = np.arange(-a + 1, a + 1)
    idx = i0[:, None] + k[None, :]
    x = u[:, None] - idx
    w = np.sinc(x) * np.sinc(x / a)
    w[np.abs(x) >= a] = 0.0
    w /= w.sum(axis=1, keepdims=True)
    return _reflect(idx, n_src), w.astype(np.float32)


def warp_affine_lanczos(img: np.ndarray, M: np.ndarray, dst_wh: tuple[int, int]) -> np.ndarray:
    """Same as cv2.warpAffine(img, M, dst_wh, INTER_LANCZOS4, BORDER_REFLECT) for an axis-aligned M
    (no rotation/shear). img uint8 [H, W, C] or float; returns the same dtype."""
    M = np.asarray(M, np.float64)
    if abs(M[0, 1]) > 1e-9 or abs(M[1, 0]) > 1e-9:
        raise ValueError("numpy backend handles scale + shift only; install OpenCV for rotation (backend='cv2')")
    OW, OH = dst_wh
    H, W = img.shape[:2]
    Mi = _invert(M)
    ux = Mi[0, 0] * np.arange(OW) + Mi[0, 2]
    uy = Mi[1, 1] * np.arange(OH) + Mi[1, 2]
    ix, wx = _lanczos_taps(ux, W)
    iy, wy = _lanczos_taps(uy, H)
    src = img.astype(np.float32)
    if src.ndim == 2:
        src = src[:, :, None]
    # columns first: [H, OW, C]
    tmp = np.einsum("hokc,ok->hoc", src[:, ix], wx, optimize=True)
    out = np.einsum("okwc,ok->owc", tmp[iy], wy, optimize=True)
    if img.ndim == 2:
        out = out[:, :, 0]
    if img.dtype == np.uint8:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out.astype(img.dtype)


def _have_cv2() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


def _warp_fn(backend: str, need_rotation: bool):
    if backend == "auto":
        backend = "cv2" if _have_cv2() else "numpy"
    if backend == "cv2":
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError("upscale backend 'cv2' needs OpenCV: pip install opencv-python-headless "
                               "(or use backend='numpy' for zoom-only transforms)") from e

        def f(frame, M, dst):
            return cv2.warpAffine(frame, M, dst, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        return f, "cv2"
    if backend == "numpy":
        if need_rotation:
            raise RuntimeError("these transforms contain rotation: the numpy backend cannot apply them; "
                               "install OpenCV (pip install opencv-python-headless) and use backend='cv2'")
        return warp_affine_lanczos, "numpy"
    raise ValueError(f"unknown backend {backend!r}")


# ----------------------------------------------------------------------------- ffmpeg I/O


def _encode_cmd(dst, wh, fps, crf, audio_from, raw_input: bool, vf: Optional[str] = None, src=None) -> list[str]:
    cmd = [FFMPEG, "-y", "-loglevel", "error"]
    if raw_input:
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{wh[0]}x{wh[1]}", "-r", f"{fps:g}", "-i", "-"]
    else:
        cmd += ["-i", str(src)]
    if audio_from:
        cmd += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy", "-shortest"]
    else:
        cmd += ["-map", "0:v:0"]
    if vf:
        cmd += ["-vf", vf]
    if fps and not raw_input:
        cmd += ["-r", f"{fps:g}"]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p", str(dst)]
    return cmd


def _decode_frames(src, wh) -> Iterable[np.ndarray]:
    W, H = wh
    p = subprocess.Popen([FFMPEG, "-loglevel", "error", "-i", str(src), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
    n = W * H * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    finally:
        p.stdout.close()
        p.wait()


def _count_frames(src, runner) -> int:
    r = runner([FFPROBE, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                "stream=nb_read_frames", "-of", "csv=p=0", str(src)], capture_output=True, text=True, check=True)
    return int(r.stdout.strip().split(",")[0])


# ----------------------------------------------------------------------------- public


def upscale(src, dst, size, zoom_cancel: float = 0.0, fps: Optional[float] = None, crf: int = 14,
            audio_from=None, runner: Callable = subprocess.run, *, transforms: Optional[Sequence] = None,
            anchor: str = "last", backend: str = "auto") -> dict:
    """Upscale `src` to `size` (e.g. '1920x1080'; cover + centre crop) with optional zoom cancel.

    zoom_cancel: total model push-in by the last frame (0.007 = 0.7 %). transforms: per-frame 2x3 affines
    frame0 -> frame n (overrides zoom_cancel). audio_from: file whose audio is muxed (default: src's own audio).
    `runner` executes the plain one-pass ffmpeg command and the probes (injectable for tests).
    Returns {"output", "size", "method", "frames"?}.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    OW, OH = parse_size(size)
    audio = audio_from if audio_from is not None else src
    plain = not zoom_cancel and transforms is None
    if plain:
        # one ffmpeg pass: Lanczos to the cover size, centre crop (same geometry as upwarp's plain path)
        vf = (f"scale={OW}:{OH}:force_original_aspect_ratio=increase:flags=lanczos+accurate_rnd+full_chroma_int,"
              f"crop={OW}:{OH}")
        cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(src)]
        if audio_from is not None:
            cmd += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a:0?"]
        else:
            cmd += ["-map", "0:v:0", "-map", "0:a:0?"]
        cmd += ["-vf", vf]
        if fps:
            cmd += ["-r", f"{fps:g}"]
        cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "copy",
                str(dst)]
        runner(cmd, check=True, capture_output=True)
        return {"output": str(dst), "size": [OW, OH], "method": "ffmpeg-lanczos"}

    info = probe_video(src, runner)
    W, H = info["width"], info["height"]
    in_fps = fps or info["fps"] or 25.0
    n = len(transforms) if transforms is not None else (info["frames"] or _count_frames(src, runner))
    warps = frame_warps(n, (W, H), (OW, OH), zoom=zoom_cancel, transforms=transforms, anchor=anchor)
    rot = any(abs(A[0, 1]) > 1e-9 or abs(A[1, 0]) > 1e-9 for A in warps)
    warp, used = _warp_fn(backend, rot)

    cmd = _encode_cmd(dst, (OW, OH), in_fps, crf, audio, raw_input=True)
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    done = 0
    try:
        for i, frame in enumerate(_decode_frames(src, (W, H))):
            A = warps[min(i, n - 1)].astype(np.float64)
            out = warp(np.ascontiguousarray(frame), A, (OW, OH))
            enc.stdin.write(np.ascontiguousarray(out).tobytes())
            done += 1
    finally:
        enc.stdin.close()
        rc = enc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg encode failed ({rc}) for {dst}")
    return {"output": str(dst), "size": [OW, OH], "method": f"warp-lanczos4-{used}", "frames": done,
            "zoom_cancel": zoom_cancel, "anchor": anchor}


# ----------------------------------------------------------------------------- measuring


def _read_frame_gray(path, index: int, runner=subprocess.run) -> np.ndarray:
    info = probe_video(path, runner)
    W, H = info["width"], info["height"]
    if index < 0:
        index = (info["frames"] or _count_frames(path, runner)) + index
    r = runner([FFMPEG, "-loglevel", "error", "-i", str(path), "-vf", f"select=eq(n\\,{index})", "-vsync", "0",
                "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.uint8)[: W * H].reshape(H, W).astype(np.float32)


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    r = int(3 * sigma + 0.5)
    x = np.arange(-r, r + 1)
    k = np.exp(-x * x / (2 * sigma * sigma))
    k /= k.sum()
    pad = np.pad(img, r, mode="reflect")
    tmp = sum(k[i] * pad[:, i:i + img.shape[1]] for i in range(2 * r + 1))
    return sum(k[i] * tmp[i:i + img.shape[0], :] for i in range(2 * r + 1))


def _bilinear(img: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    H, W = img.shape
    X = np.clip(X, 0, W - 1.001)
    Y = np.clip(Y, 0, H - 1.001)
    x0 = np.floor(X).astype(np.int64)
    y0 = np.floor(Y).astype(np.int64)
    fx, fy = X - x0, Y - y0
    a = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    b = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return a * (1 - fy) + b * fy


def fit_zoom(first: np.ndarray, last: np.ndarray, column_mask: Optional[np.ndarray] = None,
             sigma: float = 1.5, iters: int = 30) -> tuple[float, float, float]:
    """Least-squares (scale, tx, ty) with last(x) = first(c + (x - c - t) / scale). Returns (scale - 1, tx, ty)."""
    f = _blur(first.astype(np.float64), sigma)
    g = _blur(last.astype(np.float64), sigma)
    H, W = f.shape
    cx, cy = (W - 1) / 2, (H - 1) / 2
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    m = 0.08
    valid = (xs > W * m) & (xs < W * (1 - m)) & (ys > H * m) & (ys < H * (1 - m))
    if column_mask is not None:
        valid &= np.asarray(column_mask, bool)[None, :]
    gy, gx = np.gradient(f)
    s, tx, ty = 1.0, 0.0, 0.0
    for _ in range(iters):
        X = cx + (xs - cx - tx) / s
        Y = cy + (ys - cy - ty) / s
        w = _bilinear(f, X, Y)
        r = (w - g)[valid]
        Ix = _bilinear(gx, X, Y)[valid]
        Iy = _bilinear(gy, X, Y)[valid]
        dX, dY = X[valid] - cx, Y[valid] - cy
        # d X / d s = -(x - c - t)/s^2 = -dX / s ; d X / d tx = -1/s
        J = np.stack([-(Ix * dX + Iy * dY) / s, -Ix / s, -Iy / s], axis=1)
        delta, *_ = np.linalg.lstsq(J, -r, rcond=None)
        s, tx, ty = s + delta[0], tx + delta[1], ty + delta[2]
        if np.abs(delta[0]) < 1e-8 and np.abs(delta[1:]).max() < 1e-5:
            break
    return float(s - 1.0), float(tx), float(ty)


def measure_zoom(path, person_columns: Optional[tuple[int, int]] = None, margin: int = 24,
                 runner: Callable = subprocess.run) -> float:
    """Push-in between the first and last frame of `path` (0.007 = 0.7 %), from the static background.
    `person_columns` = (x0, x1) in clip pixels to ignore (the speaker), widened by `margin`."""
    first = _read_frame_gray(path, 0, runner)
    last = _read_frame_gray(path, -1, runner)
    mask = None
    if person_columns is not None:
        mask = np.ones(first.shape[1], bool)
        mask[max(0, person_columns[0] - margin):person_columns[1] + margin] = False
    return fit_zoom(first, last, mask)[0]
