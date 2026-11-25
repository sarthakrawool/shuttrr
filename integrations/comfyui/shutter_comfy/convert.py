"""Conversions between Shutter's files and ComfyUI's types, without ComfyUI.

- IMAGE is [B,H,W,C] float32 in 0..1, C=3 (docs/INTEGRATIONS.md). These functions work on numpy
  arrays; `to_tensor`/`from_tensor` wrap them in torch only when torch is importable.
- AUDIO is {"waveform": [B,C,T] float32 in -1..1, "sample_rate": int}.
- Video frames come out of an mp4 through PyAV when installed, else the ffmpeg command line.
"""

from __future__ import annotations

import io
import json
import shutil
import struct
import subprocess
import wave
import zlib
from typing import Any, Dict, Optional, Tuple

import numpy as np

PNG_SIG = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------- torch (optional)

def _torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError:
        return None


def to_tensor(a: np.ndarray):
    """A numpy array as a torch tensor when torch is available (it is inside ComfyUI), else unchanged."""
    t = _torch()
    return t.from_numpy(np.ascontiguousarray(a)) if t is not None else a


def from_tensor(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().float().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------- images

def frames_to_image(frames: np.ndarray) -> np.ndarray:
    """uint8 [N,H,W,C] or [H,W,C] (C 1, 3 or 4) to IMAGE float32 [N,H,W,3] in 0..1."""
    a = np.asarray(frames)
    if a.ndim == 3:
        a = a[None]
    if a.ndim != 4:
        raise ValueError(f"expected [N,H,W,C] frames, got shape {a.shape}")
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    elif a.shape[-1] == 4:
        a = a[..., :3]
    elif a.shape[-1] != 3:
        raise ValueError(f"expected 1, 3 or 4 channels, got {a.shape[-1]}")
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    return np.clip(a.astype(np.float32), 0.0, 1.0)


def image_to_uint8(image: Any) -> np.ndarray:
    """IMAGE [B,H,W,C] (tensor or array) float 0..1 to uint8 [B,H,W,3]."""
    a = from_tensor(image)
    if a.ndim == 3:
        a = a[None]
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    a = a[..., :3]
    return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def decode_png(data: bytes) -> np.ndarray:
    """PNG bytes to uint8 [H,W,C]. Uses Pillow when present (ComfyUI ships it), else a stdlib decoder
    for 8-bit non-interlaced grey/RGB/RGBA PNGs (what Shutter's adapters write)."""
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(data))
        im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
        return np.asarray(im)
    except ImportError:
        pass
    if data[:8] != PNG_SIG:
        raise ValueError("not a PNG")
    pos, idat, hdr = 8, b"", None
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", chunk)
        elif kind == b"IDAT":
            idat += chunk
        pos += 12 + n
    if hdr is None:
        raise ValueError("PNG without IHDR")
    w, h, depth, ctype, _, _, interlace = hdr
    ch = {0: 1, 2: 3, 6: 4}.get(ctype)
    if depth != 8 or ch is None or interlace:
        raise ValueError("only 8-bit non-interlaced grey/RGB/RGBA PNGs decode without Pillow")
    raw = zlib.decompress(idat)
    stride = w * ch
    out = np.zeros((h, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int32)
    for y in range(h):
        f = raw[y * (stride + 1)]
        line = np.frombuffer(raw, np.uint8, stride, y * (stride + 1) + 1).astype(np.int32)
        cur = np.zeros(stride, dtype=np.int32)
        for x in range(stride):
            a = cur[x - ch] if x >= ch else 0
            b = prev[x]
            c = prev[x - ch] if x >= ch else 0
            if f == 0:
                p = 0
            elif f == 1:
                p = a
            elif f == 2:
                p = b
            elif f == 3:
                p = (a + b) // 2
            else:
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                p = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
            cur[x] = (line[x] + p) & 0xFF
        out[y] = cur
        prev = cur
    return out.reshape(h, w, ch)


def encode_png(rgb: np.ndarray) -> bytes:
    """uint8 [H,W,3] to PNG bytes (stdlib only)."""
    a = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, c = a.shape
    ctype = {1: 0, 3: 2, 4: 6}[c]
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    return (PNG_SIG + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def png_to_image(data: bytes) -> np.ndarray:
    return frames_to_image(decode_png(data))


def image_to_png(image: Any, index: int = 0) -> bytes:
    """One frame of an IMAGE batch as PNG bytes (what /v1/files receives for an image input)."""
    return encode_png(image_to_uint8(image)[index])


# ---------------------------------------------------------------- audio

def audio_to_wav(audio: Dict[str, Any]) -> bytes:
    """ComfyUI AUDIO {"waveform": [B,C,T], "sample_rate"} to 16-bit PCM WAV (first batch item)."""
    wf = from_tensor(audio["waveform"])
    if wf.ndim == 3:
        wf = wf[0]
    if wf.ndim == 1:
        wf = wf[None]
    sr = int(audio["sample_rate"])
    pcm = (np.clip(wf.T, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(wf.shape[0])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def wav_to_audio(data: bytes) -> Dict[str, Any]:
    """16-bit PCM WAV to AUDIO with waveform [1,C,T] float32."""
    with wave.open(io.BytesIO(data), "rb") as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        frames = w.readframes(n)
    if sw != 2:
        raise ValueError("only 16-bit WAV is supported")
    a = np.frombuffer(frames, "<i2").astype(np.float32).reshape(-1, ch).T / 32768.0
    return {"waveform": a[None], "sample_rate": sr}


# ---------------------------------------------------------------- video

def _ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def probe_video(path: str) -> Tuple[int, int, float, bool]:
    """(width, height, fps, has_audio) via ffprobe."""
    fp = shutil.which("ffprobe")
    if fp is None:
        raise RuntimeError("reading video needs PyAV or ffmpeg/ffprobe on PATH")
    out = subprocess.run([fp, "-v", "error", "-show_streams", "-of", "json", path],
                         capture_output=True, check=True, text=True).stdout
    streams = json.loads(out).get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise ValueError(f"{path} has no video stream")
    num, _, den = str(v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return int(v["width"]), int(v["height"]), fps, any(s.get("codec_type") == "audio" for s in streams)


def read_video(path: str) -> Tuple[np.ndarray, float, Optional[Dict[str, Any]]]:
    """An mp4 as (IMAGE frames float32 [N,H,W,3], fps, AUDIO or None)."""
    try:
        import av  # type: ignore
    except ImportError:
        av = None
    if av is not None:
        with av.open(path) as c:
            vs = c.streams.video[0]
            fps = float(vs.average_rate or 0)
            frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        audio = None
        with av.open(path) as c:
            if c.streams.audio:
                chunks, sr = [], c.streams.audio[0].rate
                for f in c.decode(audio=0):
                    nch = len(f.layout.channels)
                    a = f.to_ndarray()
                    if a.shape[0] != nch:          # packed layout: [1, T*C] interleaved
                        a = a.reshape(-1, nch).T
                    if a.dtype.kind == "i":
                        a = a.astype(np.float32) / float(np.iinfo(a.dtype).max + 1)
                    chunks.append(a.astype(np.float32))
                if chunks:
                    audio = {"waveform": np.concatenate(chunks, axis=1)[None], "sample_rate": int(sr)}
        return frames_to_image(np.stack(frames)), fps, audio
    ff = _ffmpeg()
    if ff is None:
        raise RuntimeError("reading video needs PyAV or ffmpeg on PATH")
    w, h, fps, has_audio = probe_video(path)
    raw = subprocess.run([ff, "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    audio = None
    if has_audio:
        wav = subprocess.run([ff, "-v", "error", "-i", path, "-vn", "-acodec", "pcm_s16le", "-f", "wav", "-"],
                             capture_output=True, check=True).stdout
        audio = wav_to_audio(_fix_streamed_wav(wav))
    return frames_to_image(frames), fps, audio


def _fix_streamed_wav(data: bytes) -> bytes:
    """ffmpeg writing WAV to a pipe leaves the RIFF/data sizes unset; patch them from the length."""
    if data[:4] != b"RIFF":
        return data
    i = data.find(b"data", 12)
    if i < 0:
        return data
    out = bytearray(data)
    out[4:8] = struct.pack("<I", len(data) - 8)
    out[i + 4:i + 8] = struct.pack("<I", len(data) - i - 8)
    return bytes(out)
