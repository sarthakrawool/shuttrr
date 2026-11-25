"""Uploaded input files (POST /v1/files) and job outputs (GET /v1/jobs/{id}/output).

An upload is stored at `<work_dir>/uploads/<id>/<name>`: inside the engine's work area, so a client
on another filesystem (a container, another Mac) never needs to share a path with the engine. A
request's `inputs` may name an upload by its `path` or by its bare `id`. The Rust daemon uses the same
layout and rules (daemon/src/files.rs).
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any, Optional

__all__ = ["UPLOAD_ID_RE", "uploads_dir", "safe_name", "save_upload", "resolve_input", "content_type_for",
           "UploadError"]

UPLOAD_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
    ".gif": "image/gif", ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".m4a": "audio/mp4",
    ".json": "application/json", ".txt": "text/plain; charset=utf-8",
}


class UploadError(ValueError):
    """Bad upload (empty, too large, bad name)."""


def uploads_dir(work_dir: str) -> str:
    return os.path.join(work_dir, "uploads")


def safe_name(name: Optional[str], content_type: Optional[str] = None) -> str:
    """A plain file name: the last path component, only [A-Za-z0-9._-], at most 100 characters,
    never empty or hidden. The extension is guessed from the content type when missing."""
    base = os.path.basename((name or "").replace("\\", "/")).strip()
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).lstrip(".")[:100]
    if not base:
        base = "upload"
    if "." not in base and content_type:
        ct = content_type.split(";")[0].strip().lower()
        for ext, t in _TYPES.items():
            if t.split(";")[0] == ct:
                base += ext
                break
    return base


def content_type_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in _TYPES:
        return _TYPES[ext]
    guess, _ = mimetypes.guess_type(path)
    return guess or "application/octet-stream"


def save_upload(work_dir: str, name: Optional[str], data: bytes, content_type: Optional[str] = None,
                max_bytes: Optional[int] = None) -> dict[str, Any]:
    """Store one upload; returns the POST /v1/files response."""
    if not data:
        raise UploadError("the upload is empty")
    if max_bytes is not None and len(data) > max_bytes:
        raise UploadError(f"the upload is {len(data)} bytes, more than the {max_bytes} byte limit (max_upload_mb)")
    fid = secrets.token_hex(8)
    fname = safe_name(name, content_type)
    d = Path(uploads_dir(work_dir)) / fid
    d.mkdir(parents=True, exist_ok=True)
    path = d / fname
    tmp = d / (fname + ".part")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    ct = (content_type or "").split(";")[0].strip()
    if not ct or ct == "application/octet-stream":
        ct = content_type_for(fname)
    return {"id": fid, "path": str(path.resolve()), "name": fname, "size": len(data),
            "content_type": ct, "sha256": hashlib.sha256(data).hexdigest(), "created_at": time.time()}


def resolve_input(work_dir: str, value: str) -> str:
    """An `inputs` value -> an absolute path: a bare upload id is looked up under the uploads
    directory; anything else is a path (`~` expanded, made absolute)."""
    if UPLOAD_ID_RE.match(value):
        d = Path(uploads_dir(work_dir)) / value
        if d.is_dir():
            files = sorted(p for p in d.iterdir() if p.is_file() and not p.name.endswith(".part"))
            if files:
                return str(files[0].resolve())
    return os.path.abspath(os.path.expanduser(value))
