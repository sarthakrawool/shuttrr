"""OpenAI Images compatible shim: POST /v1/images/generations and /v1/images/edits.

Each image is one native Shutter job (seed, seed + 1, ...), so the queue, the memory guard and the
modes apply as for any request. Image-kind models only. The request waits for its jobs; when the client
disconnects (OpenAI's API has no cancel) the jobs are cancelled. Errors use OpenAI's shape
`{"error": {"message", "type", "param", "code"}}`. The Rust daemon serves the same shapes
(daemon/src/images.rs); docs/API.md lists what is supported.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import re
import time
from typing import Any, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from shutter.core.files import UploadError
from shutter.core.stats import client_label
from shutter.types import JobRequest, JobState

__all__ = ["routes", "parse_size", "MAX_N"]

MAX_N = 10
DEFAULT_CLIENT = "openai-images"
POLL_S = 0.05
_SIZE_RE = re.compile(r"^\s*(\d{1,5})\s*[xX]\s*(\d{1,5})\s*$")
_DATA_URL_RE = re.compile(r"^data:([\w/+.-]+)?(;[\w=.-]+)*;base64,(.*)$", re.DOTALL)


class ShimError(Exception):
    def __init__(self, status: int, message: str, param: Optional[str] = None, code: Optional[str] = None,
                 kind: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.message, self.param, self.code, self.kind = status, message, param, code, kind

    def response(self) -> JSONResponse:
        return JSONResponse({"error": {"message": self.message, "type": self.kind, "param": self.param,
                                       "code": self.code}}, status_code=self.status)


def parse_size(v: Any) -> tuple[Optional[int], Optional[int]]:
    """"WxH" -> (W, H); None / "" / "auto" -> (None, None)."""
    if v is None or v == "" or v == "auto":
        return None, None
    m = _SIZE_RE.match(v) if isinstance(v, str) else None
    if not m or int(m.group(1)) <= 0 or int(m.group(2)) <= 0:
        raise ShimError(400, f"size must be \"WIDTHxHEIGHT\" (e.g. 1024x1024) or \"auto\", got {v!r}", "size")
    return int(m.group(1)), int(m.group(2))


def _int(v: Any, name: str, lo: int, hi: Optional[int] = None) -> int:
    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
        v = int(v)
    if isinstance(v, bool) or not isinstance(v, int) or v < lo or (hi is not None and v > hi):
        rng = f"{lo}..{hi}" if hi is not None else f">= {lo}"
        raise ShimError(400, f"{name} must be an integer {rng}", name)
    return v


def _num(v: Any, name: str) -> float:
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            pass
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ShimError(400, f"{name} must be a number", name)
    return float(v)


def _image_models(engine: Any) -> list[str]:
    return [m["id"] for m in engine.registry.list() if m.get("kind") == "image"]


def _common(engine: Any, body: dict[str, Any]) -> dict[str, Any]:
    """Validate the fields generations and edits share; returns the job template + options."""
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ShimError(400, "prompt is required", "prompt")
    if body.get("stream") in (True, "true", "1"):
        raise ShimError(400, "stream is not supported by Shuttrr yet; send stream=false", "stream")
    fmt = body.get("output_format") or "png"
    if fmt != "png":
        raise ShimError(400, f"output_format {fmt!r} is not supported; Shuttrr returns png", "output_format")
    rf = body.get("response_format") or "b64_json"
    if rf not in ("b64_json", "url"):
        raise ShimError(400, "response_format must be b64_json or url", "response_format")
    n = _int(body.get("n", 1), "n", 1, MAX_N)
    w, h = parse_size(body.get("size"))
    models = {m["id"]: m for m in engine.registry.list()}
    model = body.get("model")
    if model in (None, ""):
        imgs = _image_models(engine)
        if not imgs:
            raise ShimError(400, "no image model is configured (GET /v1/models lists kind \"image\")", "model")
        model = imgs[0]
    if not isinstance(model, str) or model not in models:
        raise ShimError(404, f"unknown model {model!r}; image models: {', '.join(_image_models(engine)) or 'none'}",
                        "model", "model_not_found")
    kind = models[model].get("kind") or "unknown"
    if kind != "image":
        raise ShimError(400, f"model {model!r} is a {kind} model; the Images API serves image models only", "model")
    params = body.get("params") or {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except ValueError:
            raise ShimError(400, "params must be a JSON object", "params")
    if not isinstance(params, dict):
        raise ShimError(400, "params must be a JSON object", "params")
    tmpl: dict[str, Any] = {"model": model, "prompt": prompt, "params": dict(params)}
    if w:
        tmpl["width"], tmpl["height"] = w, h
    if body.get("negative_prompt") not in (None, ""):
        tmpl["negative_prompt"] = str(body["negative_prompt"])
    if body.get("steps") not in (None, ""):
        tmpl["steps"] = _int(body["steps"], "steps", 1)
    if body.get("guidance") not in (None, ""):
        tmpl["guidance"] = _num(body["guidance"], "guidance")
    seed = _int(body["seed"], "seed", 0) if body.get("seed") not in (None, "") else int(time.time_ns() % 2**31)
    return {"tmpl": tmpl, "n": n, "seed": seed, "response_format": rf, "size": f"{w}x{h}" if w else None}


def _decode_data_url(v: Any, param: str) -> tuple[bytes, Optional[str]]:
    if not isinstance(v, str):
        raise ShimError(400, f"{param} must be a data: URL (base64)", param)
    m = _DATA_URL_RE.match(v.strip())
    if not m:
        raise ShimError(400, f"{param}: only data: URLs (base64) are accepted; Shuttrr does not fetch URLs", param)
    try:
        return base64.b64decode(m.group(3), validate=False), m.group(1)
    except (binascii.Error, ValueError):
        raise ShimError(400, f"{param}: the base64 data is invalid", param)


async def _run(engine: Any, request: Request, opts: dict[str, Any], inputs: dict[str, str]) -> Response:
    client = client_label(None, request.headers.get("x-shutter-client"),
                          request.headers.get("user-agent")) or DEFAULT_CLIENT
    jobs = []
    try:
        for i in range(opts["n"]):
            d = dict(opts["tmpl"], seed=opts["seed"] + i, inputs=dict(inputs))
            d["params"] = dict(d["params"])
            jobs.append(await asyncio.to_thread(engine.submit, JobRequest.from_dict(d), client))
    except Exception as e:
        for j in jobs:
            engine.cancel(j.id)
        raise ShimError(400, str(e))
    ids = [j.id for j in jobs]
    deadline = time.monotonic() + float(getattr(engine.config, "images_timeout_s", 3600.0))
    done: dict[str, Any] = {}
    try:
        while len(done) < len(ids):
            for jid in ids:
                if jid in done:
                    continue
                j = engine.get(jid)
                if j is not None and j.state in (JobState.DONE, JobState.FAILED, JobState.CANCELLED):
                    done[jid] = j
                    if j.state != JobState.DONE:
                        for other in ids:
                            if other not in done:
                                engine.cancel(other)
                        detail = (j.result or {}).get("error_detail")
                        raise ShimError(500, f"job {jid} {j.state.value}: {j.message}"
                                        + (f" ({detail})" if detail else ""), None,
                                        "job_" + j.state.value, "server_error")
            if len(done) == len(ids):
                break
            if time.monotonic() > deadline:
                raise ShimError(504, f"the images were not ready within {getattr(engine.config, 'images_timeout_s', 3600)} s"
                                f" (images_timeout_s); jobs {', '.join(ids)} were cancelled", None, "timeout",
                                "server_error")
            if await request.is_disconnected():
                raise asyncio.CancelledError()
            await asyncio.sleep(POLL_S)
    except BaseException:
        for jid in ids:
            if jid not in done:
                engine.cancel(jid)
        raise
    data = []
    base = str(request.base_url).rstrip("/")
    for jid in ids:
        if opts["response_format"] == "url":
            data.append({"url": f"{base}/v1/jobs/{jid}/output", "revised_prompt": opts["tmpl"]["prompt"]})
            continue
        status, path = engine.output_file(jid)
        if status != 200:
            raise ShimError(500, path, None, "output_missing", "server_error")
        with open(path, "rb") as f:
            data.append({"b64_json": base64.b64encode(f.read()).decode("ascii"),
                         "revised_prompt": opts["tmpl"]["prompt"]})
    out: dict[str, Any] = {"created": int(time.time()), "data": data, "output_format": "png",
                           "shutter": {"model": opts["tmpl"]["model"], "jobs": ids}}
    if opts["size"]:
        out["size"] = opts["size"]
    return JSONResponse(out)


def routes(engine: Any) -> list[Route]:
    async def generations(request: Request) -> Response:
        try:
            try:
                body = json.loads(await request.body() or b"null")
            except (ValueError, UnicodeDecodeError):
                raise ShimError(400, "body must be JSON")
            if not isinstance(body, dict):
                raise ShimError(400, "body must be a JSON object")
            return await _run(engine, request, _common(engine, body), {})
        except ShimError as e:
            return e.response()

    async def edits(request: Request) -> Response:
        try:
            images: list[tuple[bytes, Optional[str], Optional[str]]] = []  # (data, name, content type)
            mask = None
            file_ids: list[str] = []
            ctype = request.headers.get("content-type", "").lower()
            if ctype.startswith("multipart/form-data"):
                form = await request.form()
                body: dict[str, Any] = {}
                for k, v in form.multi_items():
                    if isinstance(v, str):
                        body[k] = v
                    elif k in ("image", "image[]", "images", "images[]"):
                        images.append((await v.read(), v.filename, v.content_type))
                    elif k == "mask":
                        mask = (await v.read(), v.filename, v.content_type)
            else:
                try:
                    body = json.loads(await request.body() or b"null")
                except (ValueError, UnicodeDecodeError):
                    raise ShimError(400, "body must be JSON or multipart/form-data")
                if not isinstance(body, dict):
                    raise ShimError(400, "body must be a JSON object")
                refs = body.get("images")
                if refs is None and body.get("image") is not None:
                    refs = [body["image"]]
                for i, ref in enumerate(refs or []):
                    if isinstance(ref, str):
                        ref = {"image_url": ref}
                    if not isinstance(ref, dict):
                        raise ShimError(400, "images must be a list of {\"image_url\": data URL} or {\"file_id\"}", "images")
                    if ref.get("file_id"):
                        file_ids.append(str(ref["file_id"]))
                    else:
                        data, ct = _decode_data_url(ref.get("image_url"), f"images[{i}].image_url")
                        images.append((data, None, ct))
                m = body.get("mask")
                if isinstance(m, str):
                    m = {"image_url": m}
                if isinstance(m, dict):
                    if m.get("file_id"):
                        file_ids.append("mask:" + str(m["file_id"]))
                    else:
                        data, ct = _decode_data_url(m.get("image_url"), "mask.image_url")
                        mask = (data, None, ct)
            opts = _common(engine, body)
            if not images and not [f for f in file_ids if not f.startswith("mask:")]:
                raise ShimError(400, "image is required (multipart file \"image\", or JSON images[])", "image")
            roles: list[str] = []
            inputs: dict[str, str] = {}
            try:
                for data, name, ct in images:
                    info = await asyncio.to_thread(engine.save_upload, name or "image.png", data, ct)
                    role = "image" if not roles else f"image_{len(roles) + 1}"
                    roles.append(role)
                    inputs[role] = info["path"]
                if mask is not None:
                    inputs["mask"] = (await asyncio.to_thread(engine.save_upload, mask[1] or "mask.png",
                                                              mask[0], mask[2]))["path"]
            except UploadError as e:
                raise ShimError(413 if "limit" in str(e) else 400, str(e), "image")
            for fid in file_ids:
                if fid.startswith("mask:"):
                    inputs["mask"] = fid[5:]
                else:
                    role = "image" if not roles else f"image_{len(roles) + 1}"
                    roles.append(role)
                    inputs[role] = fid
            return await _run(engine, request, opts, inputs)
        except ShimError as e:
            return e.response()

    return [Route("/v1/images/generations", generations, methods=["POST"]),
            Route("/v1/images/edits", edits, methods=["POST"])]
