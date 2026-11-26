"""HTTP API (Starlette). Binds to 127.0.0.1 by default.

No auth on loopback. A non-loopback bind is refused unless `api_key` is configured; then
requests from non-loopback clients must send `Authorization: Bearer <api_key>` (or
`?key=<api_key>`, which the dashboard and EventSource use).
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from shutter.core.engine import Engine, LoadRefused, LoadUnsupported, SubmitError, UnloadRefused
from shutter.core.files import UploadError, content_type_for
from shutter.core.stats import client_label
from shutter.server import openai_images
from shutter.types import LEGACY_INPUT_KEYS, JobRequest, JobState

__all__ = ["create_app", "check_bind", "is_loopback", "serve"]

DASHBOARD = Path(__file__).with_name("dashboard.html")
STATIC = Path(__file__).with_name("static")  # dashboard CSS + fonts (public, no auth)
HEARTBEAT_S = 15.0
RECENT_EVENTS = 50


def is_loopback(host: Optional[str]) -> bool:
    if not host:
        return False
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def check_bind(host: str, api_key: Optional[str]) -> None:
    """Refuse a non-loopback bind without an api key."""
    if not is_loopback(host) and not api_key:
        raise ValueError(
            f"refusing to listen on {host}: Shuttrr has no auth on loopback. Set \"api_key\" in "
            f"config.json to listen on other interfaces (clients then send Authorization: Bearer).")


def _err(status: int, msg: str) -> JSONResponse:
    return JSONResponse({"error": msg, "detail": msg}, status_code=status)


_INT_FIELDS = ("seed", "priority")
_POS_INT_FIELDS = ("width", "height", "frames", "steps")
_NUM_FIELDS = ("duration_s", "fps", "guidance")
_STR_FIELDS = ("prompt", "negative_prompt", "output", "client") + LEGACY_INPUT_KEYS


def validate_submit(data: Any) -> Optional[str]:
    """Type checks of a POST /v1/jobs body (the engine checks values). None when fine. The daemon
    gives the same messages (daemon/src/engine.rs)."""
    if not isinstance(data, dict):
        return "body must be a JSON object with at least \"model\""
    if not data.get("model") or not isinstance(data.get("model"), str):
        return "\"model\" is required (a model id from GET /v1/models)"
    if "framing" in data and data["framing"] is not None and not isinstance(data["framing"], str):
        return "framing must be a string (it is passed to the model as params.framing)"
    if "params" in data and not isinstance(data["params"], dict):
        return "params must be an object"
    if "inputs" in data and data["inputs"] is not None:
        if not isinstance(data["inputs"], dict):
            return "inputs must be an object: {\"image\": path, \"audio\": path, ...}"
        for k, v in data["inputs"].items():
            if v is not None and not isinstance(v, str):
                return f"inputs.{k} must be a file path or an upload id"
    for k in _NUM_FIELDS:
        if data.get(k) is not None and (isinstance(data[k], bool) or not isinstance(data[k], (int, float))):
            return f"{k} must be a number"
    for k in _INT_FIELDS:
        if k in data and (isinstance(data[k], bool) or not isinstance(data[k], int)):
            return f"{k} must be an integer"
    for k in _POS_INT_FIELDS:
        if data.get(k) is not None and (isinstance(data[k], bool) or not isinstance(data[k], int)):
            return f"{k} must be an integer"
    for k in _STR_FIELDS:
        if data.get(k) is not None and not isinstance(data[k], str):
            return f"{k} must be a string"
    return None


def _job_json(job: Any) -> dict[str, Any]:
    # eta_s comes from the engine (measured step times + decode estimate, None until the first step is done;
    # core/eta.py). Before 0.2.11 it was elapsed x (1 - progress) / progress, which ran far too low.
    return job.to_dict()


class _AuthMiddleware:
    """Pure ASGI (keeps SSE streaming and disconnect detection intact)."""

    def __init__(self, app: Any, api_key: Optional[str], seen: Any = None):
        self.app = app
        self.api_key = api_key
        self.seen = seen  # called with the X-Shutter-Client header of every API request

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if self.api_key and scope["type"] == "http" and not scope["path"].startswith("/static/"):
            request = Request(scope)
            client = request.client.host if request.client else None
            if not is_loopback(client):
                auth = request.headers.get("authorization", "")
                token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
                token = token or request.query_params.get("key", "")
                if not hmac.compare_digest(token.encode(), self.api_key.encode()):
                    resp = _err(401, "missing or wrong API key (Authorization: Bearer ...)")
                    await resp(scope, receive, send)
                    return
        if self.seen is not None and scope["type"] == "http":
            for k, v in scope.get("headers") or []:
                if k == b"x-shutter-client":
                    try:
                        self.seen(client_label(v.decode("utf-8", "replace")))
                    except Exception:
                        pass
                    break
        await self.app(scope, receive, send)


def create_app(engine: Engine, api_key: Optional[str] = None) -> Starlette:
    if api_key is None:
        api_key = getattr(engine.config, "api_key", None)

    async def system(request: Request) -> Response:
        return JSONResponse(await asyncio.to_thread(engine.system))

    async def models(request: Request) -> Response:
        return JSONResponse({"models": await asyncio.to_thread(engine.models)})

    async def load_model(request: Request) -> Response:
        mid = request.path_params["model_id"]
        wait = False
        body = await request.body()
        if body.strip():
            try:
                data = json.loads(body)
            except ValueError:
                return _err(400, "body must be JSON")
            if isinstance(data, dict):
                wait = bool(data.get("wait", False))
        try:
            job = await asyncio.to_thread(engine.load_model, mid, _client_of(request), wait)
        except LoadUnsupported as e:
            return _err(409, str(e))
        except SubmitError as e:
            return _err(404, str(e))
        except LoadRefused as e:
            return JSONResponse({"error": str(e), "refused": e.info}, status_code=409)
        rows = await asyncio.to_thread(engine.models)
        row = next((r for r in rows if r["id"] == mid), None)
        return JSONResponse({"model": row, "request": job.to_dict()}, status_code=202)

    async def unload_model(request: Request) -> Response:
        mid = request.path_params["model_id"]
        try:
            released = await asyncio.to_thread(engine.unload_model, mid)
        except SubmitError as e:
            return _err(404, str(e))
        except UnloadRefused as e:
            return _err(409, str(e))
        rows = await asyncio.to_thread(engine.models)
        row = next((r for r in rows if r["id"] == mid), None)
        return JSONResponse({"model": row, "released": released})

    def _client_of(request: Request, body_value: Any = None) -> Optional[str]:
        return client_label(body_value, request.headers.get("x-shutter-client"),
                            request.headers.get("user-agent"))

    async def submit(request: Request) -> Response:
        try:
            body = await request.body()
            data = json.loads(body or b"null")
        except (ValueError, UnicodeDecodeError):
            return _err(400, "body must be JSON")
        bad = validate_submit(data)
        if bad:
            return _err(400, bad)
        client = _client_of(request, data.pop("client", None))
        try:
            req = JobRequest.from_dict(data)
            job = await asyncio.to_thread(engine.submit, req, client)
        except SubmitError as e:
            return _err(400, str(e))
        except (TypeError, ValueError) as e:
            return _err(400, f"bad job request: {e}")
        return JSONResponse(_job_json(job), status_code=201)

    async def job_output(request: Request) -> Response:
        status, val = await asyncio.to_thread(engine.output_file, request.path_params["job_id"])
        if status != 200:
            return _err(status, val)
        return FileResponse(val, media_type=content_type_for(val), filename=os.path.basename(val))

    async def upload(request: Request) -> Response:
        ctype = request.headers.get("content-type", "")
        try:
            if ctype.lower().startswith("multipart/form-data"):
                form = await request.form()
                part = form.get("file")
                if part is None or isinstance(part, str):
                    part = next((v for _, v in form.multi_items() if not isinstance(v, str)), None)
                if part is None:
                    return _err(400, "multipart upload needs a file part (field name \"file\")")
                data = await part.read()
                name, ptype = part.filename, part.content_type
            else:
                data = await request.body()
                name = request.query_params.get("name") or request.headers.get("x-filename")
                ptype = ctype or None
        except Exception as e:  # malformed multipart, part too large
            return _err(400, f"bad upload: {e}")
        try:
            info = await asyncio.to_thread(engine.save_upload, name, data, ptype)
        except UploadError as e:
            return _err(413 if "limit" in str(e) else 400, str(e))
        return JSONResponse(info, status_code=201)

    async def memory(request: Request) -> Response:
        q = request.query_params
        extra: dict[str, Any] = {}
        for k in ("width", "height", "frames", "steps"):
            if q.get(k):
                try:
                    extra[k] = int(q[k])
                except ValueError:
                    return _err(400, f"{k} must be an integer")
        return JSONResponse(await asyncio.to_thread(engine.memory, q.get("model") or None, extra))

    async def clients(request: Request) -> Response:
        return JSONResponse(engine.clients.list())

    async def list_jobs(request: Request) -> Response:
        q = request.query_params
        try:
            limit = int(q.get("limit", "200"))
        except ValueError:
            return _err(400, "limit must be an integer")
        states = None
        if q.get("state"):
            try:
                states = [JobState(s.strip()) for s in q["state"].split(",") if s.strip()]
            except ValueError:
                return _err(400, "state must be one of: " + ", ".join(s.value for s in JobState))
        jobs = await asyncio.to_thread(engine.list, states, max(1, min(limit, 1000)))
        return JSONResponse({"jobs": [_job_json(j) for j in jobs]})

    async def get_job(request: Request) -> Response:
        job = await asyncio.to_thread(engine.get, request.path_params["job_id"])
        if job is None:
            return _err(404, f"no job {request.path_params['job_id']}")
        return JSONResponse(_job_json(job))

    async def cancel_job(request: Request) -> Response:
        job = await asyncio.to_thread(engine.cancel, request.path_params["job_id"])
        if job is None:
            return _err(404, f"no job {request.path_params['job_id']}")
        return JSONResponse(_job_json(job))

    async def pause(request: Request) -> Response:
        engine.pause()
        return JSONResponse({"paused": True})

    async def resume(request: Request) -> Response:
        engine.resume()
        return JSONResponse({"paused": False})

    async def get_settings(request: Request) -> Response:
        return JSONResponse(await asyncio.to_thread(engine.settings))

    async def post_settings(request: Request) -> Response:
        body = await request.body()
        try:
            data = json.loads(body) if body.strip() else {}
        except (ValueError, UnicodeDecodeError):
            return _err(400, "body must be JSON")
        if not isinstance(data, dict):
            return _err(400, 'body must be a JSON object: {"mode"?, "efficiency"?}')
        mode, eff = data.get("mode"), data.get("efficiency")
        if mode is not None and not isinstance(mode, str):
            return _err(400, "mode must be one of: fast, low_memory")
        if eff is not None and (isinstance(eff, bool) or not isinstance(eff, (int, float))):
            return _err(400, "efficiency must be a number from 0 to 1")
        try:
            return JSONResponse(await asyncio.to_thread(engine.update_settings, mode, eff))
        except SubmitError as e:
            return _err(400, str(e))
        except (OSError, ValueError) as e:
            return _err(500, f"could not save the settings: {e}")

    async def release(request: Request) -> Response:
        key = None
        body = await request.body()
        if body.strip():
            try:
                data = json.loads(body)
            except ValueError:
                return _err(400, "body must be JSON")
            if isinstance(data, dict):
                key = data.get("key")
            if key is not None and not isinstance(key, str):
                return _err(400, "key must be a string or null")
        released = await asyncio.to_thread(engine.release, key)
        return JSONResponse({"released": released})

    async def events(request: Request) -> Response:
        try:
            max_events = int(request.query_params.get("max_events", "0"))
            recent_n = int(request.query_params.get("recent", str(RECENT_EVENTS)))
        except ValueError:
            return _err(400, "max_events and recent must be integers")
        sub = engine.bus.subscribe(maxlen=500)
        recent = engine.bus.recent(recent_n)
        seen = {id(e) for e in recent}

        async def stream():
            sent = 0
            last_beat = time.monotonic()
            try:
                yield ": shutter events\n\n"
                for ev in recent:
                    yield f"data: {json.dumps(ev.to_dict())}\n\n"
                    sent += 1
                    if max_events and sent >= max_events:
                        return
                while True:
                    if await request.is_disconnected():
                        return
                    ev = await asyncio.to_thread(sub.get, 1.0)
                    if ev is not None:
                        if id(ev) in seen:
                            seen.discard(id(ev))
                            continue
                        yield f"data: {json.dumps(ev.to_dict())}\n\n"
                        sent += 1
                        if max_events and sent >= max_events:
                            return
                    if time.monotonic() - last_beat >= HEARTBEAT_S:
                        last_beat = time.monotonic()
                        yield ": heartbeat\n\n"
            except asyncio.CancelledError:
                return
            finally:
                sub.close()

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # -- model downloads (docs/INSTALL.md). The download is its own process; these only start, stop and read it.
    def _catalog() -> dict[str, Any]:
        from shutter.core import catalog as cat
        from shutter.core.config import Config

        rows = cat.catalog_rows(engine.config)
        known = getattr(engine.config, "models", {})
        if any((r["download"] or {}).get("state") == "done" and r["id"] not in known for r in rows):
            fresh = Config.load(engine.config.path)      # a download finished: pick the new model up
            engine.config.models = fresh.models
            engine.config.default_mode = fresh.default_mode
            engine.config.default_efficiency = fresh.default_efficiency
            rows = cat.catalog_rows(engine.config)
        return {"models": rows}

    async def get_catalog(request: Request) -> Response:
        return JSONResponse(await asyncio.to_thread(_catalog))

    async def start_download(request: Request) -> Response:
        import subprocess
        import sys

        from shutter.core import catalog as cat

        mid = request.path_params["model_id"]
        if mid not in cat.CATALOG:
            return _err(404, f"unknown model '{mid}'")
        total = cat.total_memory()
        fit = cat.classify_fit(mid, total) if total else {"level": "unknown"}
        if fit["level"] == "no_fit" and request.query_params.get("force") not in ("1", "true"):
            return _err(409, f"{cat.CATALOG[mid]['name']} does not fit this Mac. {fit['reason']}")
        st = cat.read_state(engine.config, mid)
        if not (st and st.get("state") in cat.RUNNING):      # downloading, or converting a Wan download
            env = dict(os.environ, SHUTTER_HOME=str(engine.config.home))
            log = open(os.path.join(engine.config.home, "download.log"), "ab")
            subprocess.Popen([sys.executable, "-m", "shutter.cli", "download", mid, "--quiet"], env=env,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
            for _ in range(20):                          # let it write its first progress line
                await asyncio.sleep(0.1)
                st = cat.read_state(engine.config, mid)
                if st and st.get("pid") and st.get("state") in cat.RUNNING + ("failed", "done"):
                    break
        return JSONResponse(await asyncio.to_thread(_catalog), status_code=202)

    async def stop_download(request: Request) -> Response:
        from shutter.core import catalog as cat

        mid = request.path_params["model_id"]
        if mid not in cat.CATALOG:
            return _err(404, f"unknown model '{mid}'")
        cat.cancel(mid, engine.config)
        await asyncio.sleep(0.3)
        return JSONResponse(await asyncio.to_thread(_catalog))

    # ---- request options, estimates and the person's own defaults (docs/REQUEST_OPTIONS.md)
    def _ro_call(fn, mid, *args):
        from shutter.core import request_options as ro
        try:
            return 200, fn(engine.config, mid, *args)
        except KeyError:
            return 404, f"unknown model '{mid}'"
        except ro.OptionsError as e:
            return 400, str(e)

    async def _json_body(request: Request) -> Any:
        body = await request.body()
        return json.loads(body) if body.strip() else {}

    async def get_request_options(request: Request) -> Response:
        from shutter.core import request_options as ro
        code, val = await asyncio.to_thread(_ro_call, ro.request_options, request.path_params["model_id"])
        return JSONResponse(val) if code == 200 else _err(code, val)

    async def post_estimate(request: Request) -> Response:
        from shutter.core import request_options as ro
        try:
            data = await _json_body(request)
        except ValueError:
            return _err(400, "body must be JSON")
        if not isinstance(data, dict):
            return _err(400, "body must be a JSON object")
        code, val = await asyncio.to_thread(_ro_call, ro.estimate, request.path_params["model_id"], data)
        return JSONResponse(val) if code == 200 else _err(code, val)

    def _apply_options(mid: str) -> None:
        """Give the running adapter the saved options (the config file is the source)."""
        saved = type(engine.config).load(engine.config.path).models.get(mid)
        if saved is not None:
            engine.config.models[mid] = saved
        try:
            ad = engine.registry._adapters.get(mid)
            if ad is not None:
                ad.options.clear()
                ad.options.update((saved or {}).get("options") or {})
        except Exception:
            pass

    async def put_defaults(request: Request) -> Response:
        from shutter.core import request_options as ro
        mid = request.path_params["model_id"]
        try:
            data = await _json_body(request)
        except ValueError:
            return _err(400, "body must be JSON")
        if not isinstance(data, dict):
            return _err(400, "body must be a JSON object")
        code, val = await asyncio.to_thread(_ro_call, ro.save_defaults, mid, data)
        if code != 200:
            return _err(code, val)
        await asyncio.to_thread(_apply_options, mid)
        code, val = await asyncio.to_thread(_ro_call, ro.request_options, mid)
        return JSONResponse(val) if code == 200 else _err(code, val)

    async def delete_defaults(request: Request) -> Response:
        from shutter.core import request_options as ro
        mid = request.path_params["model_id"]
        code, val = await asyncio.to_thread(_ro_call, ro.reset_defaults, mid)
        if code != 200:
            return _err(code, val)
        await asyncio.to_thread(_apply_options, mid)
        code, val = await asyncio.to_thread(_ro_call, ro.request_options, mid)
        return JSONResponse(val) if code == 200 else _err(code, val)

    async def dashboard(request: Request) -> Response:
        try:
            return HTMLResponse(DASHBOARD.read_text(encoding="utf-8"))
        except OSError:
            return HTMLResponse("<h1>Shuttrr</h1><p>dashboard.html is missing.</p>")

    routes = [
        Route("/", dashboard),
        Route("/v1/system", system),
        Route("/v1/models", models),
        Route("/v1/models/{model_id}/load", load_model, methods=["POST"]),
        Route("/v1/models/{model_id}/unload", unload_model, methods=["POST"]),
        Route("/v1/models/{model_id}/request-options", get_request_options, methods=["GET"]),
        Route("/v1/models/{model_id}/estimate", post_estimate, methods=["POST"]),
        Route("/v1/models/{model_id}/defaults", put_defaults, methods=["PUT"]),
        Route("/v1/models/{model_id}/defaults", delete_defaults, methods=["DELETE"]),
        Route("/v1/catalog", get_catalog, methods=["GET"]),
        Route("/v1/catalog/{model_id}/download", start_download, methods=["POST"]),
        Route("/v1/catalog/{model_id}/download", stop_download, methods=["DELETE"]),
        Route("/v1/settings", get_settings, methods=["GET"]),
        Route("/v1/settings", post_settings, methods=["POST"]),
        Route("/v1/jobs", list_jobs, methods=["GET"]),
        Route("/v1/jobs", submit, methods=["POST"]),
        Route("/v1/jobs/{job_id}", get_job, methods=["GET"]),
        Route("/v1/jobs/{job_id}", cancel_job, methods=["DELETE"]),
        Route("/v1/jobs/{job_id}/output", job_output, methods=["GET"]),
        Route("/v1/files", upload, methods=["POST"]),
        Route("/v1/memory", memory, methods=["GET"]),
        Route("/v1/clients", clients, methods=["GET"]),
        *openai_images.routes(engine),
        Route("/v1/queue/pause", pause, methods=["POST"]),
        Route("/v1/queue/resume", resume, methods=["POST"]),
        Route("/v1/pool/release", release, methods=["POST"]),
        Route("/v1/events", events),
        Mount("/static", app=StaticFiles(directory=STATIC, check_dir=False), name="static"),
    ]
    app = Starlette(routes=routes, middleware=[Middleware(_AuthMiddleware, api_key=api_key,
                                                                       seen=engine.clients.seen)])
    app.state.engine = engine
    return app


def serve(engine: Engine, host: Optional[str] = None, port: Optional[int] = None,
          log_level: str = "info") -> None:
    """Run the API with uvicorn in this thread (starts and stops the engine)."""
    import uvicorn

    host = host or engine.config.host
    port = int(port or engine.config.port)
    check_bind(host, engine.config.api_key)
    app = create_app(engine)
    engine.start()
    try:
        uvicorn.run(app, host=host, port=port, log_level=log_level)
    finally:
        engine.stop()
