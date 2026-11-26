"""A small, dependency-free HTTP client for Shutter's native API (docs/API.md).

Standard library only, so the pack adds nothing to ComfyUI's environment and the client can be
tested without ComfyUI. Every call is blocking; the node runs them in a thread.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

PACK_VERSION = "0.2.11"
DEFAULT_URL = os.environ.get("SHUTTER_URL", "http://127.0.0.1:8484")
CLIENT_NAME = f"comfyui-shutter/{PACK_VERSION}"


class ShutterError(RuntimeError):
    """An HTTP error from Shutter (status and message), or Shutter being unreachable (status 0)."""

    def __init__(self, status: int, message: str):
        super().__init__(f"Shutter {status}: {message}" if status else message)
        self.status = status
        self.message = message


class ShutterClient:
    def __init__(self, url: Optional[str] = None, token: Optional[str] = None, timeout: float = 30.0,
                 client_name: str = CLIENT_NAME):
        self.url = (url or DEFAULT_URL).rstrip("/")
        self.token = token if token is not None else os.environ.get("SHUTTER_TOKEN")
        self.timeout = timeout
        self.client_name = client_name

    # ------------------------------------------------------------------ transport

    def _request(self, method: str, path: str, body: Optional[bytes] = None,
                 headers: Optional[Dict[str, str]] = None, raw: bool = False) -> Any:
        h = {"X-Shutter-Client": self.client_name, "Accept": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        h.update(headers or {})
        req = urllib.request.Request(self.url + path, data=body, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = r.read()
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace")
            try:
                v = json.loads(text)
                err = v.get("error", text)
                msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            except ValueError:
                msg = text or e.reason
            raise ShutterError(e.code, msg) from None
        except (urllib.error.URLError, OSError) as e:
            raise ShutterError(0, f"cannot reach Shutter at {self.url}: {getattr(e, 'reason', e)}") from None
        if raw:
            return data
        return json.loads(data) if data else None

    def _json(self, method: str, path: str, payload: Any = None) -> Any:
        body = None if payload is None else json.dumps(payload).encode()
        return self._request(method, path, body, {"Content-Type": "application/json"} if body else None)

    # ------------------------------------------------------------------ API

    def system(self) -> Dict[str, Any]:
        return self._json("GET", "/v1/system")

    def models(self) -> List[Dict[str, Any]]:
        return self._json("GET", "/v1/models")["models"]

    def memory(self, model: Optional[str] = None, **request: Any) -> Dict[str, Any]:
        """GET /v1/memory; with `model` (and width/height/frames/steps) adds the `model` fit block."""
        q = {k: v for k, v in dict(model=model, **request).items() if v is not None}
        return self._json("GET", "/v1/memory" + ("?" + urllib.parse.urlencode(q) if q else ""))

    def upload(self, data: bytes, name: str, content_type: str = "application/octet-stream") -> Dict[str, Any]:
        """POST /v1/files as multipart; returns {id, path, name, size, content_type, sha256, created_at}."""
        boundary = "shutter-" + uuid.uuid4().hex
        safe = name.replace('"', "_").replace("\r", "_").replace("\n", "_")
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{safe}\"\r\n"
                f"Content-Type: {content_type}\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
        return self._request("POST", "/v1/files", body,
                             {"Content-Type": f"multipart/form-data; boundary={boundary}"})

    def submit(self, request: Dict[str, Any]) -> Dict[str, Any]:
        body = dict(request)
        body.setdefault("client", self.client_name)
        return self._json("POST", "/v1/jobs", body)

    def job(self, job_id: str) -> Dict[str, Any]:
        return self._json("GET", f"/v1/jobs/{urllib.parse.quote(job_id)}")

    def cancel(self, job_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self._json("DELETE", f"/v1/jobs/{urllib.parse.quote(job_id)}")
        except ShutterError as e:
            if e.status in (404, 409):   # gone or already finished: nothing to cancel
                return None
            raise

    def output(self, job_id: str) -> bytes:
        return self._request("GET", f"/v1/jobs/{urllib.parse.quote(job_id)}/output", raw=True)

    def release(self, key: Optional[str] = None) -> Any:
        """POST /v1/pool/release: free idle model components (all of them when key is None)."""
        return self._json("POST", "/v1/pool/release", {"key": key})
