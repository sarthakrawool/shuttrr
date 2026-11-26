# Shuttrr HTTP API

Both servers serve the same routes and JSON shapes:
- the Rust daemon (`shutterd`, `daemon/src/api.rs`)
- the Python reference server (`shutter serve`, `shutter/server/api.py`)

Both test suites check them (`daemon/tests/daemon.rs`, `tests/test_api*.py`, `tests/test_openai_images.py`).
The address defaults to `http://127.0.0.1:8484`.

Conventions:
- Sizes are bytes (integers). Times are Unix seconds (floats). Durations end in `_s`.
- Native errors are `{"error": "<message>", "detail": "<message>"}` with a 4xx/5xx status.
- The OpenAI shim uses OpenAI's error shape instead (see the last section).

## Auth and client names

- **Auth.** Loopback clients need no auth. A non-loopback bind is refused unless `api_key` is set in
  `config.json`. Remote clients then send `Authorization: Bearer <api_key>`, or `?key=<api_key>` (used by
  the dashboard and EventSource).
- **`X-Shutter-Client` header** (optional, free text, trimmed to 120 characters).
  - On any request, it refreshes that client's `last_seen` in `GET /v1/clients`.
  - On a submit, it names the job's `client` unless the body has its own `client`.
- **Client label precedence on submit:**
  1. the body's `client`
  2. the `X-Shutter-Client` header
  3. the first word of the `User-Agent`

## Requests (jobs)

### `POST /v1/jobs`: submit

```json
{
  "model": "longcat-avatar",           // required: an id from GET /v1/models
  "prompt": "a red fox in snow",
  "negative_prompt": "",
  "width": 832, "height": 480,          // positive integers or omitted (model default)
  "frames": 93, "duration_s": null, "fps": 25,
  "steps": 8, "guidance": 4.0,
  "seed": 42,                           // integer; random when omitted
  "inputs": {"image": "<path or upload id>", "audio": "<path or upload id>"},
  "output": null,                       // output path; default <outputs>/<id>.<ext>
  "priority": 0,
  "params": {"mode": "low_memory", "efficiency": 0.75},   // adapter-specific, passed through
  "client": "comfyui-shutter/0.2.5"     // optional, recorded on the job (not part of the request)
}
```

- **The request is a plain inference request.** The core fields are listed in the order shown, and
  `params` holds everything adapter-specific.
- **`params` mirror.** Before an estimate or a run, each adapter sees the core fields mirrored into
  `params`, unless `params` already sets them. The mirrored fields are `negative_prompt`, `frames`, `fps`,
  `steps` and `guidance`, and `width` + `height` become `size` = `"HxW"`. The stored request is not
  changed.
- **`inputs`** maps a role (`[A-Za-z0-9_-]{1,40}`) to a file.
  - The value is an absolute path on the server, or an upload's `id` / `path` from `POST /v1/files`.
  - An `id` is resolved to its path when the request is accepted.
  - A missing file returns `400 "<role> file not found: <path>"`.
- **Legacy keys are still accepted but never emitted:**
  - Top-level `image` / `audio` / `video` / `mask` move into `inputs`; an explicit `inputs` entry wins.
  - Top-level `framing` moves into `params.framing`, the reels add-on's setting; an explicit `params`
    entry wins.
  - Any other unknown top-level key moves into `params`.
- **Validation (400):**
  - `width`, `height`, `frames` and `steps` must be positive integers.
  - `fps`, `duration_s` and `guidance` must be numbers ≥ 0.
  - `seed` and `priority` must be integers; the text fields must be strings.
- **Response:** `201` with the job.

### Job JSON (`GET /v1/jobs/{id}`, `GET /v1/jobs`, `DELETE /v1/jobs/{id}`)

```json
{
  "id": "d98530c84d6c",
  "state": "queued | waiting_memory | running | done | failed | cancelled",
  "created_at": 1795589086.56, "started_at": 1795589086.56, "finished_at": 1795589086.57,
  "progress": 1.0,                       // 0..1
  "stage": "denoise 2/2",                // human-readable; step counters parse from it
  "message": "",                         // last status line or the error text
  "eta_s": 12.5,                         // only while running and progress > 1%
  "result": {"output": "/…/outputs/d98530c84d6c.mp4", "seconds": 0.002, "peak_bytes": 17179869184, "…": "…"},
  "node": "local",
  "client": "comfyui-shutter/0.2.5",     // null when no client was named
  "request": { "model": "…", "prompt": "…", "negative_prompt": "", "width": null, "height": null,
               "frames": null, "duration_s": null, "fps": null, "steps": 2, "guidance": null,
               "seed": 42, "inputs": {}, "output": "…", "priority": 0, "params": {} }
}
```

- `GET /v1/jobs?state=done,failed&limit=200` returns `{"jobs": [...]}`.
- `DELETE /v1/jobs/{id}` cancels the job and returns it, or 404.

### `GET /v1/jobs/{id}/output`: download the result

- `200`: the file bytes, with `Content-Type` from the extension (`video/mp4`, `image/png`, ...) and
  `Content-Disposition: attachment; filename="<id>.<ext>"`.
- `409 {"error": "job <id> is queued; it has no output yet"}` when the job is not done.
- `404` when the job is unknown, has no output, or its output file is gone (`"... is gone: <path>"`).

## Files

### `POST /v1/files`: upload an input

The body can take either of two forms:
- `multipart/form-data` with a file part, named `file` (otherwise the first file part is used). Other
  form fields, such as `purpose`, are ignored.
- A raw body, with the name in `?name=start.png` or `X-Filename`, and the type in `Content-Type`.

Response `201`:

```json
{"id": "3f9c0a1b2c3d4e5f", "path": "/…/work/uploads/3f9c0a1b2c3d4e5f/start.png", "name": "start.png",
 "size": 7, "content_type": "image/png", "sha256": "…", "created_at": 1795589100.2}
```

- The file is stored at `<work_dir>/uploads/<id>/<safe name>`, so a client never needs to share a
  filesystem with the engine.
- An empty body returns 400.
- A body larger than `max_upload_mb` (config, default 2048) returns 413.
- A multipart body without a file part returns 400.
- Pass the `id` or the `path` in a request's `inputs`.

## Memory

### `GET /v1/memory[?model=<id>&width=&height=&frames=&steps=]`

```json
{
  "total": 68719476736,
  "available": 51539607552,
  "reserve": 0,                    // headroom the user set with headroom_gb (0: none, the default since 0.2.3)
  "reserve_auto": true,
  "free_for_models": 51539607552, // max(0, available - reserve), capped at max_fraction * total if set
  "max_job_bytes": null,          // that cap, when configured
  "swap": 0,                      // swap in use, bytes
  "pressure": "normal",           // normal | warn | critical
  "held": 536870912,              // bytes Shuttrr's model pool holds
  "resident_bytes": 536870912,    // held, fully in RAM
  "streamed_bytes": 0,            // held, weights streamed from SSD (only a window is resident)
  "components": [
    {"key": "dummy/dit", "model": "dummy", "size_bytes": 536870912, "residency": "resident",
     "in_use": false, "pinned": false, "idle_s": 1.0}
  ],
  "protected": [{"name": "oMLX", "busy": false}],
  "model": {                      // only with ?model=
    "id": "dummy", "need_bytes": 1073741824, "warm_bytes": 536870912,
    "fits": true, "short_by_bytes": 0, "error": null
  }
}
```

- **`residency`** is `"resident"` or `"streamed"`. The worker reports it per pool entry. A component
  loaded by a streaming rung (SSD weight streaming, docs/MODES.md) is `"streamed"`.
- **`model` block.**
  - `need_bytes` is the adapter's peak estimate for that request, using the current mode and
    efficiency. `warm_bytes` is the part of it already loaded.
  - `short_by_bytes = max(0, need - warm - free_for_models)`, and `fits = short_by_bytes == 0`.
  - On an unknown model, or an adapter that cannot estimate, `error` is set and the numbers are `null`.
- A non-integer `width` / `height` / `frames` / `steps` returns 400.
- The daemon also returns `"error"` at the top level if it cannot read memory.

The ComfyUI pack uses `fits` to decide whether to unload ComfyUI's own models before submitting.

## Clients and throughput

### `GET /v1/clients`

```json
{"window_s": 600,
 "clients": [{"client": "comfyui-shutter/0.2.5", "requests": 3, "requests_total": 5,
              "first_seen": 1795589100.2, "last_seen": 1795589400.9, "idle_s": 4.2}]}
```

- Clients are kept in memory only, so a restart forgets them.
- `requests` counts submits in the last `window_s`; `requests_total` counts submits since the client was
  first seen.
- A client seen only through the header has `requests: 0`.
- Rows are sorted by `last_seen`, newest first. Clients idle longer than the window drop out.

### `GET /v1/system`: additions

```json
"throughput": {
  "window_s": 3600,
  "finished": 12,              // done requests in the window
  "mean_seconds": 41.3,        // their mean wall time (started -> finished), null when none
  "running": {"id": "…", "model": "…", "step": 3, "steps": 8, "steps_per_s": 0.41}   // or null
}
```

- **`running`** describes the job running now. `step` and `steps` come from its stage text
  (`"denoise 3/8"`, `"segment 1/2 denoise 3/8"`, `"Step 4 / 10"`), and are null before the first counter.
- **`steps_per_s`** counts across segment restarts. It is null until two different step counts have
  been seen.
- The worker also sends `step`/`steps` with each progress event (docs/WORKER_PROTOCOL.md).
- **`pool.entries[]`** now carries `residency` (`"resident"` / `"streamed"`), alongside `key`,
  `size_bytes`, `idle_s`, `pinned` and `in_use`.

Unchanged routes:
- `GET /v1/models`: `{"models": [{id, family, kind: "video"|"image"|…, dir, installed, error, options}]}`
- `GET`/`POST /v1/settings`
- `POST /v1/queue/pause`, `POST /v1/queue/resume`
- `POST /v1/pool/release` with body `{"key": null | "<key>"}`, returning `{"released": [...]}`
- `GET /v1/events` (SSE; job events now include `client` on `job.queued`)

## OpenAI Images shim

`POST /v1/images/generations` and `POST /v1/images/edits` serve image-kind models only.

- Each image is one native job. The seeds run `seed`, `seed+1`, ..., so the queue, the memory guard and
  the modes apply as for any request.
- The HTTP request waits for its jobs. A client disconnect cancels them (best effort in the daemon; see
  below).
- If the jobs take longer than `images_timeout_s` (config, default 3600), the response is 504 and the
  jobs are cancelled.

**Generations** (JSON):

| Field | Support |
|---|---|
| `prompt` | required |
| `model` | an image model id; default is the first image model. Unknown → 404 `model_not_found`; a video model → 400 |
| `n` | 1..10 |
| `size` | `"WxH"` or `"auto"`/omitted (model default) |
| `response_format` | `b64_json` (default) or `url`. A `url` points at `GET /v1/jobs/{id}/output` on this server |
| `output_format` | `png` only (anything else → 400) |
| `stream` | not supported (400) |
| `quality`, `style`, `background`, `moderation`, `user` | accepted and ignored |
| Shuttrr extras | `negative_prompt`, `steps`, `guidance`, `seed`, `params` (an object, or a JSON string in multipart) |

**Edits**: either of two forms:
- `multipart/form-data`: the files `image` (or `image[]`, `images`, `images[]`, repeatable) and an
  optional `mask`, plus the same text fields.
- JSON `{"images": [{"image_url": "data:image/png;base64,…"} | {"file_id": "<upload id>"}], "mask": {...}}`.
  Only `data:` URLs are accepted; Shuttrr does not fetch remote URLs.

The images become `inputs.image`, `inputs.image_2`, ..., and the mask becomes `inputs.mask`. A missing
image returns 400 `image is required`.

Response:

```json
{"created": 1795589100, "size": "64x48", "output_format": "png",
 "data": [{"b64_json": "iVBOR…", "revised_prompt": "a red fox"}],
 "shutter": {"model": "dummy-image", "jobs": ["d98530c84d6c", "…"]}}
```

Errors:

```json
{"error": {"message": "…", "type": "invalid_request_error" | "server_error", "param": "size" | null, "code": null | "model_not_found" | "timeout" | "job_failed" | "job_cancelled"}}
```

A failed job returns 500 `server_error` with the job's message. The client name defaults to
`openai-images`, unless `X-Shutter-Client` or the User-Agent names one.

Limits:
- The daemon caps a generations body at 16 MB, and an edits body at `max_upload_mb`.
- In the daemon, cancelling on disconnect relies on hyper dropping the handler future when the client
  goes away. This is covered by a drop guard but not by a test. The 504 timeout path is tested.

## Fields the macOS app reads

| Where | Field |
|---|---|
| Job | `client` (string or null) |
| `GET /v1/clients` | `window_s`, `clients[].client`, `.requests`, `.requests_total`, `.first_seen`, `.last_seen`, `.idle_s` |
| `GET /v1/system` | `throughput.window_s`, `.finished`, `.mean_seconds`, `.running` (null or `{id, model, step, steps, steps_per_s}`) |
| `GET /v1/system` | `pool.entries[].residency` |
| `GET /v1/memory` | `components[].residency` (`"resident"` / `"streamed"`), `resident_bytes`, `streamed_bytes`, `held`, `free_for_models`, `reserve`, `swap`, `pressure`, `model.fits`, `model.short_by_bytes` |

## Model downloads

`GET /v1/catalog` lists the models Shuttrr can download: `{"models": [{"id", "name", "summary", "kind",
"source", "size_bytes", "min_memory_bytes", "fits_this_mac", "free_bytes", "dir", "installed", "notes",
"download", "fit", "disk", "recommended", "makes", "sample"}], "restart_needed"}`. `download` is null or `{"state",
"done_bytes", "total_bytes", "error"}` with state `downloading`, `done`, `failed`, `stopped` (the download process
died; it can resume) or `cancelled`.

`fit` is the engine's one fit classifier (`catalog.classify_fit`), against this Mac's total memory (not what is
free now), at the size the model runs by default here: `{"level": "recommended" | "barely" | "no_fit" | "unknown",
"label": "Recommended" | "Fits but barely" | "Doesn't fit", "reason", "detail", "peak_bytes", "measured", "at"}`.
It uses the measured peak for that size when there is one (`checked_on`), otherwise the engine's estimate; an
estimate never counts the SSD-streaming steps.

- `recommended`: measured at this size, and the full speed peak is at most 50% of total memory.
- `barely`: the lowest step's peak is at most 70% of total memory, or the model is only an estimate here
  (reason "Not tested on a Mac this size yet.", or "Tight on this Mac. Close other apps first.").
- `no_fit`: anything above, or under the model's smallest Mac (reason "Needs about 32 GB. This Mac has 16 GB.").
  `POST /v1/catalog/{id}/download` refuses these with 409 unless `?force=1`.

`reason` is one short line for the app to show. `disk` is `{"ok", "need_bytes", "label"}` (label empty when the
download fits on the disk). `recommended` is `fit.level == "recommended"`. `makes` is one line such as "Makes
320x512 clips on this Mac". `sample` is null or the request "Try a Sample" sends (`{"prompt", "width", "height",
"frames", "steps", "seed"}`); the app posts it to `POST /v1/jobs` with the model's id. Rows come sorted:
Recommended, then Fits but barely, then Doesn't fit.
`restart_needed` (Rust engine only) is true when a model finished downloading after the engine started.

`POST /v1/catalog/{id}/download` starts or resumes the download and returns the catalog (202).
`DELETE /v1/catalog/{id}/download` stops it and returns the catalog; the partial files stay for a resume.
