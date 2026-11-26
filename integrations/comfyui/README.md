# comfyui-shutter

ComfyUI nodes that run Shuttrr's models over Shuttrr's HTTP API. Shuttrr keeps its own process and memory
manager; ComfyUI never loads MLX. Background and design are in [docs/INTEGRATIONS.md](../../docs/INTEGRATIONS.md)
section 1, and the API is in [docs/API.md](../../docs/API.md).

> **Not run inside a real ComfyUI.** ComfyUI was not installed where this pack was written, so the pack
> has never been loaded by ComfyUI's custom-node loader or run in a real graph. Here is what has been tested:
>
> - **Tested against the real Python reference server** (`tests/test_comfyui_pack.py`, using the dummy
>   models): the HTTP client, the IMAGE/AUDIO/video conversions and the runner. The runner tests cover
>   memory check, uploads, submit, polling, progress, cancel, download, decode and release.
> - **Tested against a hand-written stand-in, not the real modules:** `nodes.py` was exercised against a
>   minimal fake of `comfy_api.latest`, `comfy.utils` and `comfy.model_management`. That checks this pack's
>   own wiring. It does not check that the real modules behave as assumed.
> - **Written from the V3 docs and ComfyUI 0.46 source notes in INTEGRATIONS.md, never executed:**
>   - the exact `io.*.Input` / `io.*.Output` keyword names (`options`, `optional`, `tooltip`, `multiline`,
>     `display_name`)
>   - `io.Hidden.unique_id`
>   - `InputImpl.VideoFromFile`
>   - `comfy_entrypoint` being `async`
>   - `ProgressBar.update_absolute`
>   - `processing_interrupted` and `throw_exception_if_processing_interrupted`
>   - `unload_all_models` and `soft_empty_cache`
>   - torch tensors in and out
> - **Never run:** reading video through PyAV. The tests use the ffmpeg command-line fallback.
> - **Model list loading:** the model combo is filled from `/v1/models` when ComfyUI loads the node. If
>   Shuttrr is not running at that moment, the combo shows a placeholder until ComfyUI reloads. The
>   `model_id` field overrides the combo.
>
> Expect to fix small API mismatches the first time this runs in a real ComfyUI.

## Install

Copy or symlink this directory into `ComfyUI/custom_nodes/comfyui-shutter`. It has no extra dependencies:
HTTP uses the Python standard library, and numpy, torch and Pillow come with ComfyUI. Start Shuttrr
(`shutter serve` or `shutterd`) before ComfyUI.

Configuration:
- `SHUTTER_URL`: the Shuttrr server address. The default is `http://127.0.0.1:8484`.
- `SHUTTER_TOKEN`: a bearer token, if Shuttrr requires one.

## Nodes

**Shuttrr Generate** (async; category *Shuttrr*)

| Input | Meaning |
|---|---|
| `model` | combo filled from `GET /v1/models` (image- and video-kind models) |
| `prompt`, `negative_prompt` | text |
| `width`, `height`, `frames`, `fps`, `steps`, `guidance` | 0 = the model's default |
| `seed` | -1 = Shuttrr picks |
| `memory_mode` | `default` / `fast` / `low_memory`; sent as `params.mode` (docs/MODES.md) |
| `efficiency` | the low-memory slider (`params.efficiency`); 0 = default |
| `free_comfy_models` | default on. Before submitting, the node asks `GET /v1/memory?model=…`. When `fits` is false, it calls ComfyUI's `unload_all_models()` and `soft_empty_cache()` in process. (`POST /free` would only apply after the prompt ends.) |
| `keep_loaded` | default on. Off: after the request, `POST /v1/pool/release` frees Shuttrr's idle models, so later ComfyUI nodes get the RAM back |
| `server_url` | the Shuttrr address |
| `model_id` | free text that overrides the combo |
| `params_json` | extra adapter params as a JSON object, for example `{"framing": "small"}` for the reels add-on |
| `image`, `mask`, `audio` (optional) | uploaded with `POST /v1/files`, then passed as `inputs.image`, `inputs.mask` and `inputs.audio` |

Outputs:
- `images`: `IMAGE` `[B,H,W,3]` float32 0..1. This is one frame for an image model, or every frame for a
  video model.
- `video`: `VIDEO` wrapping the mp4. It is `None` for image models, and the output is only present when
  the installed ComfyUI has `io.Video`.
- `audio`: `AUDIO` from the mp4's audio track, if it has one. The output is only present when ComfyUI has
  `io.Audio`.

Progress goes to ComfyUI's progress bar. The node polls the job every 250 ms. If you press Cancel in
ComfyUI, the node sends `DELETE /v1/jobs/{id}`. The Shuttrr job is also cancelled if the node fails or its
task is cancelled.

**Shuttrr Unload**: `POST /v1/pool/release`. **Shuttrr Memory**: `GET /v1/memory` as JSON text.

Requests from this pack are tagged with the client name `comfyui-shutter/0.2.5`. That name appears in
`GET /v1/clients` and on each job.

## Layout

```
__init__.py              ComfyUI entry: exports comfy_entrypoint only when comfy_api imports
shutter_comfy/client.py  stdlib HTTP client for the native API
shutter_comfy/convert.py PNG <-> IMAGE, WAV <-> AUDIO, mp4 -> frames (+ audio); numpy, torch optional
shutter_comfy/runner.py  one request end to end, driven by plain callbacks (testable without ComfyUI)
shutter_comfy/nodes.py   the V3 nodes; the glue between runner callbacks and ComfyUI
```

## Tests

From the repository root: `pytest tests/test_comfyui_pack.py`. The tests start the Python reference
server on a free local port with the dummy models. The video tests need `ffmpeg` and `ffprobe` on PATH.
