# Shuttrr: front-end integrations

Researched 2026-11-25. Tags: **[V]** = checked by me in the project's current source or docs (link given);
**[R]** = from memory, not re-checked. Snapshots read: ComfyUI 0.46.0 (commit 0d0e105, 2026-11-24),
draw-things-community HEAD of 2026-11-24, drawthingsai/draw-things-comfyui 1.14.0, SwarmUI HEAD of 2026-11-22.

## Summary

| Front-end | What it can plug into | Effort | What the user gets |
|---|---|---|---|
| **ComfyUI** | Custom node pack (Python, in ComfyUI's process) that calls Shuttrr's HTTP API | Low–medium | Shuttrr models as nodes inside any Comfy graph; IMAGE/VIDEO/AUDIO outputs feed normal Comfy nodes |
| **Draw Things** | Only its own gRPC "server offload" protocol (DT is the client) | High, with constraints | DT's UI driving Shuttrr as its compute. No HTTP client mode, no plugins, no networking from scripts |
| **Open WebUI** | "OpenAI compatible" image connection, A1111, or ComfyUI | Low (once the OpenAI-images shim exists) | Image generation from chat |
| **SwarmUI** | ComfyUI-API backend (+ Shuttrr nodes) or Auto WebUI (A1111) backend | Free once the Comfy nodes or A1111 shim exist | Swarm UI on top of Shuttrr |
| **Scripts / agents / MCP tools** | Shuttrr's native API, OpenAI-images shim, A1111 shim | Already there (native) | Programmatic use |
| Krita AI Diffusion | Only a real ComfyUI server with its own required node set | Not worth it | — |
| Mochi Diffusion, DiffusionBee, InvokeAI | No external-backend setting that I could verify | — | — |

## 1. ComfyUI

**Packaging and registration** [V]
- The V1 style still loads: a module-level `NODE_CLASS_MAPPINGS` (plus optional `NODE_DISPLAY_NAME_MAPPINGS`). In V1, a class declares `INPUT_TYPES`, `RETURN_TYPES` and `FUNCTION`. Source: [nodes.py](https://github.com/comfyanonymous/ComfyUI/blob/master/nodes.py), `load_custom_node`.
- The V3 schema exists now, and the loader checks for `comfy_entrypoint` when `NODE_CLASS_MAPPINGS` is absent. It is described in the [V3 migration guide](https://docs.comfy.org/custom-nodes/v3_migration).
  - Import with `from comfy_api.latest import ComfyExtension, io, ui`. You can pin a version instead (`comfy_api.v0_0_2`).
  - A node subclasses `io.ComfyNode` and provides `define_schema() -> io.Schema` and a classmethod `execute(...)`, which may be `async`.
  - `execute` returns `io.NodeOutput(...)`.
  - The extension class implements `async get_node_list()`, and `comfy_entrypoint()` returns that extension.
- Publishing is documented at [registry publishing](https://docs.comfy.org/registry/publishing).
  - Run `comfy node init` to generate `pyproject.toml`: `[project] name/version/license/dependencies` and `[tool.comfy] PublisherId/DisplayName/Icon`.
  - Then publish with `comfy node publish`, or with the GitHub Action `Comfy-Org/publish-node-action` using the `REGISTRY_ACCESS_TOKEN` secret.
- Recommendation: write V3. Declare no dependencies beyond `requests`/`aiohttp` (no torch pins).

**Types that cross node boundaries** [V unless marked]. Sources: [datatypes](https://docs.comfy.org/custom-nodes/backend/datatypes), [images & masks](https://docs.comfy.org/custom-nodes/backend/images_and_masks), `comfy_api/latest/_io.py`.
- `IMAGE`: `torch.Tensor [B,H,W,C]`, C=3, channel-last, float32 in 0..1. `LoadImage` does `np.float32 / 255.0`.
- `MASK`: `[B,H,W]`, float 0..1.
- `LATENT`: `dict` with `samples [B,C,H,W]` and optional `noise_mask`, `batch_index`, `type`. The channel count depends on the model.
- `AUDIO`: `{"waveform": Tensor[B,C,T], "sample_rate": int}`. The V3 TypedDict misspells it `sampler_rate`, but the nodes use `sample_rate`.
- `VIDEO`: a `VideoInput` object exposing `get_components()`, which returns `VideoComponents(images: IMAGE, frame_rate: Fraction, audio: AUDIO|None, metadata)`.
  - Implementations: `VideoFromFile` (path/BytesIO) and `VideoFromComponents`, in `comfy_api/latest/_input_impl/video_types.py`.
  - For Shuttrr, the easiest output is to return an mp4 path wrapped in `VideoFromFile`, plus the frames as `IMAGE`.
- `CONDITIONING` (a list of `[tensor, {pooled_output, ...}]`) and `MODEL` (`ModelPatcher`) are torch objects tied to Comfy's own samplers. Shuttrr cannot produce or consume them meaningfully. Use **custom opaque types** instead (e.g. `SHUTTER_MODEL`, `SHUTTER_COND`) that hold handles to state kept inside Shuttrr.

**Progress and interruption** [V]
- `comfy.utils.ProgressBar(total, node_id)`, then `.update_absolute(v, total, preview)`. Updates are throttled except the first, the last, and any update carrying a preview.
- In V3, `ComfyAPI().execution.set_progress(value, max_value, preview_image=...)` replaces `PROGRESS_BAR_HOOK`.
- Interruption: `POST /interrupt` sets a global flag. A node polls `comfy.model_management.processing_interrupted()` or calls `throw_exception_if_processing_interrupted()` (raises `InterruptProcessingException`).
- A Shuttrr node should poll Shuttrr's job status about every 250 ms. On each poll it updates the bar (and decodes a preview if one is sent), checks the interrupt flag, and calls Shuttrr's cancel endpoint when the flag is set.
- The official Draw Things pack does exactly this. It uses `comfy.utils.ProgressBar(config.steps, unique_id)`, re-raises Comfy's interrupt exception, and adds its own `/dt_grpc/interrupt` route ([draw-things-comfyui](https://github.com/drawthingsai/draw-things-comfyui)).

**Memory coordination on Apple Silicon** [V, `comfy/model_management.py`]
- With MPS, ComfyUI sets `vram_state = SHARED`. Models load straight to the `mps` device.
- `get_free_memory()` for `mps` returns *system* `virtual_memory_available()`. So ComfyUI does see the RAM Shuttrr holds and evicts its own models when RAM runs low. It will not proactively release memory for Shuttrr, though.
- `unload_all_models()` calls `free_memory(1e30, device)` for every device. `soft_empty_cache()` calls `torch.mps.empty_cache()`.
- `POST /free {"unload_models": true, "free_memory": true}` only sets queue flags. They are applied **after the current prompt finishes**, in the main.py worker loop. It is therefore useless from inside a running node.
- What to do in the Shuttrr node before submitting a job:
  1. Call `GET /v1/memory` on Shuttrr, which returns needed and free bytes.
  2. If ComfyUI holds models and the free space is short, call `comfy.model_management.unload_all_models()` and then `soft_empty_cache()`, in process.
  3. Expose a `free_comfy_models` toggle on the node, defaulting to on.
- The reverse direction: ComfyUI nodes that run after Shuttrr need RAM back. Add a `keep_loaded` option (like Ollama's `keep_alive`) that maps to Shuttrr's unload/idle-timeout policy. Also ship a `Shuttrr Unload` node.

**Existing packs worth copying** [V]
- [drawthingsai/draw-things-comfyui](https://github.com/drawthingsai/draw-things-comfyui) is the closest analogue: an external engine over the network.
  - It has a coarse **Draw Things Sampler** node returning `IMAGE, AUDIO`.
  - Config-carrying side nodes build up the request: Prompt, Positive/Negative, LoRA, Control Net, Hints, Refiner, Upscaler, passing custom types `DT_LORA`, `DT_CNET`, `DT_HINTS`, ...
  - It keeps model lists in sync through custom HTTP routes and a JS frontend extension.
  - It is V1 style (`NODE_CLASS_MAPPINGS`).
- [stavsap/comfyui-ollama](https://github.com/stavsap/comfyui-ollama) calls a local HTTP server (default `http://127.0.0.1:11434`). It sends IMAGE inputs as base64 PNG and exposes a `keep_alive` option for unloading the remote model.
- [raysers/Mflux-ComfyUI](https://github.com/raysers/Mflux-ComfyUI) imports `mflux` **in process** (MLX inside ComfyUI's Python). Its last commit was Dec 2024. That design is what Shuttrr should avoid: two allocators in one process.
- thoddnn/ComfyUI-MLX exists (Registry listings), but its repo did not clone (it may be private or moved); I believe it is also in-process. [R]
- ComfyUI's own "API nodes" (`comfy_api_nodes/`) call partner clouds through the `api.comfy.org` proxy (`--comfy-api-base`). They are not a mechanism for local engines.

**Granularity: recommendation**
- Make **one coarse `Shuttrr Generate` node the primary path**. It should be async, accept optional IMAGE/MASK inputs (img2img, i2v), and return `IMAGE` (+ `VIDEO`, `AUDIO`).
  - Reason: Shuttrr's staged loading and SSD streaming plan the whole pipeline (text encoder, then denoiser, then VAE) as a unit. Split nodes would force Shuttrr to keep several components resident between calls, or reload them, and hand Comfy tensors it cannot use anyway.
- Add **side nodes that only build the request** (the draw-things-comfyui pattern) rather than separate loader/encode/sample/decode stages that compute independently.
- Optional later: `Shuttrr Encode Prompt` returning a `SHUTTER_COND` handle, for prompt reuse across seeds, and `Shuttrr Decode` from a `SHUTTER_LATENT` handle. Both are handles into Shuttrr's cache, never tensors.

## 2. Draw Things

**What exists** [V, [draw-things-community](https://github.com/drawthingsai/draw-things-community)]
- The repo is GPL-3.0 (`LICENSE.md`), with LGPL possible case by case, and a `CLA/` directory.
- It includes `gRPCServerCLI` (macOS and Linux/CUDA binaries). The app connects to it via **Server Offload**; Docker users must enter IP:port by hand.

**The protocol** [V, `Libraries/GRPC/Models/Sources/imageService/imageService.proto`]
- `service ImageGenerationService` with these RPCs:
  - `GenerateImage(ImageGenerationRequest) returns (stream ImageGenerationResponse)`
  - `FilesExist`
  - `UploadFile` (bidirectional stream, chunked, sha256)
  - `Echo`
  - `Pubkey`
  - `Hours`
- `ImageGenerationRequest` fields:
  - `prompt`, `negativePrompt`
  - `configuration`: bytes, a **FlatBuffer `GenerationConfiguration`** from `Libraries/DataModels/Sources/config.fbs`, with fields such as `model: string`, `start_width/height` (ushort), `seed`, `steps`, `sampler`, `num_frames`, LoRAs and controls.
  - `image` and `mask`: sha256 references into `contents`, a content-addressed list of blobs.
  - `hints`: `HintProto` with tensors and weights
  - `override`: a `MetadataOverride` of JSON-encoded model/LoRA/ControlNet specs
  - `sharedSecret`, `chunked`
- Response stream fields: `currentSignpost` (textEncoded / sampling{step} / imageDecoded / …), `previewImage`, `generatedImages` (repeated bytes, so video comes back as frames), `generatedAudio`, chunking (`chunkState`, `downloadSize`) and `remoteDownload` progress.
- **Tensor encoding.**
  - The format is s4nnc `Tensor` serialisation: a ccv_nnc tensor header, then Float16 NHWC values in **[-1, 1]**, optionally compressed with zip+fpzip.
  - Server side: `Tensor<FloatType>(data:, using: [.zip, .fpzip])` in `ImageGenerationServiceImpl.swift`.
  - The header layout is documented in draw-things-comfyui `image_handlers.py`.
- **Model identity.**
  - `configuration.model` is a Draw Things file name (`*.ckpt` in DT's own s4nnc/SQLite format).
  - With `--model-browser`, `Echo` returns the server's `.ckpt` files and a `MetadataOverride` holding `ModelZoo.Specification` JSON. This is how a server advertises models the app does not know.
- **Auth and TLS.**
  - Authentication is an optional `sharedSecret` in each request; `Echo` reports `sharedSecretMissing`.
  - TLS is on by default, using a server cert/key **bundled in the binary**. The app trusts a bundled DT root CA plus ISRG Root X1, and can skip hostname verification. Turn TLS off with `--no-tls`.
  - The client wrapper takes a `TLS: Bool` parameter.
- **Discovery.** Bonjour/DNS-SD service type `_dt-grpc._tcp` (`GRPCServerAdvertiser.swift`). The default port is 7859.

**Could Shuttrr be DT's compute?** Technically possible; I found nothing in the source or docs that forbids it. It requires all of the following:
1. A gRPC server (e.g. `tonic` in Rust) implementing at least `Echo`, `FilesExist`, `UploadFile` and `GenerateImage`, with the same proto.
2. Parsing the FlatBuffer config (`flatc --rust` on `config.fbs`).
3. Encoding and decoding s4nnc tensors (header + fp16; fpzip only if compression is negotiated on).
4. Mapping DT model file names, or names Shuttrr advertises via `MetadataOverride.models` specs, onto Shuttrr models.
5. Accepting DT LoRA and control conventions, or rejecting them clearly.
6. Plaintext operation (the user turns TLS off in DT) unless we reuse DT's bundled certificate.

**Costs and risks:**
- The proto, the `.fbs` and the reference code are GPL-3.0. Copying them into Shuttrr, or generating code from them, likely puts that component under GPL; this needs a legal decision.
  - Mitigation: ship the adapter as a separate GPL'd process ("shutter-dt-bridge") that talks to the Shuttrr daemon over its HTTP API.
- DT's UI exposes only what DT's ModelZoo knows about samplers, sizes and model families. Shuttrr-only models need valid `Specification` JSON to appear at all, and whether DT honours an arbitrary version string is unverified.
- The protocol is internal and changes with app releases; for example, `chunked`, `generatedAudio` and `remoteDownload` were added over time. Expect breakage on DT updates.

**What is not possible** [V in source unless marked]
- DT's **HTTP API is server-only**: `HTTPAPIServer.swift` registers `POST /sdapi/v1/txt2img`, `POST /sdapi/v1/img2img`, `GET /sdapi/v1/options` and `GET /`. DT cannot be pointed at an A1111 or OpenAI URL as a client.
- **Scripts cannot reach Shuttrr.** The JavaScriptCore bridge (`Libraries/Scripting/Sources/ScriptExecutor.swift`) exposes canvas, mask, file, moodboard and `generateImage` functions. It has no `fetch`, XMLHttpRequest or URLSession, and the image loaders take src/data strings.
- No plug-in system was found in the public repo or docs. [V for the repo, R for the app]
- The reverse direction is easy and already exists: Shuttrr, or any tool, can *use* DT as a backend through DT's HTTP API or gRPC (e.g. draw-things-comfyui). That does not make Shuttrr usable *from* DT.

**Conclusion:** the only way to make Shuttrr usable *from* Draw Things is to impersonate a gRPCServerCLI over the GPL'd internal protocol. That is a significant, fragile adapter, so build it last, as a separate process, if at all.

## 3. De-facto local APIs

**AUTOMATIC1111 `/sdapi/v1/*`** [V, [api.py](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/api/api.py)]
- Status: the repo's last commit is 2024-07-27, so it is frozen. Forge and Draw Things serve subsets of it.
- Endpoints:
  - `POST txt2img` / `img2img`: JSON with `prompt`, `negative_prompt`, `width`, `height`, `steps`, `cfg_scale`, `seed`, `sampler_name`, `batch_size`, `init_images[]` (base64), `mask`, `denoising_strength`, `override_settings`. Returns `{images:[b64], parameters, info}`.
  - `GET progress`: `{progress 0..1, eta_relative, state, current_image b64, textinfo}`.
  - `POST interrupt` and `POST skip`.
  - `GET/POST options`: `sd_model_checkpoint` selects the model.
  - `GET sd-models`, `samplers`, `schedulers`, `memory`, `unload-checkpoint`.
- The call is synchronous: progress requires polling from a second connection.
- Consumers: Open WebUI [V docs](https://docs.openwebui.com/features/chat-conversations/image-generation-and-editing/automatic1111), SwarmUI's AutoWebUI backend [V source], and many scripts and bots [R]. Image only.

**OpenAI Images** [V, [reference](https://developers.openai.com/api/reference/resources/images.md)]
- `POST /v1/images/generations` takes `model`, `prompt`, `n`, `size`, `quality`, `output_format`, `background`, `stream`, `partial_images` (0–3).
- `POST /v1/images/edits` takes `images[]` and `mask` (via `image_url`/`file_id` in JSON, or multipart).
- The response is `{created, data:[{b64_json, revised_prompt}], usage, ...}`.
- Streaming uses SSE events `image_generation.partial_image` / `.completed` (and `image_edit.*`).
- `variations` is retired, and `response_format` is legacy.
- There is no cancel: the client closes the connection.
- Consumers: Open WebUI ("OpenAI compatible" connection with a custom URL [V](https://docs.openwebui.com/features/chat-conversations/image-generation-and-editing/openai)), LiteLLM, and any OpenAI SDK with `base_url` [R].

**OpenAI Videos** [V, [reference](https://developers.openai.com/api/reference/resources/videos.md)]
- The page states that **the Sora 2 models and Videos API were shut down on 2026-09-24**, with no replacement.
- The shape was an async job: `POST /videos` returns `{id, status: queued|in_progress|completed|failed, progress}`; poll with `GET /videos/{id}`, then fetch `GET /videos/{id}/content?variant=video|thumbnail|spritesheet`; `DELETE` also existed.
- It is not a useful compatibility target now. The *pattern* (job, poll, content) is still the right shape for Shuttrr's native video API.

**ComfyUI server API** [V, `server.py`]
- Endpoints: `POST /prompt` `{prompt: <API-format graph>, client_id}` returns `{prompt_id}`; `GET /history/{id}`; `GET /view?filename=`; `POST /upload/image`; `GET /object_info`; `POST /interrupt`; `POST /free`; the newer `GET /api/jobs` and `POST /api/jobs/{id}/cancel`.
- The `/ws?clientId=` websocket emits `status`, `executing`, `progress`, `executed`, `execution_success`, `execution_error` and `execution_interrupted`.
- Should Shuttrr be ComfyUI-API-compatible? **No.** Clients submit graphs of *Comfy node class names* (KSampler, CLIPTextEncode, ...), so compatibility means re-implementing Comfy's node semantics.
  - Clients that consume it (Krita AI Diffusion [V README](https://github.com/Acly/krita-ai-diffusion), SwarmUI, Open WebUI's ComfyUI mode, and most Wan/Hunyuan video front-ends [R]) get Shuttrr through the **node pack inside a real ComfyUI**, not through emulation.
  - Exception: Open WebUI and SwarmUI let the user supply the workflow JSON. A workflow containing `Shuttrr Generate` therefore works with them unchanged.

**Video specifically:** I found no cross-front-end local video API. Wan/Hunyuan/LTX tools either run in process (Gradio apps) or drive ComfyUI workflows [R]. Shuttrr's native async job API plus the ComfyUI `VIDEO` output covers this.

## 4. Other Mac front-ends
- **Open WebUI**: Settings > Images has OpenAI-compatible (custom URL), Automatic1111 and ComfyUI connections. [V docs]
- **SwarmUI**: built-in backends ComfyUI API, ComfyUI self-start, AutoWebUI (A1111) API and self-start, and Swarm-to-Swarm. [V `src/Backends`, `src/BuiltinExtensions`] Through ComfyUI with the Shuttrr nodes, it also needs the Shuttrr node in its Comfy workflow tab.
- **Krita AI Diffusion**: ComfyUI only, with a fixed required node/model set; MPS on macOS 14+ is supported. [V README] It will not call custom Shuttrr nodes.
- **Mochi Diffusion** (Core ML), **DiffusionBee**, **InvokeAI**: I could not verify any external-backend option. As far as I know, all run their own engines in process. [R]

## Recommended build order

Status: steps 1, 2 and 3 are built in part. The API actually shipped is in docs/API.md:
- uploads are `POST /v1/files`
- outputs are `GET /v1/jobs/{id}/output`
- cancel is `DELETE /v1/jobs/{id}`

Built so far:
- The node pack is in `integrations/comfyui`: Generate, Unload and Memory. The LoRA/Control builders,
  previews and the A1111 shim are not built.
- The OpenAI Images shim has no SSE partials.

1. **Native Shuttrr API (stabilise first; everything else is a thin adapter over it)**
   - `GET /v1/health`, `GET /v1/models`, `GET /v1/memory` (total/free/held/needed-by-model), `POST /v1/models/{id}/load|unload`
   - `POST /v1/jobs` (image or video; JSON with base64 or uploaded-file refs; `mode: fast|low_memory`, `efficiency`) returns `{job_id}`
   - `GET /v1/jobs/{id}` (status, step/total, stage, eta, preview url), `GET /v1/jobs/{id}/events` (SSE), `POST /v1/jobs/{id}/cancel`
   - `GET /v1/jobs/{id}/outputs/{n}` (png / mp4 / wav), `POST /v1/uploads`
2. **ComfyUI node pack `comfyui-shutter`** (V3, Registry-published, HTTP only, no torch beyond what Comfy has)
   - `Shuttrr Generate` (primary). Inputs: model (combo filled from `/v1/models`), prompt, negative, width, height, frames, fps, steps, cfg, seed, mode, efficiency, optional IMAGE/MASK, `free_comfy_models`, `keep_loaded`. Outputs: IMAGE, VIDEO, AUDIO.
   - `Shuttrr LoRA` and `Shuttrr Control`: request builders with custom types.
   - `Shuttrr Unload` and `Shuttrr Memory Info`.
   - Progress through ProgressBar/`set_progress` with previews; cancel on Comfy interrupt; unload Comfy models in process when `/v1/memory` says so.
   - Later: handle-based `Shuttrr Encode Prompt` → `SHUTTER_COND` and `Shuttrr Decode`.
3. **OpenAI Images shim**
   - `POST /v1/images/generations` and `/v1/images/edits`, with b64 responses and SSE partials mapped from previews.
   - `model` maps to a Shuttrr model id. Unlocks Open WebUI, LiteLLM and SDK users.
4. **A1111 shim (image only)**
   - `txt2img`, `img2img`, `progress`, `interrupt`, `options` (GET/POST `sd_model_checkpoint`), `sd-models`, `samplers`, `memory`.
   - Unlocks SwarmUI's AutoWebUI backend, older tools and Open WebUI's A1111 mode.
5. **(Optional, last) `shutter-dt-bridge`**
   - A separate GPL-3.0 process implementing `ImageGenerationService` (Echo, FilesExist, UploadFile, GenerateImage) over the native API.
   - Advertise via `_dt-grpc._tcp`. Start plaintext with a shared secret.
   - Only after a licence decision and a spike that proves DT accepts Shuttrr-advertised model specs.

## Not possible or not worth doing
- **Draw Things as an HTTP or OpenAI client of Shuttrr**: not possible. DT only serves `/sdapi/v1/*`; it never calls out.
- **Draw Things script or plug-in integration**: not possible. The JS bridge has no network API, and there is no plug-in system.
- **Shipping DT's proto or fbs inside the main (non-GPL) Shuttrr binary**: avoid; the source is GPL-3.0. Keep any DT adapter separate.
- **Emulating the ComfyUI server API (`/prompt` with Comfy graphs)**: not worth it. It means re-implementing Comfy node semantics. Use the node pack inside real ComfyUI.
- **Running MLX in process inside ComfyUI** (the Mflux-ComfyUI approach): not worth it. Two memory managers in one process defeats Shuttrr's no-swap guarantees.
- **Passing Comfy `MODEL`/`CONDITIONING`/`LATENT` tensors to Shuttrr**: not meaningful. They are PyTorch objects for Comfy's samplers. Use opaque handles instead.
- **Relying on ComfyUI's `POST /free` from a node**: it does not work mid-prompt (flags apply after the prompt ends). Call `unload_all_models()` in process.
- **OpenAI Videos compatibility**: not worth it. The API was shut down on 2026-09-24, and no local front-end consumes it.
- **Krita AI Diffusion, Mochi Diffusion, DiffusionBee, InvokeAI**: no external-engine hook found.
