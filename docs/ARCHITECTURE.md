# Shuttrr architecture

Shuttrr is a Mac-only inference engine for large video and image diffusion models. Goal: the
same output quality in less memory, never pushing the Mac into swap, on MLX and Metal. It is an
engine that front-ends plug into (ComfyUI node pack, OpenAI Images shim, scripts, the Swift app),
not a content tool: a request is a plain inference request (docs/API.md), and anything specific to
one kind of content lives in optional add-ons that adapters read from `params`.

```
 SwiftUI menu bar app (app/Shutter)  ──HTTP──┐
 CLI (shutter submit/status/...)     ──HTTP──┤
 other tools, later exo              ──HTTP──┤
                                             ▼
                    shutter.server.api  (Starlette, 127.0.0.1:8484)
                                             │
                    shutter.core.engine.Engine  (one process, one GPU worker thread)
        ┌──────────────┬───────────────┬─────┴────────┬───────────────┐
   core.queue      core.memory     core.pool      core.cache      core.registry
   JobStore +      MemoryGuard +   ModelPool      ContentCache    models + adapters
   Scheduler       SystemProbe +   (keep warm,    (prompt, audio
                   GpuLock         evict, TTL)    embeddings)
                                             │
                    shutter.adapters.*  (LongCat avatar first, then FastH3, MiniMax H3)
                                             │
                    core.stream (SSD weight streaming)

 optional, outside the core:  shutter.addons.reels (framing, crop box, room lock, zoom-cancel
                              upscale, shot placement), read only by adapters that support it
 integrations/comfyui         ComfyUI node pack (HTTP client of the API)
```

## Rules the code must enforce

1. One heavy job at a time. One GPU worker thread; an external file lock (`GpuLock`)
   shared with other tools on the machine.
2. Never push the Mac into swap. Before every load and every job the MemoryGuard checks
   (a) the adapter's estimate fits in available memory minus headroom, (b) swap is not
   growing, (c) memory pressure is normal, (d) no protected process (the user's chat model
   on oMLX, port 8000) has a model loaded. If any fails the job waits (`waiting_memory`);
   it is never started "hoping it fits". Shuttrr never unloads or changes anything in
   oMLX, Splash or Strata.
3. Staged loading. Text encoder, then DiT, then VAE; each released when its stage ends
   unless the pool has room to keep it warm.
4. Every lossless speed-up must be bit-identical to the reference path; anything that is
   not is opt-in and labelled.
5. Deterministic shots. A shot's output depends only on the shot and the model build, so
   shots can later run on other Macs (exo) and be joined.

## Module contracts

All shared data types live in `shutter/types.py`; the adapter contract in
`shutter/adapters/base.py`. Read both before writing any module.

### core/memory.py
- `SystemProbe` (Protocol): `sample() -> MemorySample` with fields `total`, `available`
  (free + inactive + purgeable + speculative), `swap_used`, `pressure` ("normal" | "warn"
  | "critical"), `ts`. `MacProbe` reads `sysctl hw.memsize`, `vm_stat`,
  `sysctl vm.swapusage`, `sysctl kern.memorystatus_vm_pressure_level`. `FakeProbe` for tests.
- `ProtectedService` (Protocol): `busy() -> Optional[str]` returns a reason string when
  the service holds memory we must not compete with. `OmlxService(base_url)` checks the
  oMLX server (read-only GETs only) for loaded models; unreachable server = not busy.
- `MemoryGuard(probe, protected=[...], headroom_bytes, swap_growth_bytes)`:
  `check(need_bytes, what) -> Decision(ok, reason, sample)`; `admit(need_bytes, what)`
  raises `MemoryRefused`; `watch()` used during a run: returns a reason to abort when swap
  grows by more than `swap_growth_bytes` since the run started or pressure is critical.
- `GpuLock(path)`: cross-process exclusive lock with `fcntl.flock`, context manager,
  `try_acquire(timeout)`; writes owner pid + label into the file.

### core/queue.py
- `JobStore(path)`: sqlite persistence of `Job` (JSON blob + indexed state, priority,
  created_at). `add`, `get`, `update`, `list(states=None, limit)`, `next_queued()`
  (highest priority, then oldest). On open, jobs left `running` become `queued` again
  (crash recovery) with message "requeued after restart".
- `EventBus`: thread-safe publish/subscribe; `subscribe()` returns a queue-like object
  with `get(timeout)`; bounded, drops oldest.

### core/pool.py
- `ModelPool(guard, on_event)`: implements `ModelPoolLike`. Tracks `key -> (object,
  size_bytes, last_used, pinned)`. `get()` admits through the guard; if refused it evicts
  least-recently-used unpinned entries and retries, then raises `MemoryRefused`.
  `release()` drops the reference, runs `gc.collect()` and, when MLX is importable,
  `mx.clear_cache()`. `evict_idle(ttl_s)`, `stats()`.
- Each entry has a `residency`: `"resident"` (all in RAM) or `"streamed"` (weights streamed
  from SSD). It is set by `get(..., residency="streamed")`, or detected from the object
  (`obj.residency`, a dict's `"residency"` key, or LongCat's `_shutter_stream_cfg`).
  `GET /v1/memory` reports it per component.

### core/cache.py
- `ContentCache(root, max_bytes)`: `get_or_compute(namespace, key_parts, compute)`; key =
  sha256 of the JSON of key_parts (files are hashed by content, via `file_digest(path)`).
  Values are dicts of numpy arrays (or MLX arrays, converted) stored as `.npz`; LRU
  eviction by size. Used for prompt embeddings and audio features.

### core/stream.py  (SSD weight streaming)
- `SafetensorsIndex(paths)`: parses safetensors headers without loading data; maps
  tensor name -> (file, offset, nbytes, dtype, shape).
- `BlockStreamer(index, groups, window, readahead)`: `groups` is an ordered list of
  tensor-name lists (one per transformer block). A background thread reads group i+1..
  i+readahead from SSD (plain `os.pread`, releasing the GIL) while the caller computes
  group i; at most `window` groups are resident. `with streamer.block(i) as tensors:`
  yields `dict[name, mx.array]`. Records bytes read, read seconds, wait seconds (time the
  compute thread blocked on I/O = the real cost of streaming).
- `StreamedStack(blocks_factory, streamer)`: runs a stack of identical blocks by
  loading each block's weights into one or two reusable module shells. Output must be
  bit-identical to the fully loaded stack (tested on MLX CPU).

### Families (shutter/families)
- A family is a set of checkpoints that share one architecture (Wan today). Shuttrr supports
  a family, not a list of files: any checkpoint whose tensors have the family's shapes runs,
  fine-tunes and other people's builds included. Anything else is refused with a reason,
  before anything is loaded. `shutter.families.inspect(path)` returns a `Checkpoint`
  (family, variant, task, parts, missing parts, notes, problems). `shutter inspect` prints it.
- A family declares its **parts** and nothing about memory. A `Part` (families/base.py)
  says which files hold a component (denoiser, text encoder, decoder, tokenizer), how its
  tensors are named, how many repeated blocks it has, and how big a block is.
- The engine owns the memory steps, and works on parts rather than models:
  `core/parts.py` turns part sizes plus a stage's working memory into a peak for each
  memory step (resident, split, staged, stream_wide, stream_tight). `core/stream.py` streams any
  part's block stack from the SSD. `core/text_encode.py` runs a T5-style prompt encoder one block
  at a time. So a new family gets all five steps without new memory code. It only provides
  its part sizes and per-stage working memory (for Wan: families/wan/spec.py).
- `Family.check_memory(ckpt, total_bytes)` gives the reasons a Mac cannot run a checkpoint
  even at the lowest step. The catalog, `shutter inspect` and `shutter download` use it.
- Each family brings its own converter into a Shuttrr build (`shutter convert`, for Wan
  families/wan/convert.py) and an adapter that runs the build (adapters/wan.py).
  docs/WAN.md covers what runs and what is refused.

### core/engine.py
- `Engine(config)`: owns store, bus, guard, pool, cache, registry and the worker thread.
  `submit(JobRequest) -> Job`, `cancel(id)`, `get(id)`, `list()`, `system()`, `start()`,
  `stop()`. Worker loop: take next queued job, `adapter.estimate`, guard check (else
  `waiting_memory`, re-check every `memory_poll_s`), take GpuLock, `adapter.run`, record
  result with timings and peak memory, publish events.

### core/registry.py
- `Registry(config)`: model id -> (adapter class, model dir, options) from
  `~/.shutter/config.json` plus built-in adapter families. `adapter_for(model_id)`.

### server/api.py
- The routes and JSON shapes are in docs/API.md: system, models, memory, settings, jobs (submit,
  list, get, cancel, output download), files (uploads), clients, events (SSE), the OpenAI Images
  shim (`server/openai_images.py`) and `GET /` (dashboard).

### The request
- `JobRequest` (types.py): model, prompt, negative_prompt, width, height, frames, duration_s, fps,
  steps, guidance, seed, inputs {role: path}, output, priority, params. `Adapter.prepare(request)`
  mirrors the core fields into params for adapters that read their options there. Old top-level
  `image`/`audio` go into `inputs`; the old `framing` key goes into `params.framing`.

### addons/reels (optional)
- `Framing` and `framing_of(request)` (from `params.framing`); `upscale.py`: ffmpeg upscale to
  the edit size with optional per-frame zoom cancel; `planner.py`: `Node`, `plan_placement(shots,
  nodes)` (shot-parallel: chains stay on one node, chains spread over nodes by free memory and
  speed; the interface an exo bridge would call). Only the LongCat adapter uses them.
