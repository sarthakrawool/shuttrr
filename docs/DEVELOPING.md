# Developing Shuttrr

Shuttrr is a Mac-only inference engine for large video and image diffusion models (LongCat
Avatar first) on MLX and Metal, made to plug into ComfyUI (`integrations/comfyui`), OpenAI
Images clients and scripts. It runs one heavy job at a time, keeps models warm between jobs, and above
all never pushes the Mac into swap or competes with the chat model you are already running.

## Quick start

```bash
scripts/install.sh                # engine in ~/.shutter/venv, Shuttrr.app in ~/Applications
shutter doctor                    # is this Mac ready?
shutter download longcat-avatar-1.6-8bit      # 38 GB from Hugging Face (or press Download in the app)
shutter serve                     # http://127.0.0.1:8484 : API + dashboard (the app starts this itself)
shutter submit --model dummy --param steps=5 --wait        # try it without a model
shutter submit --model longcat-avatar-1.6-8bit --image start.png --audio vo.wav \
  --prompt "a woman talking at her desk" --wait
shutter status | shutter system | shutter models | shutter cancel ID
```

Full install guide, memory table and the 16 GB notes: docs/INSTALL.md. To use weights you already have, add
them to `models` in `~/.shutter/config.json`:
`{"longcat-avatar-1.6-8bit": {"adapter": "longcat-avatar", "dir": "~/weights/...", "options": {}}}`.

## Rules it enforces

1. **One heavy job at a time.** One GPU worker thread. Before each job it takes the flock
   `gpu_lock_path` *and* the shared `mkdir` lock dirs in `gpu_lock_dirs` (the `gpulock.sh`
   convention), and gives both back afterwards, also on failure or cancel. It never removes
   a lock it did not create, even a stale one: it reports the holder instead.
2. **Never into swap.** A job starts only when its adapter's memory estimate plus
   `headroom_gb` (0 unless set) fits in available memory, pressure is not critical, no protected service (oMLX on
   port 8000, big `omlx|splash|strata` processes) holds a model, and no other render
   (`busy_process_pattern`) runs. Otherwise the job waits in `waiting_memory` with the
   reason. During a run, swap growth over `swap_growth_mb` or critical pressure stops the job
   ("stopped to protect the Mac") and unloads every model.
3. **Hands off other apps.** Shuttrr only reads from oMLX (GET requests); it never unloads
   or changes anything in oMLX, Splash or Strata.
4. Models stay loaded for `idle_unload_s` (900 s) after the last job; 0 unloads at once.

## API

Full reference with JSON shapes: docs/API.md. `GET /v1/system`, `GET /v1/models`, `GET /v1/catalog`, `POST|DELETE /v1/catalog/{id}/download`,
`GET /v1/memory`, `POST /v1/jobs`, `GET /v1/jobs?limit=&state=`, `GET|DELETE /v1/jobs/{id}`,
`GET /v1/jobs/{id}/output`, `POST /v1/files`, `GET /v1/clients`, `POST /v1/queue/pause|resume`,
`POST /v1/pool/release`, `GET|POST /v1/settings`, `GET /v1/events` (SSE),
`POST /v1/images/generations|edits` (OpenAI Images shim), `GET /` (dashboard). Loopback only by default; binding elsewhere
needs `api_key` in config, and remote clients then send `Authorization: Bearer <key>`.

## Layout

```
shutter/core/      engine (worker loop), queue (sqlite jobs, event bus), memory (probes,
                   guard, GpuLock), dirlock (mkdir locks), pool, cache, stream, registry, config
shutter/adapters/  base contract, longcat (LongCat Avatar), dummy + dummy-image (tests, demos)
shutter/addons/    optional add-ons read through params (reels: framing, crop, upscale, planner)
shutter/server/    api.py (Starlette), openai_images.py (shim), dashboard.html
integrations/      comfyui/ node pack
shutter/cli.py     the `shutter` command
app/Shutter/       SwiftUI menu bar app (talks to the same HTTP API)
docs/              ARCHITECTURE.md, API.md, WORKER_PROTOCOL.md, MODES.md, INTEGRATIONS.md
```

## Testing the memory watchdog on a real Mac

`SHUTTER_TEST_ADMIT_ANY=1` (environment, test only) makes the guard admit any request whatever its estimate, so an
oversized request (for example Wan 2.1 1.3B at 480x832, 81 frames, `params.rung = "resident"` and a large
`params.budget_bytes`) really runs and the watchdog has to stop it. Measured on a 16 GB M5 MacBook Pro (0.2.3): it
freed caches when swap grew 1.1 GB during the denoise, the decode then pushed swap past +2 GB, the job was stopped,
and because the decode is one long GPU call the engine recorded the job as stopped and exited with code 75 (the app
restarts it). Swap went back down at once and the Mac stayed usable. Results: `shutter-ref/bench/v0.2.3/`.

