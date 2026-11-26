# Shuttrr process split and worker protocol

```
 Swift app (menu bar + window)  ──HTTP──▶  shutterd  (Rust, always on, a few MB)
 CLI / dashboard / exo later    ──HTTP──▶     │  owns: HTTP API, job store (sqlite), queue,
                                              │  memory guard, GPU locks, events, watchdog
                                              │
                                   stdin/stdout JSON lines
                                              ▼
                                  shutter worker (Python + MLX, started on demand)
                                  owns: model code, adapters, model pool (warm weights),
                                  prompt/audio cache, SSD weight streaming, kernels
```

Why the split: the maths runs in MLX/Metal whichever language calls it, and the model
ports are Python. What benefits from Rust is everything that must be small, always on and
impossible to wedge: the daemon. Because the model lives in a child process, the daemon can
give all its memory back to the Mac with certainty by ending that process, and its
watchdog can stop a job the instant swap starts to grow.

## Rules for the daemon
- Exactly one worker process at a time, started lazily on the first job, command from
  config (`worker_cmd`, default `["python3", "-m", "shutter.worker"]`).
- Admission before `run`: busy-render process check, memory guard on the worker's
  estimate (protected services, critical pressure, need <= available; no reserve unless `headroom_gb` is set), flock GPU lock,
  mkdir dir locks. Otherwise the job is `waiting_memory` with a plain-language reason.
- Watchdog while a job runs (every 2 s): if swap grew more than `swap_growth_mb` since
  the job started, or pressure is critical: SIGKILL the worker's process group, fail the
  job with "stopped to protect the Mac: <reason>", release locks.
- Cancel: send `cancel`; if the worker has not answered within `cancel_grace_s` (30 s),
  kill it.
- Idle: `idle_unload_s` after the last job, stop the worker (`shutdown`, then kill after
  5 s). That returns every byte of model memory.
- Worker crash or EOF mid-job: job fails with the last stderr lines; locks released.
- Daemon restart: jobs left `running` are requeued.

## Messages (one JSON object per line, UTF-8)

Worker stdout carries ONLY protocol lines. Everything else (logs, prints from model code)
must go to stderr; the daemon keeps the last 200 stderr lines per job.

daemon -> worker
| op | fields | meaning |
|---|---|---|
| `hello` | | sent once after start |
| `estimate` | `id`, `request` (JobRequest dict) | ask for a MemoryEstimate |
| `run` | `id`, `job` (Job dict), `work_dir` | run the job |
| `cancel` | `id` | stop the running job at the next step |
| `release` | `key` (string or null = all) | drop warm components |
| `stats` | | ask for a `pool` event |
| `models` | | ask for a `models` event |
| `shutdown` | | exit cleanly |

worker -> daemon
| ev | fields |
|---|---|
| `ready` | `version`, `pid`, `protocol` (=1) |
| `estimate` | `id`, `peak`, `steady`, `disk_scratch`, `note`, or `error` |
| `progress` | `id`, `progress` (0..1), `stage`, and `step`/`steps` when the stage has a counter |
| `done` | `id`, `result` (must contain `output`, `seconds`, `peak_bytes`) |
| `failed` | `id`, `error`, `cancelled` (bool) |
| `pool` | `total_bytes`, `entries` [{`key`,`size_bytes`,`idle_s`,`pinned`,`residency`}] |
| `released` | `keys` |
| `models` | `models` [{`id`,`family`,`kind`,`dir`,`installed`,`error`?}] |
| `log` | `level`, `message` (optional; shown in the job log) |

Details fixed by the worker implementation (`shutter/worker.py`):
- `released` may carry `deferred` (keys in use by the running job); a second `released`
  follows when the job ends.
- `failed` may carry `memory_refused: true` (a model load was refused mid-run): the
  daemon should put the job back to `waiting_memory`, not fail it.
- A `cancel` for a run that has not started yields `failed` with `cancelled: true`; a
  cancel for an unknown id is ignored.
- EOF on stdin, `shutdown` or SIGTERM during a job: the job ends as `failed` with error
  "worker shutting down", then the worker exits 0. A start-up failure is a `log` event
  and exit 1 with no `ready`.
- `pool` entries also carry `in_use`; `log` may carry `id`.
- `pool` entries carry `residency`: `"resident"` or `"streamed"` (weights streamed from SSD,
  only a window resident). A missing value means `"resident"`. It feeds `GET /v1/memory`.
- `step`/`steps` on `progress` are parsed from the stage text (`"denoise 3/8"`,
  `"segment 1/2 denoise 3/8"`, `"Step 4 / 10"`; `shutter/core/stats.py`). The daemon parses
  the stage itself too, so a worker that omits them still feeds `throughput.running`.
- The `request` dict (in `estimate` and inside `job` for `run`) has the plain inference
  fields `model, prompt, negative_prompt, width, height, frames, duration_s, fps, steps,
  guidance, seed, inputs, output, priority, params` (docs/API.md). Input files are in
  `inputs` by role (`image`, `audio`, ...), already resolved to absolute paths. Before
  `estimate` and `run`, the worker calls `Adapter.prepare(request)`, which mirrors the core
  fields into `params` where `params` does not set them, on a copy. The `job.client` field
  is informational.
- The daemon sends an absolute `output` path in the job's request. The worker deletes
  `work_dir` after success unless the output is inside it.
- Progress never goes backwards, at most one event per 0.25 s, the last value always sent
  before `done`/`failed`.

Unknown ops/events are ignored by both sides (forward compatibility). The HTTP API keeps
exactly the JSON shapes of the Python reference server (`shutter/server/api.py`,
`tests/test_api.py`).
