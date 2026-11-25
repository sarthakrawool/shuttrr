# shutterd

The always-on part of Shutter, in Rust: the HTTP API, the job store (sqlite), the queue,
the memory guard, the GPU locks, the event stream and the watchdog. The model code runs in
a Python worker child process (`python3 -m shutter.worker`), started when a job needs it
and stopped when idle. See `../docs/WORKER_PROTOCOL.md` for the split and the protocol.

Release build on Linux x86_64: a 2.3 MB binary, about 4.5 MB RSS when idle.

## Build and test

```sh
cd daemon
cargo build --release          # target/release/shutterd
cargo test                     # unit + integration tests (needs python3)
```

The integration tests start the real daemon on a free port with a fake worker
(`tests/fake_worker.py`). `tests/real_worker.rs` runs the real Python worker with the
built-in `dummy` model, and is skipped when `shutter/worker.py` cannot be imported.

Release profile: `opt-level = "z"` (smallest binary; the daemon spends its time waiting,
so speed is not the concern), `lto`, `codegen-units = 1`, `panic = "abort"`, `strip`.

Dependencies: tokio (current-thread runtime, minimal features), axum (http1 only),
serde_json, rusqlite (bundled SQLite), libc, regex-lite, futures-core. tempfile is used by
the tests only. HTTP client calls (CLI, oMLX check) go through a small hand-written
HTTP/1.1 client over `TcpStream` (`src/httpc.rs`).

## Run

```sh
shutterd serve [--host 127.0.0.1] [--port 8484]   # --port 0 picks a free port
```

It prints one line, `shutterd 0.1.0 on http://127.0.0.1:8484  (home ...)`, then logs to
stderr only. SIGTERM or SIGINT: the running job is put back to `queued`
("interrupted by shutdown; it will run again"), the worker is sent `shutdown` and killed
after `shutdown_grace_s`, and the process exits with code 0. A bind to a non-loopback
address is refused (exit code 2) unless `api_key` is set.

CLI (talks HTTP to the daemon; `--json` prints the raw answer, `--url` or `SHUTTER_URL`
picks the daemon, `SHUTTER_API_KEY` sends a key):

```sh
shutterd submit --model dummy --prompt "a cat" --param steps=3 --wait [--mode low_memory --efficiency 0.5]
shutterd settings [--mode fast|low_memory] [--efficiency 0..1]   # no flags: show; flags: change and save
shutterd status [ID] [--limit N] [--state done,failed]
shutterd cancel ID
shutterd models | system | pause | resume | unload [KEY] | version
```

## Config

The same file as the Python engine: `$SHUTTER_HOME/config.json` (default
`~/.shutter/config.json`). The daemon reads it, and writes it only for `POST /v1/settings`
(a read-modify-write of `default_mode` / `default_efficiency` through a temp file and a
rename, keeping every other key, known or not; a file that is not a JSON object is left
alone and the POST fails with 500). Unknown keys are ignored. A value of
the wrong type falls back to its default with a warning on stderr. Paths may use `~`, and
relative paths are resolved against the home directory.

Keys shared with the Python reference (same defaults): `home`, `host` (127.0.0.1), `port`
(8484), `db_path`, `cache_dir`, `work_dir`, `output_dir`, `keep_work_dirs`,
`gpu_lock_path`, `gpu_lock_dirs`, `busy_process_pattern`, `headroom_gb` (null: automatic,
an eighth of total memory clamped to 2..8 GB; a number always wins), `default_mode`
(`fast`), `default_efficiency` (0.5),
`swap_growth_mb` (256), `max_fraction`, `memory_poll_s` (15), `idle_unload_s` (900),
`omlx_url`, `omlx_api_key`, `protected_process_pattern`, `protected_process_min_gb` (20),
`api_key`, `dummy_model`, `models`.

Keys added by the daemon:

| key | default | meaning |
|---|---|---|
| `worker_cmd` | `["python3", "-m", "shutter.worker"]` | worker argv (a string is split on whitespace) |
| `worker_cwd` | the repo root found from the executable, else unset | worker working directory |
| `cancel_grace_s` | 30 | time a running job has to stop after `cancel` before the worker is killed |
| `shutdown_grace_s` | 5 | time the worker has to exit after `shutdown` |
| `watch_interval_s` | 2 | watchdog period while a job runs (swap growth, critical pressure) |
| `sample_interval_s` | 5 | period of `memory.sample` events while a job runs |
| `sse_heartbeat_s` | 15 | `: heartbeat` comment period on `/v1/events` |
| `worker_start_timeout_s` | 120 | time the worker has to send `ready` |
| `estimate_timeout_s` | 120 | time the worker has to answer `estimate` |
| `omlx_timeout_s` | 1.5 | timeout of the read-only oMLX GETs |
| `static_dir` | see below | where `/` and `/static/*` are served from |
| `test_probe_file` | unset | TESTS ONLY: read memory samples from this JSON file instead of the OS |

`/` serves `<static_dir>/dashboard.html` and `/static/<path>` serves
`<static_dir>/static/<path>`, with content types and no way out of the directory. The
static directory is taken from the `SHUTTER_STATIC` environment variable, then
`static_dir`, then `<repo>/shutter/server` found from the executable's location, then
`<exe>/../Resources/server` (an app bundle).

The worker gets `SHUTTER_HOME`, `PYTHONUNBUFFERED=1` and `SHUTTER_DAEMON_PID` in its
environment and runs in its own session and process group, so a kill reaches every process
it started.

## How it differs from the Python reference

- The model code is in a child process, so idle unload (`idle_unload_s`) ends the worker
  process and returns every byte. The Python engine evicted warm pool entries. Idle unload
  also runs while jobs wait for memory, because a job's memory estimate is asked once and
  cached for the job.
- Admission order: busy-render check, then the worker's estimate, then the memory guard,
  then the flock, then the dir locks. The Python engine asked for the estimate first.
  The guard checks `need = estimate.peak`: the worker's estimate must already subtract
  what it holds warm. If the guard refuses and the worker holds warm components, the
  daemon sends `release` (all of them) and checks once more. The Python engine released
  only components that belonged to other models.
- Modes (docs/MODES.md): `GET /v1/settings` returns `{mode, efficiency, rung, ladder,
  measured_for}`; the ladder is a built-in copy of `LADDER` in `shutter/core/modes.py`
  (a pytest checks they match), with `peak_bytes` / `seconds` / `identical` filled in from
  `<home>/measured.json` when present (a missing or bad file is ignored). `POST
  /v1/settings {mode?, efficiency?}` validates (400), saves and publishes
  `settings.changed`. At submit the defaults are copied into `request.params` when the job
  did not set `mode` / `efficiency`. `/v1/system` adds `settings {mode, efficiency, rung}`,
  and `guard.headroom_bytes` is the effective value, with `guard.headroom_auto`.
- `/v1/models` is built from the config (the built-in `dummy` model plus `models`).
  While a worker is running, its `models` answer fills in `kind`, `installed` and `error`.
- `/v1/system` reads the pool from the worker (`stats`, 0.5 s timeout, or the last
  answer). With no worker, the pool is empty. It adds `"engine": "shutterd"` and a
  `worker` object (`running`, `pid`, `version`, `started_at`, `uptime_s`). `version` is
  the daemon's version.
- A failed job's `result.log_tail` is one string: up to 200 lines of the worker's stderr
  and `log` events since the job started, plus any stdout lines that were not protocol.
- Extra events: `worker.started`, `worker.stopped`, `job.log` (worker `log` lines),
  `model.unloaded`, `job.requeued`. `job.started` also carries the worker `pid`.
- When the memory probe fails (`vm_stat` or `sysctl` broken), admission waits with
  "Cannot read this Mac's memory state ...". The Python engine raised an error. During a
  run, a failed sample is skipped.
- `duration_s: true` is rejected (Python accepted booleans as numbers).
- New failure messages, each with `log_tail`: a worker that exits before `ready` gives
  "The model worker could not start: <the worker's `log` error, a Python exception line, or
  the last meaningful stderr line>. Set worker_cmd in config.json to a Python that has numpy
  and mlx." (the stderr tail and exit status go in `log_tail`); "the worker stopped unexpectedly (exit
  code N): <last stderr line>"; "the worker did not answer the memory estimate within N s
  and was stopped"; and, for a cancel that hit the grace period, "cancelled while running;
  the worker did not stop within N s and was stopped".
- A `failed` event with `memory_refused: true` puts the job back to `waiting_memory`.
  After 3 such refusals the job fails, as in the Python engine.
- A worker that closes its stdout but keeps running is killed after 2 s.

## Not verified on macOS

The code type-checks for `aarch64-apple-darwin` (`cargo check`), but it has only run on
Linux. Things to check on a Mac:

- `MacProbe`: it runs `sysctl -n hw.memsize hw.pagesize`, `vm_stat`,
  `sysctl -n vm.swapusage` and `sysctl -n kern.memorystatus_vm_pressure_level`. Its
  parsers are tested with captured macOS output.
- `ps -axo pid=,rss=,command=`.
- `setsid` and `killpg` on the worker process.
- `flock` against other tools' locks.
