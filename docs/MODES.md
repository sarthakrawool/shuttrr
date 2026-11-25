# Modes

Two modes, chosen per job (`params.mode`, `params.efficiency`) with engine-wide defaults
(`default_mode`, `default_efficiency` in config.json, changed through `/v1/settings`).

## Fast
Optimise for time. Use every lossless speed-up, keep weights resident and warm between
jobs. Memory stays healthy by construction: before a job the planner takes
`budget = available - headroom` and, if the fast plan's estimated peak does not fit, steps
down the low-memory ladder just far enough to fit (and says so in the job's `stage`/notes).
It never starts a job whose estimate exceeds the budget.

## Low memory, with an efficiency slider (0.0 .. 1.0)
The slider trades time for memory along a ladder. Every rung is bit-identical to the
reference output unless marked otherwise. The slider value picks the highest rung whose
`at` is <= the value.

| at | id | shown as | what changes |
|---|---|---|---|
| 0.00 | `resident` | Full speed | same as Fast: lossless speed-ups, DiT resident |
| 0.25 | `split` | Slightly lower memory | each DiT block runs as 3 stages with an eval between (`LC_BLOCK_SPLIT=1`) |
| 0.50 | `staged` | Lower memory | + FFN in 4 frame chunks (`LC_FFN_CHUNKS=4`), and the DiT is released before the VAE decode (staged loading; next job reloads it) |
| 0.75 | `stream_wide` | Very low memory | + DiT block weights streamed from SSD, window of 8 blocks resident |
| 1.00 | `stream_tight` | Lowest memory | + window of 2 blocks resident: lowest memory, slowest |

Optional and never chosen by the modes, because it is not bit-identical: `vae_bf16` (half the decode
memory, 46.6 dB vs the fp32 decode). The one exception is set-up on a Mac with less than 24 GB: `shutter
download` writes `vae_bf16: true` into that model's options, because the full decode cannot fit there, and
says so (docs/INSTALL.md).

## Contract
- Python: `shutter/core/modes.py` exports `LADDER` (list of dicts: `at`, `id`, `label`,
  `detail`), `rung_for(efficiency) -> dict`, `resolve_mode(params, defaults) -> ModeChoice`
  and `fit_to_budget(estimates_by_rung, budget_bytes, start_rung) -> rung`. Adapters map a
  rung id to their own switches and report per-rung `MemoryEstimate`s.
- Measured numbers per model and rung live in `~/.shutter/measured.json`
  (`{model_id: {size: {rung_id: {"peak_bytes":..., "seconds":..., "identical": true}}}}`),
  written by the test chain; estimates prefer measured values over formulas.
- HTTP: `GET /v1/settings` -> `{"mode": "fast"|"low_memory", "efficiency": 0..1,
  "ladder": [{"at","id","label","detail","peak_bytes"?,"seconds"?}]}`;
  `POST /v1/settings {"mode"?, "efficiency"?}` validates, persists to config.json, returns
  the same shape and publishes a `settings.changed` event. At submit, the daemon copies the
  defaults into `request.params` when the job did not set `mode` / `efficiency`.
- A finished job's `result` carries `mode`, `rung` and, when Fast had to step down,
  `stepped_down_from`.

## Opt-in speed options (not on the ladder)

Independent of mode and rung, a job can ask for techniques that are **not bit-identical** and are never chosen
automatically: `params.speed = {"block_sparse": {...}, "step_cache": {...}}` (block-sparse self-attention and
TeaCache-style step caching; details, measured quality and limits in docs/KERNELS.md, "Opt-in speed techniques").
A job that uses any of them reports `identical: false` and a `speed` section in its result (what was on, what was
refused and why, chunk pairs skipped, steps skipped). They are refused (the job runs without them, with a note) on
the SSD streaming rungs.
