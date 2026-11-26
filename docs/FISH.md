# Fish Audio Drama 3 (text to speech)

Shuttrr runs Fish Audio Drama 3 through the `fish-drama3` adapter (`shutter/adapters/drama3.py`). The model code
is TensorFold's MLX port; Shuttrr adds what makes it fit and stay fast on a smaller Mac.

| What | Effect | Same audio? |
|---|---|---|
| Model stays loaded between requests | no ~41 GB reload per take | yes |
| Low-memory load, 1 GB MLX cache, unused parts dropped, windowed decode | peak about 44 GB instead of 59 to 65 GB | yes (measured: codes identical) |
| Reference voice encoded once, codes kept on disk | no re-encode per take | yes |
| Several takes in one request (`takes` or `seeds`) | one load, one reference | yes |
| Batched takes (`batch: true`, when the model code has it) | about 1.3 to 1.5x one take for four | same quality, not sample-identical |
| Smaller builds per memory step (`variants`) | 5-bit about 29 GB, 4-bit about 25 GB | no: only as good as the build |

| Experts read from the SSD through a RAM cache (stream_wide 24 GB, stream_tight 18 GB cache; Beta) | 8-bit peak 34.6 / 28.2 GB MLX instead of 42.7, about 3x slower (21 / 19 frames a second instead of 64) | yes (measured: tokens identical) |

The two lowest memory steps (Very low / Lowest memory) stream the routed experts of the same 8-bit build from the
SSD into an LRU cache (`expert_cache.py`, `TF_D3_EXPERT_CACHE_GB`): no extra download. On a 48 GB Mac Fast steps
down to them by itself, and the catalog says "Fits, about 3x slower". The cache size per step is the option
`stream_cache_gb` (`{"stream_wide": 24, "stream_tight": 18}`; 0 switches a step off; at least 8 GB). Measured on the
Mac Studio with the cache capped (`~/fishq/stream_test`), not yet run end to end in Shuttrr on a 48 GB Mac.
Streaming works with the vendored runtime only (not an external `tensorfold_src`, unless `stream_external`).

## Set up

```json
{"models": {"fish-drama3": {"adapter": "fish-drama3", "dir": "~/models/drama-3-mlx", "options": {
  "tensorfold_src": "~/models/drama3-mem/code/src",
  "variants": {"staged": "~/models/drama-3-mlx-gptq5g64", "stream_tight": "~/models/drama-3-mlx-gptq4g32"}
}}}}
```

`variants` is optional. With it, Low memory (or Fast, when the full build does not fit in free memory) uses the
largest build that fits without swap; the result says which build ran and whether it is lossless.

## Request

```json
{"model": "fish-drama3", "prompt": "[excited] One line to speak.", "seed": 5,
 "inputs": {"audio": "/path/voice.wav"},
 "params": {"ref_text": "what the clip says", "takes": 2, "temperature": 0.75}}
```

The result has `output` (first take, 44.1 kHz mono WAV), `outputs` (all takes), per-take frames a second,
`build`, `lossless` and `warm`.

## Check on a Mac

`scripts/fish_check.py` runs a line cold, warm and as two takes, compares the audio with a WAV from the plain
command, and reports memory and swap. Run it under the shared GPU lock.
