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

Reading the experts from the SSD (as Shuttrr does for video models) is not used: every frame picks 8 of 256
experts in each of 40 layers, the early layers use almost all of them, and it came out 2 to 5 times slower.

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
