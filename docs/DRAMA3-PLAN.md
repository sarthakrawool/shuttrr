# Fish Audio Drama 3 in Shuttrr: licence check and plan (draft, 2026-11-27)

Branch `drama3` (local only, not pushed). Nothing uploaded or released.

## 1. Licences

**Model weights: Fish Audio Community License v1.0** (LICENSE in the converted folder; the Hugging Face card's
metadata says `fish-audio-research-license`, but its text and the LICENSE file say Community License v1.0).

- Quantized builds are "Derivative Works" and may be reproduced and distributed: for research / non-commercial use,
  and commercially while the distributor's annual revenue is under USD 500,000 (s. III(a)). Others already do this
  publicly (mlx-community/fish-audio-drama-3-4bit and -3bit, ggml-org/drama-3-GGUF, Fish's own drama-3-int4).
- Conditions when we publish a build on Hugging Face: ship the LICENSE text, a NOTICE file with
  "These Fish Audio Materials are licensed under the Fish Audio Community License, Version 1.0. Copyright © 39 AI,
  INC. All Rights Reserved." plus a statement of how we changed it (requantization, conversion), and show
  "Built with Fish Audio" prominently in the app/README/model card. No Fish Audio logo, no implied endorsement;
  naming like "based on Fish Audio Drama 3" is allowed.
- Use restrictions (s. IV(b)) bind anyone using it: no cloning an identifiable person's voice without consent, no
  impersonation/fraud, no sexual content with minors or real people without consent, no watermark removal, and no
  using outputs to train other speech models. Because Shuttrr makes voice cloning available to its users, we must
  bind them to these terms "by enforceable terms": a click-through (accept the licence + the voice-consent rule)
  before the download and before the first request with a reference clip.
- Hosted Service (s. III(c)): offering it as an API/website to third parties needs a separate licence. Shuttrr runs
  on the person's own Mac; their local engine API is theirs, so this does not apply to the app. It would apply to
  anyone who exposes Shuttrr's API to other people as a service: say so in the terms.
- Users above USD 500K revenue need their own licence from Fish Audio for commercial use (their obligation; the
  click-through says it).

**Runtime code: TensorFold** (github.com/drowzeys/TensorFold, pyproject says Apache-2.0, GitHub lists MIT): either
way it can be vendored into Shuttrr (MIT app) with its notice. Inside it: codec port from mlx-audio (MIT), and the
backbone uses mlx-lm's qwen3_5 (MIT). TensorFold itself chooses not to redistribute weights ("for local use only"),
which is their policy, not a licence limit on us.

**Fallback if the user prefers not to host weights:** Shuttrr downloads Fish's original fishaudio/drama-3 (not
gated) and converts it locally to the 8-bit MLX build (TensorFold's converter, seconds on a Mac Studio). The
calibrated 5-bit build cannot be made locally in reasonable time (GPTQ calibration), so that path gives 8-bit only
(47.5 GB peak: 64 GB Macs and up).

## 2. Plan

- **Runtime:** vendored, done on this branch: `shutter/families/drama3/` (TensorFold 8f3d4b33 + the low-memory
  switches, imports renamed) and `shutter/families/drama3/mlx_lm_models/` (the 9 mlx-lm 0.32.0 modules it uses,
  unchanged), so the engine needs no TensorFold, no mlx-lm and no transformers; `tokenizers` (already in the engine
  environment) is the only new dependency. TensorFold's own runtime tests (10) pass on Shuttrr's MLX 0.32.5.
- **Builds (Hugging Face, after approval):**
  - `Fish-Audio-Drama-3-MLX-5bit` (calibrated GPTQ 5-bit experts, group 64): 25.0 GB on disk (24.4 GiB),
    generation peak 35.3 GB, 62 fps (~2.9x real time) on the Mac Studio M5 Max; quality equal to 8-bit in the
    21-take test (WER 3.5% vs 3.6%, speaker similarity 0.781 vs 0.783). Default.
  - 4-bit GPTQ (group 32) as the lowest memory step of the same entry: 23.0 GB on disk, 33.4 GB peak (only 2 GB
    less, so last resort).
  - `Fish-Audio-Drama-3-MLX-8bit`: 43.2 GB on disk, 47.5 GB peak, 60 fps.
  - Streamed 5-bit (expert cache 12 GB): 21.5 GB peak, ~21 fps (about real time). Its code lives in a separate
    TensorFold copy (`~/fishq/src_batch`) and is not vendored yet: phase 2, for 32 GB Macs.
- **Fit (measured peaks only, Mac Studio M5 Max 128 GB; usable = total - ~3 GB macOS):**

  | Mac | 5-bit (35.3 GB) | with streamed 5-bit (21.5 GB, phase 2) | 8-bit (47.5 GB) |
  |---|---|---|---|
  | 16 / 24 GB | Doesn't fit | Doesn't fit | Doesn't fit |
  | 32 GB | Doesn't fit | Fits but barely | Doesn't fit |
  | 48 GB | Fits but barely | Fits but barely | Doesn't fit |
  | 64 GB | Recommended | Recommended | Fits but barely |
  | 96 / 128 GB | Recommended | Recommended | Recommended |

  All measured on one 128 GB Mac only: smaller sizes are "not tested on a Mac this size yet".
- **App:** when the model is an audio model the New request sheet shows Text, Reference voice (a clip), "What the
  clip says", Takes (1 to 8), "Split long text into short parts" (on: about 120 characters per part, the user's
  chosen default; off: 300), and hides image/size/frames. Requests rows of audio jobs get a play button and the
  file; more takes are listed under it. First use shows the licence + consent click-through.
- **Not done yet on this branch:** catalog entries + audio fit rule, the click-through, the audio form and player,
  phase 2 streaming, and an end-to-end run through the Shuttrr engine on the Mac Studio.

## 3. Real-model check

Not run: on 2026-11-27 00:00 IST the Mac Studio had a Splash model loaded (58 GB resident, memorystatus level 36);
gpulock refuses below level 78, and the rules say not to unload the user's models without asking. The vendored code
is byte-identical to the patched copy that produced the measured numbers apart from import paths, but the engine
uses MLX 0.32.5 where those runs used 0.32.3, so one real run through Shuttrr is still needed.
