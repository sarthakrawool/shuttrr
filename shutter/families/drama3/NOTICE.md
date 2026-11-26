# Fish Audio Drama 3 runtime (vendored)

The Python files in this folder come from TensorFold, `src/tensorfold/families/drama3/` at commit
`8f3d4b33a770b1906131354566527503850f54bf` (https://github.com/drowzeys/TensorFold), licensed under the Apache
License 2.0 (`pyproject.toml` of that repository; its GitHub page lists MIT). Changes made for Shuttrr:

- imports renamed from `tensorfold.families.drama3` to `shutter.families.drama3`;
- low-memory switches read from the environment (`TF_D3_LOWMEM`, `TF_D3_CACHE_GB`, `TF_D3_DROP_LMHEAD`,
  `TF_D3_FREE_ENCODER`, `TF_D3_CODEC_WINDOW`, `TF_D3_EXPERT_BITS`, `TF_D3_EXPERT_CACHE_GB`), all off by default,
  which give the same audio with a lower memory peak, and loading of calibrated GPTQ expert builds;
- `expert_cache.py` (new, written for Shuttrr): the routed experts read from the SSD into an LRU cache when
  `TF_D3_EXPERT_CACHE_GB` is set, hooked into `model.py`'s low-memory load.

Third-party code inside, as stated by TensorFold's THIRD_PARTY_NOTICES.md:

- `codec_dac.py` is adapted from mlx-audio 0.5.7 (`mlx_audio/codec/models/fish_s1_dac/fish_s1_dac.py`),
  MIT License, Copyright (c) 2024 Prince Canuma. It ports Fish Audio's codec (fish-speech `modded_dac`).
- The backbone uses mlx-lm's `qwen3_5` model (MIT, Copyright (c) 2023 Apple Inc.), copied into `mlx_lm_models/`.
- The tower, frame embeddings, sampler and prompt layout follow SGLang-Omni's `sglang_omni/models/fishaudio_drama3`
  (revision 416a70c0, Apache-2.0); no SGLang-Omni source is included.

The model weights are not included. They are Fish Audio Materials under the Fish Audio Community License,
Version 1.0. Copyright (c) 39 AI, INC. All Rights Reserved. Built with Fish Audio.

`mlx_lm_models/` holds mlx-lm 0.32.0's `qwen3_5`, `qwen3_next`, `base`, `cache`, `gated_delta`, `pipeline`,
`switch_layers`, `activations` and `rope_utils` modules, unchanged (MIT License, Copyright (c) 2023 Apple Inc.,
text in `mlx_lm_models/LICENSE`), so the engine needs neither mlx-lm nor transformers.
