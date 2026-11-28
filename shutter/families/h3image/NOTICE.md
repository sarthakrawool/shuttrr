# Third-party code and weights in this folder

## minimax-h3-mlx (Apache License 2.0)

`model.py` adapts the DiT, rotary and ViT decoder code of **minimax-h3-mlx** by mrbizarro
(https://github.com/mrbizarro/minimax-h3-mlx, commit 7919020, files `minimax_h3_mlx/dit.py` and
`minimax_h3_mlx/video_vae.py`), licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0). Changes: image-only (no audio stream), ComfyUI / GGUF tensor names,
contiguous q/k/v rows in the DiT, the six PDD output heads, a float32 residual stream, the 2D image VAE decoder,
quantized loading and SSD block streaming.

## stable-diffusion.cpp (MIT License)

The sampling details (PDD sigmas and head blending, sequence layout and rotary positions, VAE latent statistics, the
PyTorch-compatible CPU noise) follow stable-diffusion.cpp (https://github.com/leejet/stable-diffusion.cpp, commit
532d2c1, `src/model/diffusion/minimax_h3.hpp`, `src/model/vae/minimax_h3_vae.hpp`, `src/core/rng_mt19937.hpp`),
Copyright (c) 2023 leejet, MIT License. The GGUF Q4_K / Q6_K decoders follow ggml (MIT License).

## ComfyUI-H3-Image-EasyFix (Apache License 2.0)

`easyfix.py` ports the Steering and Re-Noise Sampler of **ComfyUI-H3-Image-EasyFix** by PedroMarinhoDev
(https://github.com/PedroMarinhoDev/ComfyUI-H3-Image-EasyFix, commit a73c535, files `h3fix/steering.py`,
`h3fix/samplers.py`, `h3fix/common.py`), licensed under the Apache License, Version 2.0 (copy in
`data/LICENSE-EasyFix.txt`). `data/vmproj_all6.safetensors` (the steering table) is copied unchanged from that
repository's `data/` folder. Changes: rewritten for MLX and Shuttrr's own block loop (resident and SSD-streamed);
float32 residual stream, so no bf16 dither; noise from Shuttrr's CPU generator; Euler ancestral written out for the
rectified-flow case; ComfyUI node and resolution-selector code not used.

## MiniMax-H3-Image weights

The weights this code runs are not part of Shuttrr. MiniMax-H3-Image is licensed under the MiniMax-H3-Image Community
License Agreement, Copyright © 2026 MiniMax. All Rights Reserved. Converted builds made by `convert.py` carry a
NOTICE file saying so and that the files were modified.
