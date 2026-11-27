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

## MiniMax-H3-Image weights

The weights this code runs are not part of Shuttrr. MiniMax-H3-Image is licensed under the MiniMax-H3-Image Community
License Agreement, Copyright © 2026 MiniMax. All Rights Reserved. Converted builds made by `convert.py` carry a
NOTICE file saying so and that the files were modified.
