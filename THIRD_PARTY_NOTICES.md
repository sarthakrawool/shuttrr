# Third-party notices

Shuttrr's own code is MIT licensed (see `LICENSE`). This file lists what ships inside this repository or the
Shuttrr app, and what the installer fetches on the user's Mac. Apart from the vendored mlx-video code below,
nothing from these projects is modified.

## Bundled in this repository and in Shuttrr.app

| What | Where | Licence |
|---|---|---|
| Fraunces (font) | `app/Shutter/Resources/Brand/`, `shutter/server/static/brand/` | SIL Open Font License 1.1 (`OFL.txt` next to the files) |
| DM Sans (font) | same folders | SIL Open Font License 1.1 |
| JetBrains Mono (font) | same folders | SIL Open Font License 1.1 |
| Inter (font) | `shutter/server/static/fonts/Inter-Variable.woff2` | SIL Open Font License 1.1 |
| Geist Mono (font) | `shutter/server/static/fonts/GeistMono-Variable.woff2` | SIL Open Font License 1.1 |

| [mlx-video](https://github.com/Blaizzy/mlx-video) Wan model code (vendored, adapted) | `shutter/families/wan/model/` | MIT, Copyright (c) 2025 Prince Canuma (`LICENSE.mlx-video` next to the files) |

The Wan model code in `shutter/families/wan/model/` is the only vendored third-party source; it was adapted
for Shuttrr (bounded-memory VAE decode, loader changes) and keeps its MIT notice. The Metal kernels in
`shutter/kernels/` and the app's Swift code are Shuttrr's own.

## Installed on the user's Mac by `scripts/bootstrap.sh` (not shipped in the app)

| What | Why | Licence |
|---|---|---|
| [MLX](https://github.com/ml-explore/mlx) | runs the models on Apple silicon | MIT |
| [uv](https://github.com/astral-sh/uv) | makes a Python only when the Mac has no Python 3.11 to 3.13 | MIT or Apache-2.0 |
| ffmpeg, via [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) | muxes video and audio; used only when the Mac has no ffmpeg | the ffmpeg build is LGPL-2.1+ / GPL depending on the build; imageio-ffmpeg is BSD-2-Clause |
| numpy, starlette, uvicorn, python-multipart | the engine and its HTTP server | BSD-3-Clause |
| huggingface_hub, transformers, safetensors | model download and the LongCat text encoder | Apache-2.0 |
| librosa, Pillow, imageio, ftfy, sentencepiece, pyloudnorm | audio, image and text handling for LongCat | ISC, MIT-CMU (HPND), BSD-2-Clause, Apache-2.0 (ftfy), Apache-2.0, MIT |

## Rust daemon (`daemon/`, not shipped in v0.1)

Built from crates.io dependencies listed in `daemon/Cargo.toml` (tokio, axum, serde_json, rusqlite with bundled
SQLite, libc, regex-lite): MIT or Apache-2.0; SQLite is public domain.

## Model weights

Shuttrr does not include model weights. LongCat-Video-Avatar weights (and the MLX port in the model repo's
`code/` folder) are downloaded from Hugging Face on request and are covered by their own licence on the model
card: https://huggingface.co/SarthakRawool/LongCat-Video-Avatar-1.6-MLX-8bit-LowMem and the upstream
LongCat-Video-Avatar model card from Meituan. Check those terms before commercial use.
