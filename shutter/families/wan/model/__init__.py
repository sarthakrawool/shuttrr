"""Wan 2.1 / 2.2 model code for MLX, vendored from Blaizzy/mlx-video (MIT, commit 87db56a; see LICENSE.mlx-video).

Changes against upstream:
* vae.py from upstream PR #38: the Wan 2.1 VAE decodes T latent frames into 1 + 4(T-1) frames, like the reference
  (upstream main decodes 4T).
* ltx_tiling.py (upstream ltx_2/video_vae/tiling.py) from upstream PR #47: 32-frame temporal tiles, so no conv input
  reaches the size where MLX's Metal conv2d returns zeros (ml-explore/mlx#3979).
* dit.py is upstream wan_2.py; imports point inside this package.
Nothing else in the model math is changed. Shuttrr's own code (loading, memory steps, streaming, LoRA, pipeline) lives
one level up in shutter/families/wan/.
"""
