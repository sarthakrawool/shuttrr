"""Fish Audio Drama 3 text to speech (MLX): Qwen3.5 MoE backbone, codebook tower and Fish codec.

Vendored into Shuttrr from TensorFold (https://github.com/drowzeys/TensorFold, commit 8f3d4b33, Apache-2.0) with
Shuttrr's low-memory switches (TF_D3_LOWMEM and friends, off by default) and the calibrated 5-bit/4-bit expert
builds; imports renamed from tensorfold.families.drama3 to shutter.families.drama3. See NOTICE.md in this folder.

Built with Fish Audio. The model weights are under the Fish Audio Community License v1.0 and are not part of this
code; Shuttrr downloads them separately after the person accepts that licence.
"""

from __future__ import annotations

# Not a lane family: one forward samples a whole audio frame (a main token and nine codes), so the
# text engine's lanes and drafting do not apply. Used by shutter/adapters/drama3.py.
MODEL_TYPES = ("fish_omni",)
TITLE = "Fish Audio Drama 3"
LANES = False
MODELS = ()
CONVERTED_FORMAT = "tensorfold-drama3-mlx-v1"


def is_checkpoint(model_dir) -> bool:
    """Whether ``model_dir`` holds a Drama 3 checkpoint (Fish's own or the converted MLX one)."""

    import json
    from pathlib import Path

    config = Path(model_dir) / "config.json"
    if not config.is_file():
        return False
    return json.loads(config.read_text()).get("model_type") in MODEL_TYPES
