# Z-Image Turbo (text to image)

Model: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (6B DiT, Qwen3-4B prompt encoder,
Apache 2.0). Shuttrr downloads the mflux-community 8-bit build
([mflux-community/z-image-turbo-mflux-q8](https://huggingface.co/mflux-community/z-image-turbo-mflux-q8), 11.0 GB,
pinned to one revision, every file checked by size and SHA-256). Catalog id `z-image-turbo-8bit`.

Code: `shutter/adapters/zimage.py` (adapter), `shutter/families/zimage/` (model code vendored from mflux 0.24.0, MIT,
see its NOTICE.md; pipeline; LoRA).

## Request

`POST /v1/jobs` with `model`, `prompt`, `width` / `height` (multiples of 16, 256..2048, default 1024x1024), `steps`
(default 9), `guidance` (default 0: Turbo runs without guidance; above 1 uses `negative_prompt` and runs twice per
step), `seed`. Params: `count` (1..8 pictures, seeds seed, seed + 1, ...), `style` (`"3d-animated"`), `loras`
(`[{"path", "strength"}]`), `mode` / `efficiency` / `rung`. Output: PNG; more pictures sit next to it as
`<name>.s<seed>.png`, all listed in `result.outputs`. `GET /v1/models/z-image-turbo-8bit/request-options` gives the
sizes, limits, styles and an estimate (kind `"image"`).

## Memory steps (all give the same image, bit for bit)

| Step | What is in memory while it denoises |
| --- | --- |
| resident, split | the whole DiT (6.5 GB), kept warm between requests |
| staged | the whole DiT, freed before the decode |
| stream_wide / stream_tight | the DiT's 30-block main stack read from the SSD, 8 / 2 blocks at a time, plus 0.8 GB |

The prompt encoder always runs one layer at a time (layer 36 of Qwen3-4B is never run: Z-Image reads layer 35's
output). The VAE decodes block by block in row strips.

Measured on a 16 GB MacBook Pro M5 with the Shuttrr app running (engine footprint + app, 9 steps): 1024x1024 in
61 s at 3.1 GB, 768x768 in 34 s at 2.7 GB (stream_tight). See `shutter-ref/sync/image-family/REPORT.md` in the
workspace for the full runs.

## LoRAs

Applied at run time on top of the 8-bit weights (never merged): diffusers / PEFT (`lora_A` / `lora_B`), ComfyUI
(`lora_down` / `lora_up`, fused `attention.qkv`) and kohya (`lora_unet_...`) names. A LoRA costs its file size in
memory. In the app: New request > Advanced > Style LoRA and LoRA strength (0 to 2).
