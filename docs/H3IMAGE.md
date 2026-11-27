# MiniMax-H3-Image (text to image)

Model: [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image): a 20B diffusion transformer
(the "pruned" layout: the 13B AdaLN timestep path replaced by a sampled curve), conditioned on 50 layers of a
Qwen3-VL-32B text encoder, with a ViT image VAE. Six steps, no guidance.

**Licence: MiniMax-H3-Image Community License Agreement.** Not Apache/MIT. It does not cover the US, the EU, the UK
or South Korea (§I.5); commercial use is allowed below USD 1M annual revenue and not for hosted services without a
separate licence (§IV.1); a commercial UI must show "MiniMax-H3-Image" (§IV.2); outputs may not be used to improve
other models (§V.3). Redistributed weights need the licence text, a NOTICE file and a "modified" note (§III).
**Not in the catalog**: offering it to Shuttrr users is a decision for the project owner.

Code: `shutter/adapters/h3image.py` (adapter, registry id `h3-image`), `shutter/families/h3image/` (model code
adapted from minimax-h3-mlx, Apache-2.0, checked against stable-diffusion.cpp; converter; pipeline). See its
NOTICE.md.

## Build the model folder (no download)

From the files stable-diffusion.cpp uses (ComfyUI / leejet GGUFs):

    python -m shutter.families.h3image.convert \
        --dit minimax_h3_image_pruned-Q8_0.gguf \
        --te qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
        --vae minimax_h3_image_vae_fp16.safetensors --tokenizer <Qwen tokenizer folder> \
        --out ~/.shutter/models/MiniMax-H3-Image-Shuttrr

`--dit -` / `--te -` read a GGUF from stdin (`ssh host cat file.gguf | python -m ...`), so the source never has to
be on this disk. Mappings (exact up to the scales rounded to bfloat16):

| Source | Shuttrr build |
| --- | --- |
| Q8_0 (DiT) | MLX 8-bit group 32: value q -> q + 128, scale d, bias -128 d |
| BF16 matrices the release kept at 16 bits | BF16 |
| Q4_K (text encoder) | MLX 4-bit group 32: scale d * sc, bias -dmin * m (each Q4_K 32-group is affine) |
| Q6_K (text encoder v_proj / down_proj) | MLX 8-bit group 32 (re-quantized; Q6_K groups are 16 wide) |
| token embedding BF16 | BF16, only the prompt's rows are read |

The original bf16 text encoder (MiniMax-H3's `text_encoder/`, HF safetensors) also works: `--te <folder>`
(`--te-bits 4|8`). Writes use F_NOCACHE so a 40 GB conversion does not push other apps to swap.

Register it:

    "models": {"minimax-h3-image": {"adapter": "h3-image", "dir": "~/.shutter/models/MiniMax-H3-Image-Shuttrr",
                                    "name": "MiniMax-H3-Image"}}

## Request

`POST /v1/jobs` with `model`, `prompt`, `width` / `height` (multiples of 32, 256..2048, default 1024x1024), `seed`.
Params: `count` (1..8), `style` (`"3d-animated"`), `mode` / `efficiency` / `rung`. `steps`, `guidance` and
`negative_prompt` are not used (the model runs its own 6 steps, CFG 1); a note in the result says so. The seed gives
the same starting noise as `sd-cli --rng cpu` (PyTorch's CPU generator).

## Memory steps (all give the same image, bit for bit)

| Step | What is in memory while it denoises |
| --- | --- |
| resident, split | the whole DiT (~22 GB at 8-bit), kept warm between requests |
| staged | the whole DiT, freed before the decode |
| stream_wide / stream_tight | the 50-block stack read from the SSD, 8 / 2 blocks (0.43 GB each) at a time, plus ~1 GB |

The prompt encoder always runs one layer at a time (~0.37 GB); the VAE decoder (0.8 GB float16) runs in float32.
On Macs under 24 GB Fast mode uses stream_tight. Measured numbers: `shutter-ref/sync/h3-image/REPORT.md` in the
workspace.

## Not done yet

- reference pictures (`<Picture N>` editing): needs the Qwen3-VL vision tower and the VAE encoder
- LoRAs
- a catalog entry / Hugging Face build (licence decision first)
