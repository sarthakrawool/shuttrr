# Wan video models

Shuttrr runs the Wan 2.1 and Wan 2.2 video models from Wan-AI (Alibaba), their fine-tunes and their LoRAs.
Shuttrr supports the family rather than a list of files. Any checkpoint whose tensors have Wan's shapes is
recognised, whoever made it. Anything Shuttrr cannot run is refused before loading, with the reason in plain words.

Wan runs from a **Shuttrr build**: a folder made by `shutter convert` that holds the video model in 8-bit,
the umT5 text encoder in bf16, the tokenizer and the VAE. You can make one from a Hugging Face download, a ComfyUI
file or a fine-tune. `shutter download` does the same for the models in the catalog.

## What runs

| Model | Task | Video model in 8-bit | Built folder | Download (original, approx.) | Smallest Mac (estimate) |
|---|---|---|---|---|---|
| Wan 2.1 T2V 1.3B (`wan2.1-t2v-1.3b`) | text to video, 480p | 1.4 GB | ~13 GB | ~17 GB | 8 GB |
| Wan 2.2 TI2V 5B (`wan2.2-ti2v-5b`) | text to video, or image to video, 720p | 5.0 GB | ~22 GB | ~34 GB | 10 GB |
| Wan 2.2 T2V A14B (`wan2.2-t2v-a14b`) | text to video, two 14B experts | 2 x 14.3 GB | ~40 GB | ~126 GB | 9 GB |
| Wan 2.2 I2V A14B (`wan2.2-i2v-a14b`) | image to video, two 14B experts | 2 x 14.3 GB | ~41 GB | ~126 GB | 9 GB |
| Wan 2.1 T2V 14B | text to video | 14.3 GB | ~26 GB | — | — |

The built sizes come from `WanShape.dit_bytes(8)` in `shutter/families/wan/spec.py`, plus 11 GB for the text
encoder and the VAE. The smallest Mac is the lowest memory size that passes the family's memory check
(`shutter.families.family("wan").check_memory`) at the lowest memory step and the smallest size (320 x 512,
17 frames), with 4 GB kept free. **These are formulas, not measurements.** At the lowest step a 14B expert
streams from the SSD two blocks at a time, so it fits in little memory but runs slowly. Larger videos need more
memory. The catalog shows these numbers, and `shutter inspect` does the same check for any checkpoint.

The Wan 2.1 14B is not in the catalog. It still converts and runs if you have it.

**Fine-tunes.** A fine-tune with the same shapes as one of the models above is treated as that model, with its
default steps, guidance and shift. A Wan-architecture model of another size runs with Wan's general defaults
(50 steps, guidance 5, shift 5), and `inspect` says so. Check its model card for its own settings.

**LoRAs.** A Wan LoRA `.safetensors` is recognised in the common namings (official / ComfyUI `diffusion_model.`,
kohya `lora_unet_`, PEFT `lora_A` / `lora_B`). You can use it in two ways:

* at run time, per request: `params.loras = [{"path": "...", "strength": 1.0, "expert": "both" | "high" | "low"}]`.
  The 8-bit weights stay unchanged, so you can change LoRAs between requests;
* merged into a build: `shutter convert ... --lora style.safetensors:0.8`. The LoRA is added before
  quantizing, and the build records which LoRAs it holds.

`inspect` on a LoRA says which Wan size it was trained for (from its width).

## What is refused, and why

| Checkpoint | Why |
|---|---|
| Wan 2.1 image-to-video, first-last-frame, or anything with a CLIP image branch (`img_emb.*`) | The CLIP branch is not supported yet. Wan 2.2 image-to-video has no CLIP branch, and it runs. |
| VACE (`vace_*`) | Video editing with control inputs, not supported yet. |
| Fun / control / camera models | They need control inputs that are not supported yet. |
| Wan S2V, Animate, audio-driven models | A different family. LongCat avatar models run with the LongCat adapter. |
| fp8 weights, "scaled fp8" repacks | Use the bf16 or fp16 release. Shuttrr makes its own 8-bit build from it. |
| GGUF files | Shuttrr converts from safetensors into its own 8-bit build. |
| One expert of Wan 2.2 I2V A14B on its own | Both experts (high noise and low noise) are needed. Point Shuttrr at the folder that holds both. |
| The wrong VAE (a Wan 2.1 VAE for the 5B, or the reverse), or a diffusers-format VAE | Wan 2.2 TI2V 5B needs the Wan 2.2 VAE. The others need the Wan 2.1 VAE. The converter does not read diffusers VAEs yet. |
| A text width other than 4096, missing or truncated blocks | Not a complete Wan model. |

## Layouts accepted

* An official Wan-AI repo. It has `diffusion_pytorch_model*.safetensors` at the top, or `high_noise_model/` and
  `low_noise_model/` for Wan 2.2 A14B. The text encoder is `models_t5_umt5-xxl-enc-bf16.pth`, the VAE is
  `Wan2.x_VAE.pth`, and the tokenizer is `google/umt5-xxl/`.
* A `*-Diffusers` repo: `transformer/` (and `transformer_2/` for A14B), `text_encoder/`, `tokenizer/`.
  Its `vae/` is in diffusers format, so give the VAE with `--vae`.
* A ComfyUI repack or a fine-tune: one video model `.safetensors` file, with official or
  `model.diffusion_model.` names. The other parts come from elsewhere (see below).
* A Shuttrr build (`shuttrr.json`). It runs as it is.

Shuttrr reads only safetensors headers to recognise a checkpoint. `.pth` files are PyTorch pickles. Shuttrr lists
them, and converting one needs PyTorch in Shuttrr's Python (`pip install torch`). It does not need PyTorch when an
installed model can supply that part instead.

## Converting

```
shutter inspect PATH
shutter convert SOURCE OUT [--bits 8|4|0] [--lora FILE[:STRENGTH]]... [--text-encoder DIR] [--vae FILE]
                           [--tokenizer DIR] [--add MODEL_ID]
```

`inspect` reports what a folder or file is: the model, the task, each part with its size, any missing parts,
notes, any problems, and whether this Mac can run it. Add `--json` before the command
(`shutter --json inspect PATH`) to get the same information as JSON. Here it is for a small test model:

```
Wan-architecture model (3 blocks, width 256)  (wan-custom, text to video)
path      /models/repo-official
format    original weights: convert it first (shutter convert PATH OUT)
parts
  video model                        15.5 MB  3 blocks, f32, official layout
  text encoder                        0.4 MB  f32, official layout
  VAE                                 0.0 MB  f32, wan2.1 layout
  tokenizer                                   hf layout
note      No preset for this size: Shuttrr uses Wan defaults (50 steps, guidance 5, shift 5). ...
this Mac  can run it (16.0 GB of memory)
verdict   ready to convert
```

`convert` writes the build into `OUT`, an empty folder:

* `--bits 8` (the default) quantizes the video model's attention and FFN linears to 8-bit with group size 64.
  That is the same scheme as LongCat's low-memory build. `--bits 4` makes a smaller build, and `0` keeps bf16.
* `--lora FILE[:STRENGTH]` merges a LoRA before quantizing (strength 1 if not given). You can repeat it. A path
  that ends in `.safetensors` can still take a strength: `style.safetensors:0.8`.
* Missing parts: a bare fine-tune has no text encoder, tokenizer or VAE. When you do not name them, `convert`
  looks for them in the models folder (`~/.shutter/models`) and in the models in `config.json`. It finds the
  text encoder and tokenizer in an installed Shuttrr Wan build, and the text encoder and tokenizer in a LongCat
  model (`text_encoder/` or `components/text_encoder/`, `tokenizer/` or `components/tokenizer/`). It takes a
  VAE from LongCat only when its tensors are recognised as the Wan 2.1 VAE. It never takes a VAE from a Shuttrr
  build, because that VAE is already converted. The Wan 2.2 VAE only comes with the 5B model. `convert` prints
  which part it took from where.
* `--add MODEL_ID` also writes `{"adapter": "wan", "dir": OUT}` into `~/.shutter/config.json`. Restart the
  engine or the app to use it.

The converter itself uses little memory. It reads one tensor at a time and writes a few blocks per file, so a
14B model converts on a 16 GB Mac. One exception: a `.pth` text encoder is loaded whole (about 11 GB) by PyTorch.

`shutter download wan2.2-ti2v-5b` (or the Download button in the app) does the following:

1. Checks before downloading:
   * the Mac passes the memory check, otherwise the download is refused with the reason;
   * the parts it needs can be read (PyTorch, or an installed model that has them);
   * there is enough disk for the download plus the build.
2. Downloads the official repo into `~/.shutter/models/sources/<repo>`.
3. Converts it into `~/.shutter/models/<id>-shuttrr-8bit`. The progress file shows state `converting`.
4. Adds the build to `config.json`.

The original download is kept, and you can delete it afterwards. Nothing large ever downloads by itself. A
stopped download or conversion starts again from where it stopped (the conversion step starts over).

## Memory steps

Wan uses the same five memory steps as every model (docs/MODES.md). The family only declares its parts. The
engine works out each step's peak (`shutter/core/parts.py`).

| Step | What it does for Wan |
|---|---|
| `resident` | Every expert stays in memory between requests. |
| `split` | An A14B model holds one expert at a time (high noise, then low noise). |
| `staged` | As split, and the video model is freed before the VAE decode. |
| `stream_wide` | The video model's blocks are read from the SSD, 8 blocks in memory at a time. |
| `stream_tight` | The same with 2 blocks: lowest memory, slowest. |

The text encoder always runs one block at a time (`shutter/core/text_encode.py`). **All five steps produce
bit-identical output.** Streamed blocks hold exactly the same weights and run the same operations. The only
setting that is not bit-identical is `vae_tile` (a tiled decode, which blends tiles at the seams). It saves
memory at 720p, is off by default, and no memory step ever turns it on.

Until a peak has been measured (`scripts/wan_check.py` writes `measured.json`), estimates come from the build's
real part sizes and the activation formulas in `shutter/families/wan/spec.py`.

## Checked on

These Macs have produced a correct video with each model, and their peak memory was measured.

| Model | Mac | Memory | Step | Peak | Time | Date |
|---|---|---|---|---|---|---|
| | | | | | | |

The table is empty because nothing has been measured on a Mac yet. A build's `shuttrr.json` has a `checked_on`
list, and the catalog shows it (`checked_on` in `/v1/catalog`). An empty list means not yet measured.

## Credits

* Wan 2.1 / 2.2 model weights and reference code: Wan-AI / Alibaba, Apache-2.0. Each converted build includes a
  `NOTICE` file. Read the original model card for the license and acceptable-use terms.
* The MLX model code is based on mlx-video by Prince Canuma (MIT).
