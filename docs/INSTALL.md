# Installing Shuttrr

Shuttrr needs an Apple silicon Mac (M1 or newer), macOS 14 or newer, Python 3.11 to 3.13 and `ffmpeg`
(`brew install python@3.12 ffmpeg`). The Mac app also needs Swift (Xcode or its command line tools) to build.

## Install

```bash
scripts/install.sh              # engine + Mac app
scripts/install.sh --download   # the same, and download LongCat now (38 GB)
scripts/install.sh --no-app     # engine only; use the dashboard in a browser
```

**No Terminal at all:** open `Shuttrr.dmg` (made by `app/scripts/build_app.sh --dmg`), drag Shuttrr to
Applications and open it. A first-run window checks the Mac, installs the engine into `~/.shutter` with the
copy bundled inside the app (`Contents/Resources/engine/bootstrap.sh --copy`; it fetches a private Python with
uv when the Mac has none, and links a private ffmpeg when Homebrew's is missing), then offers the models.
Reopen it any time with **Set-up…** on the Models page. Both paths run the same `scripts/bootstrap.sh`.
For testing: `SHUTTER_ONBOARDING_PAGE=welcome|setup|model|ready` opens the window on one page, and
`SHUTTER_BOOTSTRAP_BARE=1` makes bootstrap.sh ignore this Mac's Python and ffmpeg.

The installer makes a private Python environment in `~/.shutter/venv`, installs Shuttrr and what LongCat
needs, writes `~/.shutter/config.json` if there is none, builds `~/Applications/Shuttrr.app`, and ends with
`shutter doctor`. It never uses sudo and is safe to run again.

## Get a model

Shuttrr can download its models itself. Any of these does the same thing:

- **In the app:** the Models page shows a "Set up Shuttrr" card until a model is installed. Press Download.
- **In the dashboard** (http://127.0.0.1:8484): the same card is on the Queue page, and "Download a model"
  is on the Models page.
- **In Terminal:** `shutter catalog` lists what can be downloaded, `shutter download longcat-avatar-1.6-8bit`
  fetches it, `shutter download longcat-avatar-1.6-8bit --cancel` stops it.

The download comes from Hugging Face (`SarthakRawool/LongCat-Video-Avatar-1.6-MLX-8bit-LowMem`, 37.5 GB) and
goes to `~/.shutter/models/`. It carries on by itself if you close the window, and it resumes from where it
stopped if it is interrupted. When it finishes, the model is added to `config.json` with settings that suit
this Mac's memory. With the Rust engine (`shutterd`) the engine restarts once to pick the new model up; the
app offers a Restart Engine button.

## Make the first video

```bash
shutter submit --model longcat-avatar-1.6-8bit --image face.png --audio voice.wav \
  --prompt "a person talking to the camera" --wait
```

or press New request in the app. The result lands in `~/.shutter/outputs/`.

## How much memory it needs

LongCat at 480p, 93 frames, 8 steps, measured on an M5 Max:

| Memory step | Peak (MLX) | Time | Same output as Full speed |
| --- | --- | --- | --- |
| Full speed | 33.8 GB | baseline | yes |
| Lower memory | about 13 GB | 5 to 7% slower | yes |
| Lowest memory | 12.9 GB | 5 to 7% slower | yes |
| Lowest memory + half-size decode | about 8.5 GB (estimate, not yet measured) | not yet measured | no (46.6 dB match) |

Fast mode picks the fastest step that fits in free memory, so most Macs need no settings.

### 16 GB Macs

A 16 GB Mac cannot hold the full-size final decode (12.9 GB) next to macOS. On Macs with less than 24 GB the
download step therefore sets two things:

- Low memory mode at Lowest memory, where the 18 GB model is read from the SSD a few blocks at a time.
- `vae_bf16: true`, the half-size final decode. This is the only setting in Shuttrr that changes the output
  (46.6 dB against the full decode). Remove it from the model's `options` in `config.json` to turn it off.

The prompt encoder (11.4 GB) is also read one block at a time on these Macs, which gives the same result in
about 2.5 GB. Expect a 16 GB Mac to be slow: the model is re-read from the SSD on every step, and the base
chips have a quarter of the GPU of an M5 Max. Close other apps first. Shuttrr stops a request by itself if
the Mac starts swapping.

## Check an install

`shutter doctor` lists what is in place and what is missing, in plain words (`shutter --json doctor` for
machine-readable output).
