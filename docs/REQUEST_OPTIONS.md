# Request options, estimates and saved defaults (0.2.10)

Shuttrr is the engine: every setting of a request is the person's choice. The chip/RAM defaults are only starting
values. These endpoints (Python engine, `shutter/server/api.py`) give the app what it needs to show and edit them.
Sizes follow the Shuttrr convention: `width` and `height` in pixels (Wan 2.2 5B native: width 1280, height 704).

## GET /v1/models/{id}/request-options

```json
{
  "model": "wan2.2-ti2v-5b",
  "editable": true,
  "defaults":      {"width": 1280, "height": 704, "frames": 25, "steps": 30, "guidance": 5.0, "shift": 5.0,
                    "fps": 24, "vae_tile": true},
  "mac_defaults":  { same keys: this Mac's starting values (chip + memory), what "Reset to defaults" goes back to },
  "user_defaults": { only the keys the person saved for this model, e.g. {"frames": 49} },
  "sizes": [{"width": 1280, "height": 704, "label": "1280x704 (native)"},
            {"width": 704, "height": 1280, "label": "704x1280 (portrait)"},
            {"width": 832, "height": 480, "label": "832x480 (faster, softer)"}],
  "limits": {"size_multiple": 32, "min_side": 256, "max_area": 901120,
             "frames_min": 5, "frames_max": 241, "frames_rule": "4n+1",
             "steps_min": 1, "steps_max": 100, "guidance_min": 1.0, "guidance_max": 20.0,
             "shift_min": 0.5, "shift_max": 20.0},
  "estimate": { an estimate object (below) for "defaults" }
}
```

`defaults` = saved user defaults over this Mac's starting values over the model's own. `editable` is false for
models whose adapter has no editable size / steps (LongCat, the test model): the app then shows only seed and
memory step. 404 for an unknown model.

## POST /v1/models/{id}/estimate

Body: any of `width`, `height`, `frames`, `steps`, `vae_tile` (missing keys = `defaults`). Answer:

```json
{
  "seconds": 613, "label": "About 10 min on this Mac", "measured": true,
  "basis": "measured on a 16 GB MacBook Pro M5",
  "peak_bytes": 5841155522,
  "fit": {"level": "recommended" | "barely" | "no_fit", "reason": "..."},
  "warnings": ["Likely too big for this Mac's memory. ..."],
  "issues": [{"field": "frames", "message": "Frames must be 4n+1 (5, 9, ... 49).", "fix": 49}]
}
```

`label` says "(estimate)" when the time is scaled from another size or another Mac (by GPU cores). `warnings` never
block a request (the memory watchdog still protects the Mac). `issues` are values the engine would change: the app
shows them next to the field and can offer `fix`.

## PUT /v1/models/{id}/defaults, DELETE /v1/models/{id}/defaults

PUT body: any of `width`, `height`, `frames`, `steps`, `guidance`, `shift`, `vae_tile`. Saved in config.json under
the model's `options` (`default_size`, `default_frames`, `default_steps`, `default_guidance`, `default_shift`,
`vae_tile`) and listed in `options.user_keys`, so updates and re-downloads keep them. DELETE puts this Mac's
starting values back. Both answer with the new request-options. Takes effect for the next request.

## Requests

POST /v1/jobs already takes `width`, `height`, `frames` (or `duration_s`), `steps`, `guidance`, `seed`, `mode`;
`shift` and `vae_tile` go in `params`.
