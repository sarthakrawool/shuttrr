"""What a person can set on a request, this Mac's starting values, their own saved defaults, and a time and memory
estimate for any choice (docs/REQUEST_OPTIONS.md).

Shuttrr only starts from the chip / memory defaults (catalog.preset_for); every value can be changed per request
(POST /v1/jobs) or saved as the person's own default for a model (config.json `options`, keys listed in
`options.user_keys` so a re-download or an update keeps them).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from shutter.core import catalog as cat
from shutter.core.config import Config

GB = 1024**3

# request field -> config option key
OPTION_KEYS = {"frames": "default_frames", "steps": "default_steps", "guidance": "default_guidance",
               "shift": "default_shift", "vae_tile": "vae_tile"}
FIELDS = ("width", "height", "frames", "steps", "guidance", "shift", "vae_tile")

LIMITS_WAN = {"min_side": 256, "frames_min": 5, "frames_max": 241, "frames_rule": "4n+1",
              "steps_min": 1, "steps_max": 100, "guidance_min": 1.0, "guidance_max": 20.0,
              "shift_min": 0.5, "shift_max": 20.0}


LIMITS_IMAGE = {"min_side": 256, "max_side": 2048, "steps_min": 1, "steps_max": 50, "guidance_min": 0.0,
                "guidance_max": 20.0, "size_multiple": 16, "count_min": 1, "count_max": 8}
IMAGE_SIZES = [(1024, 1024, "1024x1024 (square)"), (768, 768, "768x768 (square, faster)"),
               (896, 1152, "896x1152 (portrait)"), (1152, 896, "1152x896 (landscape)"),
               (720, 1280, "720x1280 (tall, 9:16)"), (1280, 720, "1280x720 (wide, 16:9)")]
#: style presets for image models: words added around the prompt (named for the look, never a studio)
IMAGE_STYLES = {
    "3d-animated": {"label": "3D animated",
                    "prefix": "3D animated film still: ",
                    "suffix": ", big expressive eyes, soft subsurface skin, stylised proportions, warm cinematic "
                              "lighting, shallow depth of field"},
}


def styled_prompt(prompt: str, style: Optional[str]) -> str:
    st = IMAGE_STYLES.get(str(style or ""))
    return f"{st['prefix']}{prompt}{st['suffix']}" if st else prompt


class OptionsError(ValueError):
    pass


#: image adapters; H3-Image always runs its own 6 steps without guidance and has no LoRA support yet
IMAGE_ADAPTERS = ("z-image", "h3-image")
FIXED_STEP_IMAGE = {"h3-image": 6}


def _is_image(spec: dict[str, Any]) -> bool:
    return spec.get("adapter") in IMAGE_ADAPTERS


def _image_mac(model_id: str, total: int, adapter: str = "z-image") -> dict[str, Any]:
    e = cat.CATALOG.get(model_id) or {}
    h, w = e.get("image_default_size") or (1024, 1024)
    steps = FIXED_STEP_IMAGE.get(adapter) or int(e.get("image_default_steps", 9))
    return {"width": int(w), "height": int(h), "steps": steps, "guidance": 0.0}


def _truthy_opt(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() not in ("0", "false", "off", "no", "")
    return bool(v)


def _image_options(cfg: Config, model_id: str, spec: dict[str, Any], total: int) -> dict[str, Any]:
    mac = _image_mac(model_id, total, spec.get("adapter") or "z-image")
    if model_id in cat.CATALOG:
        mac.update({k: v for k, v in _from_options(cat.preset_for(model_id, total)["options"]).items()
                    if k in ("width", "height", "steps", "guidance")})
    opts = dict(spec.get("options") or {})
    user = {k: v for k, v in _from_options(opts, list(opts.get("user_keys") or [])).items()
            if k in ("width", "height", "steps", "guidance")}
    defaults = {**mac, **user}
    return {"model": model_id, "kind": "image", "editable": True, "defaults": defaults, "mac_defaults": mac,
            "user_defaults": user,
            "sizes": [{"width": w, "height": h, "label": lab} for w, h, lab in IMAGE_SIZES],
            "limits": LIMITS_IMAGE,
            "styles": [{"id": k, "label": v["label"]} for k, v in IMAGE_STYLES.items()],
            "lora": spec.get("adapter") == "z-image",
            # reference pictures the model reads (<Picture 1>, ...): 0 = text to image only
            "references": 3 if spec.get("adapter") == "h3-image" else 0,
            "fixed_steps": FIXED_STEP_IMAGE.get(spec.get("adapter") or ""),
            # H3-Image: cleaner fine texture (steering + re-noise steps), a recommended switch, on unless the model's
            # options say texture_fix false; text to image only
            "texture_fix": spec.get("adapter") == "h3-image",
            "texture_fix_default": spec.get("adapter") == "h3-image" and _truthy_opt(opts.get("texture_fix", True)),
            "estimate": estimate(cfg, model_id, defaults, total_bytes=total, _defaults=defaults)}


def _image_issues(v: dict[str, Any]) -> list[dict[str, Any]]:
    lim, out = LIMITS_IMAGE, []
    m = lim["size_multiple"]
    for k in ("width", "height"):
        x = v[k]
        if x % m or not lim["min_side"] <= x <= lim["max_side"]:
            fix = min(lim["max_side"], max(lim["min_side"], int(x // m) * m))
            out.append({"field": k, "message": f"{k.capitalize()} must be a multiple of {m} between "
                        f"{lim['min_side']} and {lim['max_side']}; it would be made {fix}.", "fix": fix})
    for k in ("steps", "guidance"):
        lo, hi = lim[f"{k}_min"], lim[f"{k}_max"]
        if not lo <= v[k] <= hi:
            out.append({"field": k, "message": f"{k.capitalize()} should be between {lo:g} and {hi:g}.",
                        "fix": min(hi, max(lo, v[k]))})
    return out


def _image_estimate(cfg: Config, model_id: str, body: dict[str, Any], total: int,
                    defaults: Optional[dict[str, Any]]) -> dict[str, Any]:
    defaults = defaults or _image_options(cfg, model_id, _spec(cfg, model_id), total)["defaults"]
    v = {**defaults, **{k: x for k, x in _clean_values(body or {}).items() if k in ("width", "height", "steps",
                                                                                     "guidance")}}
    count = int((body or {}).get("count") or 1)
    issues = _image_issues(v)
    h, w, steps = v["height"], v["width"], v["steps"]
    t = cat.expected_time(model_id, total, h=h, w=w, f=1, steps=steps) if model_id in cat.CATALOG \
        else {"seconds": None, "label": "", "measured": False, "basis": ""}
    if t.get("seconds") and float(v.get("guidance") or 0) > 1:
        t = dict(t, seconds=t["seconds"] * 2, measured=False)
    if t.get("seconds") and count > 1:
        t = dict(t, seconds=t["seconds"] * count)
    if t.get("seconds"):
        sec = t["seconds"]
        t["label"] = (f"About {cat._seconds(sec)} on this Mac" + (f" for {count} images" if count > 1 else "")
                      + ("" if t["measured"] else " (estimate)"))
    peak, fit = cat.request_peak(model_id, total, h, w, 1, False) if model_id in cat.CATALOG else (0, None)
    warnings = []
    if float(v.get("guidance") or 0) > 1:
        warnings.append("Guidance above 1 runs the model twice per step (about twice as long). Z-Image Turbo is "
                        "made to run without it.")
    return {"seconds": t["seconds"], "label": t["label"], "measured": t["measured"], "basis": t["basis"],
            "peak_bytes": int(peak or 0), "fit": fit or {"level": "unknown", "reason": ""},
            "warnings": warnings, "issues": issues}


def _spec(cfg: Config, model_id: str) -> dict[str, Any]:
    spec = (cfg.models or {}).get(model_id)
    if spec is None and model_id in cat.CATALOG:
        spec = {"adapter": cat.CATALOG[model_id]["adapter"], "options": {}}
    if spec is None:
        raise KeyError(model_id)
    return spec


def _wan_config(cfg: Config, model_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    """The Wan model's own config: from its build's shuttrr.json, else from the catalog shape."""
    try:
        d = Path(cfg.model_dir(model_id))
        return json.loads((d / "shuttrr.json").read_text())["config"]
    except Exception:
        pass
    if model_id in cat.CATALOG and cat.CATALOG[model_id].get("shape"):
        from shutter.families.wan.spec import WanShape
        e = cat.CATALOG[model_id]
        return WanShape(**e["shape"]).to_config(e.get("variant", model_id))
    raise KeyError(model_id)


def _own_defaults(wcfg: dict[str, Any]) -> dict[str, Any]:
    h, w = wcfg.get("default_size") or (480, 832)
    gs = wcfg.get("sample_guide_scale", 5.0)
    return {"width": int(w), "height": int(h), "frames": int(wcfg.get("frame_num", 81)),
            "steps": int(wcfg.get("sample_steps", 50)),
            "guidance": float(gs[0] if isinstance(gs, list) else gs), "shift": float(wcfg.get("sample_shift", 5.0)),
            "fps": float(wcfg.get("sample_fps", 16)), "vae_tile": False}


def _from_options(opts: dict[str, Any], keys: Optional[list[str]] = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    use = set(keys) if keys is not None else None
    if opts.get("default_size") and (use is None or "default_size" in use):
        h, w = opts["default_size"]
        out.update(width=int(w), height=int(h))
    for field, key in OPTION_KEYS.items():
        if key in opts and (use is None or key in use):
            v = opts[key]
            out[field] = bool(v) if field == "vae_tile" else (float(v) if field in ("guidance", "shift") else int(v))
    return out


def _to_options(values: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "width" in values or "height" in values:
        out["default_size"] = [int(values["height"]), int(values["width"])]
    for field, key in OPTION_KEYS.items():
        if field in values:
            out[key] = values[field]
    return out


def _sizes(wcfg: dict[str, Any]) -> list[dict[str, Any]]:
    h, w = wcfg.get("default_size") or (480, 832)
    out = [{"width": w, "height": h, "label": f"{w}x{h} (native)"},
           {"width": h, "height": w, "label": f"{h}x{w} (portrait)"}]
    if (h, w) == (704, 1280):
        out.append({"width": 832, "height": 480, "label": "832x480 (faster, softer)"})
    elif (h, w) == (480, 832):
        out.append({"width": 512, "height": 320, "label": "512x320 (fastest)"})
    return out


def _limits(wcfg: dict[str, Any]) -> dict[str, Any]:
    st = wcfg.get("vae_stride") or [4, 8, 8]
    return {"size_multiple": 2 * int(st[1]), "max_area": int(wcfg.get("max_area") or 0), **LIMITS_WAN}


def _clean_values(body: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in FIELDS:
        if k in body and body[k] is not None:
            v = body[k]
            try:
                out[k] = (bool(v) if k == "vae_tile" else float(v) if k in ("guidance", "shift") else int(v))
            except (TypeError, ValueError):
                raise OptionsError(f"{k} must be a number") from None
    if ("width" in out) != ("height" in out):
        raise OptionsError("give width and height together")
    return out


def request_options(cfg: Config, model_id: str, total_bytes: Optional[int] = None) -> dict[str, Any]:
    spec = _spec(cfg, model_id)
    if _is_image(spec):
        return _image_options(cfg, model_id, spec, cat.total_memory() if total_bytes is None else total_bytes)
    if spec.get("adapter") != "wan":
        return {"model": model_id, "editable": False, "defaults": {}, "mac_defaults": {}, "user_defaults": {},
                "sizes": [], "limits": {}, "estimate": None}
    total = cat.total_memory() if total_bytes is None else total_bytes
    wcfg = _wan_config(cfg, model_id, spec)
    mac = _own_defaults(wcfg)
    if model_id in cat.CATALOG:
        mac.update(_from_options(cat.preset_for(model_id, total)["options"]))
    opts = dict(spec.get("options") or {})
    user = _from_options(opts, list(opts.get("user_keys") or []))
    defaults = {**mac, **user}
    return {"model": model_id, "editable": True, "defaults": defaults, "mac_defaults": mac, "user_defaults": user,
            "sizes": _sizes(wcfg), "limits": _limits(wcfg),
            "estimate": estimate(cfg, model_id, defaults, total_bytes=total, _wcfg=wcfg, _defaults=defaults)}


def _issues(v: dict[str, Any], lim: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    m = lim.get("size_multiple") or 16
    for k in ("width", "height"):
        x = v[k]
        if x % m or x < lim["min_side"]:
            fix = max(lim["min_side"], int(round(x / m)) * m)
            out.append({"field": k, "message": f"{k.capitalize()} must be a multiple of {m} (at least "
                        f"{lim['min_side']}); it would be made {fix}.", "fix": fix})
    if lim.get("max_area") and v["width"] * v["height"] > lim["max_area"]:
        out.append({"field": "width", "message": "Bigger than this model's largest size; it would be scaled "
                    "down to fit.", "fix": None})
    f = v["frames"]
    if (f - 1) % 4 or not lim["frames_min"] <= f <= lim["frames_max"]:
        fix = min(lim["frames_max"], max(lim["frames_min"], (f - 1 + 2) // 4 * 4 + 1))
        out.append({"field": "frames", "message": f"Frames must be 4n+1 between {lim['frames_min']} and "
                    f"{lim['frames_max']} (for example 25, 49, 121).", "fix": fix})
    for k in ("steps", "guidance", "shift"):
        lo, hi = lim[f"{k}_min"], lim[f"{k}_max"]
        if not lo <= v[k] <= hi:
            out.append({"field": k, "message": f"{k.capitalize()} should be between {lo:g} and {hi:g}.",
                        "fix": min(hi, max(lo, v[k]))})
    return out


def estimate(cfg: Config, model_id: str, body: dict[str, Any], total_bytes: Optional[int] = None,
             _wcfg: Optional[dict[str, Any]] = None, _defaults: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    spec = _spec(cfg, model_id)
    if _is_image(spec):
        return _image_estimate(cfg, model_id, body, cat.total_memory() if total_bytes is None else total_bytes,
                               _defaults)
    if spec.get("adapter") != "wan":
        return {"seconds": None, "label": "", "measured": False, "basis": "", "peak_bytes": 0,
                "fit": {"level": "unknown", "reason": ""}, "warnings": [], "issues": []}
    total = cat.total_memory() if total_bytes is None else total_bytes
    wcfg = _wcfg or _wan_config(cfg, model_id, spec)
    defaults = _defaults or request_options(cfg, model_id, total)["defaults"]
    v = {**defaults, **_clean_values(body or {})}
    lim = _limits(wcfg)
    issues = _issues(v, lim)
    h, w, f, steps, tiled = v["height"], v["width"], v["frames"], v["steps"], bool(v["vae_tile"])
    t = cat.expected_time(model_id, total, h=h, w=w, f=f, steps=steps, tiled=tiled) if model_id in cat.CATALOG \
        else {"seconds": None, "label": "", "measured": False, "basis": ""}
    peak, fit = cat.request_peak(model_id, total, h, w, f, tiled) if model_id in cat.CATALOG else (0, None)
    warnings = []
    usable = max(0, total - cat.macos_baseline(total)) if total else 0
    if peak and usable:
        if peak > usable:
            warnings.append(f"Likely too big for this Mac's memory (needs about {peak / GB:.1f} GB). It can still "
                            "be tried: the memory watchdog stops it before the Mac runs out.")
        elif peak + cat.EVERYDAY_APPS > usable:
            warnings.append("Tight on this Mac's memory. Close other apps first.")
    if not tiled and h * w >= 704 * 1280 and total and total < cat.SMALL_MAC_BYTES:
        warnings.append("Without the tiled decode a 704x1280 clip needs about 10 GB at the end. Turn on tiled "
                        "decode on this Mac.")
    return {"seconds": t["seconds"], "label": t["label"], "measured": t["measured"], "basis": t["basis"],
            "peak_bytes": int(peak or 0), "fit": fit or {"level": "unknown", "reason": ""},
            "warnings": warnings, "issues": issues}


def save_defaults(cfg: Config, model_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Save the person's own defaults for a model (merged with any saved before)."""
    values = _clean_values(body or {})
    if not values:
        raise OptionsError("nothing to save")
    cfg = Config.load(cfg.path)
    spec = dict(_spec(cfg, model_id))
    if spec.get("adapter") != "wan" and not _is_image(spec):
        raise OptionsError("this model has no editable defaults")
    if _is_image(spec):
        values = {k: v for k, v in values.items() if k in ("width", "height", "steps", "guidance")}
        if not values:
            raise OptionsError("nothing to save")
    opts = dict(spec.get("options") or {})
    new = _to_options(values)
    opts.update(new)
    opts["user_keys"] = sorted(set(opts.get("user_keys") or []) | set(new))
    spec["options"] = opts
    cfg.models[model_id] = spec
    cfg.save()
    return opts


def reset_defaults(cfg: Config, model_id: str, total_bytes: Optional[int] = None) -> dict[str, Any]:
    """Drop the person's saved defaults: this Mac's starting values again."""
    cfg = Config.load(cfg.path)
    spec = dict(_spec(cfg, model_id))
    opts = dict(spec.get("options") or {})
    for k in opts.pop("user_keys", []) or []:
        opts.pop(k, None)
    if model_id in cat.CATALOG:
        total = cat.total_memory() if total_bytes is None else total_bytes
        opts.update(cat.preset_for(model_id, total)["options"])
    spec["options"] = opts
    cfg.models[model_id] = spec
    cfg.save()
    return opts
