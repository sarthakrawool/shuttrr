"""Cleaner fine texture for MiniMax-H3-Image text to image (skin, fur, stone, foliage).

Plain text to image with H3-Image often settles in its first steps on grainy, speckled fine texture. Two fixes, both
ported from ComfyUI-H3-Image-EasyFix by PedroMarinhoDev (Apache-2.0, commit a73c535; see NOTICE.md), run inside
Shuttrr's own MLX pipeline:

* Steering. On step 0, after each of DiT blocks 25-44, the trimmed mean of the generated-image rows is moved along a
  fixed per-block direction u to a target projection c. The table (data/vmproj_all6.safetensors) was measured once
  from paired runs with and without an empty picture in the prompt: at strength 1 the shift moves the plain run's mean
  the way that picture would, with no extra rows in the sequence. Cost: a mean over the image rows per block.
* Re-noise steps. Some steps set the latents to the step's prediction of the finished image noised back to the next
  sigma with fresh noise, x = (1 - s1) * x0 + s1 * noise (what ComfyUI's lcm sampler does on every step), so the model
  redraws the fine texture instead of keeping what it chose early. The plan has one letter per step: S = a normal
  sampler step, R = re-noise. The default "SSRRRS" re-noises after steps 2, 3 and 4. S steps use Euler ancestral
  (rectified-flow form, eta 1, as ComfyUI's euler_ancestral on a flow model) or plain Euler.

No extra model calls: still six forwards. Pictures with reference images (edits) are left alone.

Changes from the original: MLX instead of PyTorch; the residual stream is float32 here, so the shift is added without
the bf16 dither; the hook runs inside Shuttrr's own (streamed or resident) block loop; noise comes from Shuttrr's
PyTorch-compatible CPU generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

TABLE_PATH = Path(__file__).resolve().parent / "data" / "vmproj_all6.safetensors"
STEER_STEPS = (0, 1)              # the steps a table was measured for
PLANS = {"RNmid": "SSRRRS", "RN34": "SSSRRS", "lcm": "RRRRRR", "off": "SSSSSS"}
SAMPLERS = ("euler_ancestral", "euler")
_TABLES: dict[int, dict] = {}


@dataclass
class TextureFix:
    """What a request asks for. `TextureFix.off()` = the model's plain six Euler steps."""
    steer: float = 0.7                 # 0 = off, 1 = the measured shift
    steer_steps: tuple = (0,)
    plan: str = "SSRRRS"
    noise: float = 0.95                # strength of the fresh noise on R steps
    sampler: str = "euler_ancestral"   # S steps

    @classmethod
    def off(cls) -> "TextureFix":
        return cls(steer=0.0, plan="SSSSSS", sampler="euler")

    @property
    def active(self) -> bool:
        return self.steer != 0.0 or "R" in self.plan or self.sampler != "euler"

    def describe(self) -> dict:
        return {"steer": self.steer, "steer_steps": list(self.steer_steps), "plan": self.plan, "noise": self.noise,
                "sampler": self.sampler}


def parse_plan(code: str, steps: int = 6) -> str:
    code = PLANS.get(str(code), str(code)).strip().upper().replace(" ", "").replace(",", "")
    if not code or set(code) - {"S", "R"}:
        raise ValueError(f"texture plan letters are S (sampler) and R (re-noise), got {code!r}")
    code = (code + "S" * steps)[:steps]
    return code[:-1] + "S"              # the last step (to sigma 0) always runs the sampler


def load_table(step: int = 0) -> dict[int, tuple]:
    """{block: (u (5376,) float32 mx.array, c_target, c_plain)} for `step` (0 or 1)."""
    import mlx.core as mx
    if step not in STEER_STEPS:
        raise ValueError(f"steering tables exist for steps {STEER_STEPS} only, not step {step}")
    if step not in _TABLES:
        sd = mx.load(str(TABLE_PATH))
        x = "" if step == 0 else f"_s{step}"
        u, c, cp = sd["u" + x].astype(mx.float32), np.array(sd["c" + x]), np.array(sd["c_plain" + x])
        _TABLES[step] = {int(b): (u[j], float(c[j]), float(cp[j])) for j, b in enumerate(np.array(sd["blocks"]))}
    return _TABLES[step]


def trimmed_mean(rows):
    """Mean of the rows whose RMS is at most 4x the median RMS (torch.median: the lower middle value)."""
    import mlx.core as mx
    rms = mx.sqrt(mx.mean(rows * rows, axis=-1))
    med = mx.sort(rms)[(rms.shape[0] - 1) // 2]
    keep = (rms <= 4.0 * med).astype(rows.dtype)
    return (rows * keep[:, None]).sum(axis=0) / mx.maximum(keep.sum(), 1.0)


def steering_hook(fix: TextureFix, step: int, stats: Optional[list] = None) -> Optional[Callable]:
    """hook(i, x, n_img) -> x for DiT block i on this step, or None when nothing is steered. `stats` collects
    (block, measured projection (a 0-d mx.array), c_plain, c_target) to check the table matches this model."""
    if fix.steer == 0.0 or step not in fix.steer_steps:
        return None
    import mlx.core as mx
    table = load_table(step)
    strength = float(fix.steer)

    def hook(i: int, x, n_img: int):
        t = table.get(i)
        if t is None:
            return x
        u, c_target, c_plain = t
        rows = x[0, -n_img:].astype(mx.float32)
        proj = trimmed_mean(rows) @ u
        c = c_plain + strength * (c_target - c_plain)
        shifted = rows + (c - proj) * u
        if stats is not None:
            stats.append((i, proj, c_plain, c_target))
        return mx.concatenate([x[:, :-n_img], shifted[None].astype(x.dtype)], axis=1)

    return hook


def step(lat, v, sig: float, nxt: float, letter: str, fix: TextureFix, noise: Callable[[], object]):
    """One sampler step sigma -> sigma_next from the velocity v. `noise()` draws fresh N(0, 1) latents."""
    if nxt <= 0.0 or (letter == "S" and fix.sampler == "euler"):
        return lat + v * (nxt - sig)
    x0 = lat - v * sig
    if letter == "R":
        return x0 * (1.0 - nxt) + noise() * (nxt * float(fix.noise))
    # Euler ancestral, rectified-flow form (ComfyUI sample_euler_ancestral_RF, eta 1, s_noise 1)
    sigma_down = nxt * nxt / sig
    alpha_next, alpha_down = 1.0 - nxt, 1.0 - sigma_down
    renoise = max(0.0, nxt * nxt - sigma_down * sigma_down * alpha_next * alpha_next / (alpha_down * alpha_down)) ** 0.5
    r = sigma_down / sig
    x = lat * r + x0 * (1.0 - r)
    return x * (alpha_next / alpha_down) + noise() * renoise


def from_params(p: dict, opt: Callable[[str, object], object]) -> Optional[TextureFix]:
    """The request's texture fix: params "texture_fix" (default on; false / "off" = the plain six Euler steps) and
    the fine settings "texture_steer", "texture_plan", "texture_noise", "texture_sampler". Model options with the
    same names set the defaults."""
    on = p.get("texture_fix", opt("texture_fix", True))
    if isinstance(on, str):
        on = on.strip().lower() not in ("0", "false", "off", "no", "")
    if not on:
        return None
    d = TextureFix()
    steer = float(p.get("texture_steer", opt("texture_steer", d.steer)))
    if not -1.0 <= steer <= 2.0:
        raise ValueError("texture_steer must be between -1 and 2")
    noise = float(p.get("texture_noise", opt("texture_noise", d.noise)))
    if not 0.0 <= noise <= 2.0:
        raise ValueError("texture_noise must be between 0 and 2")
    sampler = str(p.get("texture_sampler", opt("texture_sampler", d.sampler)))
    if sampler not in SAMPLERS:
        raise ValueError(f"texture_sampler must be one of {', '.join(SAMPLERS)}")
    steps = p.get("texture_steer_steps", opt("texture_steer_steps", d.steer_steps))
    steps = tuple(sorted({int(k) for k in (steps if isinstance(steps, (list, tuple)) else [steps])}))
    if set(steps) - set(STEER_STEPS):
        raise ValueError("texture_steer_steps can only hold 0 and 1")
    return TextureFix(steer=steer, steer_steps=steps, plan=parse_plan(p.get("texture_plan", opt("texture_plan", d.plan))),
                      noise=noise, sampler=sampler)
