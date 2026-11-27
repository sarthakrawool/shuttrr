"""LoRAs for Z-Image, applied at run time on top of the 8-bit weights (never merged into them).

Each target Linear (or QuantizedLinear) is wrapped so it adds ``scale * (x @ A.T) @ B.T`` (the Wan family's
`LoRALinear`). The base weights are not touched, so one 8-bit build serves any LoRA, the strength can change per
request and a warm model is reused without a reload. Turning a LoRA off restores the exact base output.

File formats read (the names Z-Image LoRAs ship with):

* diffusers / PEFT: ``transformer.layers.0.attention.to_q.lora_A.weight`` / ``lora_B`` (also ``.default.weight``,
  ``diffusion_model.`` or no prefix, ``base_model.model.``);
* ComfyUI / original Z-Image names: ``diffusion_model.layers.0.attention.qkv.lora_down.weight`` (the fused q/k/v
  projection, split into three), ``attention.out``, ``feed_forward.w1..w3``, ``x_embedder``, ``final_layer...``,
  ``t_embedder.mlp.0/2``;
* kohya: ``lora_unet_layers_0_attention_to_q.lora_down.weight`` and ``.alpha``.

Not applied (reported in `skipped`): weight-difference tensors for norms (``.diff`` / ``.diff_b``) that some full
fine-tune extracts carry, and tensors for modules this model does not have.

Merge-on-convert does not apply to Z-Image: Shuttrr downloads an already-quantized build and has no conversion step,
and folding a LoRA into 8-bit weights would change them for every later request. A run-time LoRA costs only its own
size in memory (tens to a few hundred MB) and a small amount of time per step.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

#: per block, the Linear layers a LoRA can target (module paths inside one block)
BLOCK_TARGETS = ("attention.to_q", "attention.to_k", "attention.to_v", "attention.to_out.0",
                 "feed_forward.w1", "feed_forward.w2", "feed_forward.w3", "adaLN_modulation.0")
STACKS = ("layers", "noise_refiner", "context_refiner")
#: outside the blocks: file name -> model path
GLOBAL_NAMES = {"x_embedder": "all_x_embedder.2-1", "all_x_embedder.2-1": "all_x_embedder.2-1",
                "final_layer.linear": "all_final_layer.2-1.linear",
                "all_final_layer.2-1.linear": "all_final_layer.2-1.linear",
                "final_layer.adaLN_modulation.1": "all_final_layer.2-1.adaLN_modulation.0",
                "all_final_layer.2-1.adaLN_modulation.1": "all_final_layer.2-1.adaLN_modulation.0",
                "all_final_layer.2-1.adaLN_modulation.0": "all_final_layer.2-1.adaLN_modulation.0",
                "cap_embedder.1": "cap_embedder.1", "t_embedder.mlp.0": "t_embedder.linear1",
                "t_embedder.mlp.2": "t_embedder.linear2", "t_embedder.linear1": "t_embedder.linear1",
                "t_embedder.linear2": "t_embedder.linear2"}
_SUFFIXES = ((".lora_down.weight", "down"), (".lora_up.weight", "up"), (".lora_A.weight", "down"),
             (".lora_B.weight", "up"), (".lora_A.default.weight", "down"), (".lora_B.default.weight", "up"),
             (".lora_down.default.weight", "down"), (".lora_up.default.weight", "up"), (".alpha", "alpha"))
_KOHYA_BLOCK = re.compile(r"^(layers|noise_refiner|context_refiner)_(\d+)_(.+)$")
_KOHYA_PARTS = {"attention_to_q": "attention.to_q", "attention_to_k": "attention.to_k",
                "attention_to_v": "attention.to_v", "attention_to_out_0": "attention.to_out.0",
                "attention_out": "attention.to_out.0", "attention_qkv": "attention.qkv",
                "feed_forward_w1": "feed_forward.w1", "feed_forward_w2": "feed_forward.w2",
                "feed_forward_w3": "feed_forward.w3", "adaLN_modulation_0": "adaLN_modulation.0"}


def target(name: str) -> Optional[tuple[str, str]]:
    """(model module path, "down" | "up" | "alpha") for a LoRA tensor name, else None. The fused ``attention.qkv``
    comes back as ``...attention.qkv`` (split by `read_lora`)."""
    base = kind = None
    for suf, k in _SUFFIXES:
        if name.endswith(suf):
            base, kind = name[: -len(suf)], k
            break
    if base is None:
        return None
    if base.startswith("lora_unet_"):
        b = base[len("lora_unet_"):]
        m = _KOHYA_BLOCK.match(b)
        if m and m.group(3) in _KOHYA_PARTS:
            return f"{m.group(1)}.{m.group(2)}.{_KOHYA_PARTS[m.group(3)]}", kind
        for k, v in GLOBAL_NAMES.items():
            if b == k.replace(".", "_"):
                return v, kind
        return None
    for p in ("base_model.model.", "diffusion_model.", "transformer.", "model."):
        if base.startswith(p):
            base = base[len(p):]
    if base in GLOBAL_NAMES:
        return GLOBAL_NAMES[base], kind
    m = re.match(r"^(layers|noise_refiner|context_refiner)\.(\d+)\.(.+)$", base)
    if not m:
        return None
    rest = {"attention.out": "attention.to_out.0", "attention.to_out": "attention.to_out.0"}.get(m.group(3), m.group(3))
    if rest not in BLOCK_TARGETS and rest != "attention.qkv":
        return None
    return f"{m.group(1)}.{m.group(2)}.{rest}", kind


def read_lora(path, strength: float = 1.0) -> tuple[dict[str, list], dict[str, Any]]:
    """({model module path: [(A, B, scale)]}, info) from one LoRA file. info: applied / skipped counts, rank,
    size in bytes."""
    import mlx.core as mx
    p = Path(str(path)).expanduser()
    raw = mx.load(str(p))
    by_mod: dict[str, dict[str, Any]] = {}
    skipped = 0
    for name, arr in raw.items():
        t = target(name)
        if t is None:
            skipped += 1
            continue
        by_mod.setdefault(t[0], {})[t[1]] = arr
    out: dict[str, list] = {}
    ranks = set()
    for mod, d in by_mod.items():
        if "down" not in d or "up" not in d:
            skipped += len(d)
            continue
        a, b = d["down"].astype(mx.bfloat16), d["up"].astype(mx.bfloat16)
        rank = int(a.shape[0])
        ranks.add(rank)
        alpha = float(d["alpha"].item()) if "alpha" in d else float(rank)
        scale = float(strength) * alpha / rank
        if mod.endswith(".attention.qkv"):
            head = mod[: -len("qkv")]
            if b.shape[0] % 3:
                raise ValueError(f"{p.name}: fused qkv LoRA {tuple(b.shape)} does not split into three")
            for proj, part in zip(("to_q", "to_k", "to_v"), mx.split(b, 3, axis=0)):
                out.setdefault(head + proj, []).append((a, part, scale))
        else:
            out.setdefault(mod, []).append((a, b, scale))
    if not out:
        raise ValueError(f"{p.name} has no LoRA tensors for Z-Image (is it a LoRA for another model?)")
    for v in out.values():
        for a, b, _ in v:
            mx.eval(a, b)
    info = {"path": str(p), "strength": float(strength), "modules": len(out), "skipped_tensors": skipped,
            "rank": sorted(ranks), "bytes": p.stat().st_size}
    return out, info


def parse_loras(spec) -> list[tuple[str, float]]:
    """Request `params.loras`: a list of paths or {"path", "strength"} (strength default 1.0), or one path."""
    if not spec:
        return []
    if isinstance(spec, (str, dict)):
        spec = [spec]
    out = []
    for x in spec:
        if isinstance(x, dict):
            out.append((str(x["path"]), float(x.get("strength", x.get("scale", 1.0)))))
        else:
            out.append((str(x), 1.0))
    return out


def load_loras(spec) -> tuple[dict[str, list], list[dict[str, Any]]]:
    merged: dict[str, list] = {}
    infos = []
    for path, strength in parse_loras(spec):
        if strength == 0:
            continue
        d, info = read_lora(path, strength)
        infos.append(info)
        for k, v in d.items():
            merged.setdefault(k, []).extend(v)
    return merged, infos


def lora_bytes(spec) -> int:
    total = 0
    for path, strength in parse_loras(spec):
        p = Path(path).expanduser()
        if strength and p.exists():
            total += p.stat().st_size
    return total


# ---- applying

def _cls():
    from shutter.families.wan.loader import _lora_cls
    return _lora_cls()


def _get(module, path: str):
    obj = module
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else (obj[part] if isinstance(obj, dict) else getattr(obj, part))
    return obj


def _set(module, path: str, value) -> None:
    head, _, last = path.rpartition(".")
    parent = _get(module, head) if head else module
    if last.isdigit():
        parent[int(last)] = value
    elif isinstance(parent, dict):
        parent[last] = value
    else:
        setattr(parent, last, value)


def wrap(module, paths) -> None:
    cls = _cls()
    for t in paths:
        lin = _get(module, t)
        if not isinstance(lin, cls):
            _set(module, t, cls(lin))


def set_adapters(module, paths, loras: dict[str, list], prefix: str = "") -> None:
    """Point the wrapped layers at `paths` to their adapters `loras[prefix + path]` (none: base output)."""
    cls = _cls()
    for t in paths:
        w = _get(module, t)
        if isinstance(w, cls):
            w.set_adapters(loras.get(prefix + t, []))


def global_paths() -> list[str]:
    return sorted(set(GLOBAL_NAMES.values()))


def apply_resident(dit, loras: dict[str, list], include_main: bool = True) -> None:
    """Wrap (once) and set every target of a loaded DiT. With no LoRA, layers already wrapped get empty adapters."""
    stacks = STACKS if include_main else ("noise_refiner", "context_refiner")
    for stack in stacks:
        for i, block in enumerate(getattr(dit, stack)):
            paths = [t for t in BLOCK_TARGETS if not (stack == "context_refiner" and t.startswith("adaLN"))]
            if loras:
                wrap(block, paths)
            set_adapters(block, paths, loras, f"{stack}.{i}.")
    if loras:
        wrap(dit, global_paths())
    set_adapters(dit, global_paths(), loras)


def shell_name(local: str) -> str:
    """Name of a block tensor inside a LoRA-wrapped shell (the base layer moves to `.linear`)."""
    for t in BLOCK_TARGETS:
        if local.startswith(t + "."):
            return f"{t}.linear.{local[len(t) + 1:]}"
    return local
