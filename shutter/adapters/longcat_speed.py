"""Opt-in speed techniques for the LongCat-Video-Avatar DiT. NOT bit-identical; off unless a job asks:

    params.speed = {"block_sparse": {"sparsity": 0.9375, "chunk": [4, 4, 4], "keep_cond": true},
                    "step_cache":   {"threshold": 0.1}}

(`true` instead of a dict means the defaults; `false`/absent means off; `"strict": true` at the top level turns a
refusal into an error instead of a note.) The job result gets a `speed` section with what was on, what was refused
and why, and the counts (attention chunk pairs kept/skipped, steps computed/skipped). Any requested option makes the
result `identical: false`.

No model code is copied or edited. The hooks follow the adapter's existing pattern (class swaps on the loaded DiT,
done inside the generator subclass's `_mount_fast`, which the tree calls on load and after every LoRA swap):

* block_sparse: every block's self-attention module gets a subclass of its own class whose `_process_attn`
  (the tree's single dense-SDPA entry point, also used by the KV-cache path in kv_cache.py) calls
  `shutter.kernels.block_sparse.block_sparse_attention`. The swap is done BEFORE `fast.apply`, so the compiled block
  template (cloned from block 0) carries it. Which calls go sparse:
    - noise queries attending to [conditioning/reference frames + noise frames] (first segment, cond branch; and the
      continuation KV-cache path): the conditioning tokens are a dense prefix kept for every query chunk
      (`keep_cond`, default true: the reference frame carries the identity), the noise frames are chunked 4x4x4;
    - plain self-attention with no conditioning (q and k on the same grid);
    - DENSE (counted, with the reason): attention of the conditioning/reference frames among themselves (below
      `min_q_tokens`), Reference-Skip attention (continuation with `mask_frame_range > 0`: its masked query slices do
      not line up with chunk grids), token counts that are not whole latent frames, fewer keys than queries.
  Grids not divisible by the chunk (23 noise frames; 30x52 and 44x82 patches) are zero-padded and masked.
  Refused (not applied, noted): LC_ATTN_FUSED=1 or LC_TWIN>0 (those paths bypass `_process_attn`), the SSD
  streaming rungs (they mount their own block classes), trees without `_mount_fast`.
* step_cache (TeaCache, shutter.core.step_cache): every block gets a subclass of its mounted class (FastBlock,
  SplitBlock or the eager block) that routes through a per-job driver: block 0 computes the indicator (block 0's
  timestep-modulated input, `modulate_fp32(mod_norm_attn, x, shift_msa, scale_msa)`), decides compute/skip; on a
  skipped step blocks 0..46 return x unchanged and block 47 returns x + the cached residual (fp32); on a computed
  step block 47 stores residual = out - x_in. The cache is invalidated when the LoRA multipliers change (1.6's DMD ->
  OPSD switch at `mix_steps`). Not cached (counted as `steps_without_cached_calls`): steps that run through the
  continuation KV-cache path (`kv_cache.forward_noise` calls the compiled block function directly, never a block),
  i.e. every continuation segment, and the first segment with LC_FIRST_KV=1.

Under `mx.compile` the Python inside a block runs only while tracing, so attention statistics are recorded per traced
shape ("patterns"); with `compile: false` every call is seen. The counts per pattern are per batch item and head.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from shutter.core.step_cache import TeaCache
from shutter.kernels.block_sparse import UPSTREAM_CHUNK, UPSTREAM_SPARSITY, block_sparse_attention

__all__ = ["BlockSparseCfg", "StepCacheCfg", "SpeedConfig", "parse_speed", "install_attention", "install_step_cache",
           "uninstall", "lora_signature", "modulated_input", "refusals_for", "BASE_ATTR"]

BASE_ATTR = "_shutter_speed_base"
NOT_IDENTICAL = "NOT bit-identical to the reference path (opt-in)"


@dataclass
class BlockSparseCfg:
    sparsity: float = UPSTREAM_SPARSITY
    chunk: tuple = UPSTREAM_CHUNK
    keep_cond: bool = True
    min_q_tokens: int = 4096
    head_chunk: int = 1
    q_chunk_group: Optional[int] = None
    max_gather_mb: float = 1024.0       # gathered K + V per (head group, query-chunk group)


@dataclass
class StepCacheCfg:
    threshold: float = 0.1
    warmup: int = 1
    keep_last: bool = True
    max_consecutive: Optional[int] = None
    coefficients: Optional[list] = None


@dataclass
class SpeedConfig:
    block_sparse: Optional[BlockSparseCfg] = None
    step_cache: Optional[StepCacheCfg] = None
    strict: bool = False
    # per job, filled while running
    hw: Optional[tuple] = None                       # (H/16, W/16) latent patch grid of the job
    attn: dict = field(default_factory=lambda: {"sparse": {}, "dense": {}})
    cache: Optional[TeaCache] = None
    refused: dict = field(default_factory=dict)      # option -> reason
    installed: dict = field(default_factory=dict)    # option -> modules swapped
    lora_sig: Any = None
    invalidations: int = 0

    @property
    def any(self) -> bool:
        return self.block_sparse is not None or self.step_cache is not None

    @property
    def active(self) -> bool:
        """At least one requested option will actually be mounted."""
        return any(getattr(self, o) is not None and o not in self.refused for o in ("block_sparse", "step_cache"))

    def requested(self) -> dict:
        out = {}
        if self.block_sparse is not None:
            d = asdict(self.block_sparse)
            d["chunk"] = list(d["chunk"])
            out["block_sparse"] = d
        if self.step_cache is not None:
            out["step_cache"] = asdict(self.step_cache)
        return out

    def refuse(self, option: str, reason: str) -> None:
        if self.strict:
            raise ValueError(f"speed.{option} refused: {reason}")
        self.refused[option] = reason

    def report(self, steps_run: Optional[int] = None, depth: int = 48) -> dict:
        rep: dict[str, Any] = {"identical": not self.active, "label": NOT_IDENTICAL, "requested": self.requested()}
        if self.block_sparse is not None:
            b: dict[str, Any] = {"applied": "block_sparse" not in self.refused and bool(self.installed.get("block_sparse"))}
            if "block_sparse" in self.refused:
                b["refused"] = self.refused["block_sparse"]
            b["modules"] = self.installed.get("block_sparse", 0)
            pats = []
            for sig, p in self.attn["sparse"].items():
                pats.append({**p, "q_shape": list(sig[0]), "k_shape": list(sig[1]),
                             "skipped_chunk_pairs_per_dit_call": p["pairs_skipped"] * sig[0][0] * sig[0][1] * depth})
            b["sparse_patterns"] = pats
            b["dense_calls"] = [{"q_shape": list(s[0]), "k_shape": list(s[1]), "reason": s[2], "seen": n}
                                for s, n in self.attn["dense"].items()]
            b["counting"] = ("per traced shape (mx.compile: the Python in a block runs once per trace; the block "
                             "function is shared by all blocks); `seen` counts traces or eager calls")
            rep["block_sparse"] = b
        if self.step_cache is not None:
            s: dict[str, Any] = {"applied": "step_cache" not in self.refused and bool(self.installed.get("step_cache"))}
            if "step_cache" in self.refused:
                s["refused"] = self.refused["step_cache"]
            if self.cache is not None:
                s.update(self.cache.stats())
            s["invalidations"] = self.invalidations
            if steps_run is not None:
                s["steps_run"] = steps_run
            s["note"] = ("LongCat Avatar runs a distilled 8-step schedule: each step changes the input a lot, so "
                         "little is skipped at sane thresholds; continuation (KV-cache) steps are never cached")
            rep["step_cache"] = s
        if self.refused:
            rep["refused"] = dict(self.refused)
        return rep


_BS_KEYS = {f for f in BlockSparseCfg.__dataclass_fields__}
_SC_KEYS = {f for f in StepCacheCfg.__dataclass_fields__}


def _opt(v, cls, keys, name):
    if v is None or v is False:
        return None
    if v is True:
        return cls()
    if not isinstance(v, dict):
        raise ValueError(f"speed.{name} must be true/false or an object")
    v = dict(v)
    if v.pop("enabled", True) is False:
        return None
    bad = set(v) - keys
    if bad:
        raise ValueError(f"speed.{name}: unknown option(s) {sorted(bad)}; known: {sorted(keys)}")
    return cls(**v)


def parse_speed(spec) -> Optional[SpeedConfig]:
    """`params.speed` -> SpeedConfig, or None when nothing is on (the default: the reference path untouched)."""
    if spec in (None, False, {}, ""):
        return None
    if not isinstance(spec, dict):
        raise ValueError("params.speed must be an object, e.g. {\"block_sparse\": {\"sparsity\": 0.75}}")
    spec = dict(spec)
    strict = bool(spec.pop("strict", False))
    bad = set(spec) - {"block_sparse", "step_cache"}
    if bad:
        raise ValueError(f"params.speed: unknown option(s) {sorted(bad)}; known: block_sparse, step_cache")
    bs = _opt(spec.get("block_sparse"), BlockSparseCfg, _BS_KEYS, "block_sparse")
    sc = _opt(spec.get("step_cache"), StepCacheCfg, _SC_KEYS, "step_cache")
    if bs is not None:
        bs.sparsity = float(bs.sparsity)
        if not 0.0 <= bs.sparsity < 1.0:
            raise ValueError("speed.block_sparse.sparsity must be in [0, 1)")
        bs.chunk = tuple(int(c) for c in bs.chunk)
        if len(bs.chunk) != 3 or min(bs.chunk) < 1:
            raise ValueError("speed.block_sparse.chunk must be three positive integers (t, h, w)")
        bs.min_q_tokens, bs.head_chunk = int(bs.min_q_tokens), int(bs.head_chunk)
        bs.q_chunk_group = None if bs.q_chunk_group in (None, 0) else int(bs.q_chunk_group)
        bs.max_gather_mb = float(bs.max_gather_mb)
    if sc is not None:
        sc.threshold = float(sc.threshold)
        if sc.threshold < 0:
            raise ValueError("speed.step_cache.threshold must be >= 0")
        sc.warmup = int(sc.warmup)
    if bs is None and sc is None:
        return None
    return SpeedConfig(block_sparse=bs, step_cache=sc, strict=strict)


def refusals_for(speed: SpeedConfig, *, stream: bool, gen_mod=None) -> None:
    """Record what cannot be applied for this job (before loading anything)."""
    if stream:
        for opt in ("block_sparse", "step_cache"):
            if getattr(speed, opt) is not None:
                speed.refuse(opt, "not supported with the SSD weight-streaming rungs (they mount their own block "
                                  "classes); the job ran without it")
    if speed.block_sparse is not None and "block_sparse" not in speed.refused:
        att = sys.modules.get("longcat_video_avatar.models.avatar.attention")
        if att is not None and getattr(att, "ATTN_FUSED", False):
            speed.refuse("block_sparse", "LC_ATTN_FUSED=1 bypasses _process_attn")
        elif att is not None and getattr(att, "TEMPORAL_WINDOW", 0):
            speed.refuse("block_sparse", "LC_TWIN>0 bypasses _process_attn for the noise queries")
    if gen_mod is not None and not hasattr(getattr(gen_mod, "AvatarGenerator", object), "_mount_fast"):
        for opt in ("block_sparse", "step_cache"):
            if getattr(speed, opt) is not None and opt not in speed.refused:
                speed.refuse(opt, "this LongCat tree has no AvatarGenerator._mount_fast to mount the hook on")


# ----------------------------------------------------------------------------- block-sparse attention hook

_ATTN_CLASSES: dict[type, type] = {}
_BLOCK_CLASSES: dict[type, type] = {}
_ARG_NAMES = ("num_cond_latents", "return_kv", "num_ref_latents", "ref_img_index", "mask_frame_range")


def _attn_class(base: type) -> type:
    if base in _ATTN_CLASSES:
        return _ATTN_CLASSES[base]

    def __call__(self, x, shape, *args, **kwargs):
        b = dict(zip(_ARG_NAMES, args))
        b.update(kwargs)
        refskip = ((b.get("num_cond_latents") or 0) > 1 and b.get("num_ref_latents") is not None
                   and b.get("ref_img_index") is not None and (b.get("mask_frame_range") or 0) > 0)
        old = self.__dict__.get("_shutter_bsa_call")
        object.__setattr__(self, "_shutter_bsa_call", {
            "hw": (int(shape[1]), int(shape[2])),
            "dense": "reference-skip attention (mask_frame_range > 0)" if refskip else None})
        try:
            return base.__call__(self, x, shape, *args, **kwargs)
        finally:
            object.__setattr__(self, "_shutter_bsa_call", old)

    def _process_attn(self, q, k, v):
        speed: Optional[SpeedConfig] = self.__dict__.get("_shutter_speed")
        dense = lambda a, b_, c: base._process_attn(self, a, b_, c)  # noqa: E731
        if speed is None or speed.block_sparse is None:
            return dense(q, k, v)
        cfg = speed.block_sparse
        call = self.__dict__.get("_shutter_bsa_call") or {}
        hw = call.get("hw") or speed.hw
        Nq, Nk = int(q.shape[2]), int(k.shape[2])
        reason = call.get("dense")
        if reason is None:
            if not hw:
                reason = "latent grid unknown"
            elif Nq % (hw[0] * hw[1]) or Nk % (hw[0] * hw[1]):
                reason = "token count is not a whole number of latent frames"
            elif Nq < cfg.min_q_tokens:
                reason = f"fewer than min_q_tokens={cfg.min_q_tokens} queries (conditioning-only attention)"
            elif Nk < Nq:
                reason = "fewer keys than queries"
        sig_shapes = (tuple(int(s) for s in q.shape), tuple(int(s) for s in k.shape))
        if reason is not None:
            key = sig_shapes + (reason,)
            speed.attn["dense"][key] = speed.attn["dense"].get(key, 0) + 1
            return dense(q, k, v)
        Tq = Nq // (hw[0] * hw[1])
        out, st = block_sparse_attention(q, k, v, (Tq, hw[0], hw[1]), (Tq, hw[0], hw[1]), sparsity=cfg.sparsity,
                                         chunk=cfg.chunk, scale=self.scale, k_prefix=Nk - Nq,
                                         keep_prefix=cfg.keep_cond, head_chunk=cfg.head_chunk,
                                         q_chunk_group=cfg.q_chunk_group,
                                         max_gather_bytes=int(cfg.max_gather_mb * 2**20) or None,
                                         dense_fn=dense, return_stats=True)
        rec = speed.attn["sparse"].get(sig_shapes)
        if rec is None:
            rec = {k_: (list(v_) if isinstance(v_, tuple) else v_) for k_, v_ in st.items()}
            rec["seen"] = 0
            speed.attn["sparse"][sig_shapes] = rec
        rec["seen"] += 1
        return out

    cls = type(f"BlockSparse{base.__name__}", (base,), {"__call__": __call__, "_process_attn": _process_attn,
                                                         BASE_ATTR: base})
    _ATTN_CLASSES[base] = cls
    return cls


def _is_self_attention(m) -> bool:
    return (hasattr(m, "_process_attn") and hasattr(m, "rope_3d") and hasattr(m, "qkv")
            and not hasattr(type(m), BASE_ATTR))


def install_attention(dit, speed: SpeedConfig) -> int:
    """Swap every DiT self-attention module to its block-sparse subclass (before fast.apply). Returns the count."""
    n = 0
    for _, m in dit.named_modules():
        if _is_self_attention(m):
            m.__class__ = _attn_class(type(m))
            object.__setattr__(m, "_shutter_speed", speed)
            n += 1
    return n


# ----------------------------------------------------------------------------- step cache hook


def modulated_input(block, x, t, latent_shape):
    """Block 0's timestep-modulated input, exactly as the block computes it before self-attention."""
    import mlx.core as mx
    import mlx.nn as nn

    modulate_fp32 = sys.modules["longcat_video_avatar.models.blocks"].modulate_fp32
    B, N, C = x.shape
    T = int(latent_shape[0])
    ada = block.adaLN_modulation[1](nn.silu(t)).astype(mx.float32)[:, :, None, :]
    return modulate_fp32(block.mod_norm_attn, x.reshape(B, T, -1, C), ada[..., :C], ada[..., C:2 * C])


def _residual(x_in, x_out):
    import mlx.core as mx

    return x_out.astype(mx.float32) - x_in.astype(mx.float32)


def _apply(x_in, res):
    import mlx.core as mx

    return (x_in.astype(mx.float32) + res).astype(x_in.dtype)


class _Driver:
    """Per-job state shared by the wrapped blocks of one DiT."""

    def __init__(self, speed: SpeedConfig, depth: int):
        self.speed, self.depth = speed, depth
        self.slot, self.skipping, self.x_in = None, False, None

    def call(self, block, i, base, x, y, t, y_seqlen, latent_shape, args, kwargs):
        cache = self.speed.cache
        if i == 0:
            ind = modulated_input(block, x, t, latent_shape)
            self.slot, self.skipping = cache.begin(ind, tuple(x.shape))
            self.x_in = x
        if self.skipping:
            if i == self.depth - 1:
                self.skipping = False
                out, self.x_in = cache.apply(self.slot, self.x_in, _apply), None
                return out
            return x
        out = base.__call__(block, x, y, t, y_seqlen, latent_shape, *args, **kwargs)
        if i == self.depth - 1 and self.x_in is not None:
            cache.store(self.slot, self.x_in, out, _residual)
            self.x_in = None
        return out


def _block_class(base: type) -> type:
    if base in _BLOCK_CLASSES:
        return _BLOCK_CLASSES[base]

    def __call__(self, x, y, t, y_seqlen, latent_shape, *args, **kwargs):
        d = self.__dict__
        return d["_shutter_tc_driver"].call(self, d["_shutter_tc_index"], base, x, y, t, y_seqlen, latent_shape,
                                            args, kwargs)

    cls = type(f"StepCached{base.__name__}", (base,), {"__call__": __call__, BASE_ATTR: base})
    _BLOCK_CLASSES[base] = cls
    return cls


def install_step_cache(dit, speed: SpeedConfig, num_steps: Optional[int] = None) -> int:
    """Wrap every block (after fast.apply) so the stack can be skipped. Returns the count."""
    blocks = list(dit.blocks)
    if speed.cache is None:
        sc = speed.step_cache
        speed.cache = TeaCache(threshold=sc.threshold, coefficients=sc.coefficients, warmup=sc.warmup,
                               keep_last=sc.keep_last, max_consecutive=sc.max_consecutive, num_steps=num_steps)
    drv = _Driver(speed, len(blocks))
    for i, b in enumerate(blocks):
        b.__class__ = _block_class(type(b))
        object.__setattr__(b, "_shutter_tc_driver", drv)
        object.__setattr__(b, "_shutter_tc_index", i)
    return len(blocks)


def lora_signature(dit) -> tuple:
    """Runtime-LoRA multipliers of block 0 (what changes at 1.6's DMD -> OPSD switch)."""
    try:
        b0 = dit.blocks[0]
    except Exception:
        return ()
    mult = {}
    for _, m in b0.named_modules():
        mult.update(getattr(m, "mult", {}) or {})
    return tuple(sorted((k, float(v)) for k, v in mult.items()))


def uninstall(dit) -> int:
    """Restore every swapped class (attention and blocks) to the class it was swapped from."""
    n = 0
    if dit is None or not hasattr(dit, "named_modules"):
        return 0
    for _, m in dit.named_modules():
        base = getattr(type(m), BASE_ATTR, None)
        if base is not None and type(m).__dict__.get(BASE_ATTR) is base:
            m.__class__ = base
            for a in ("_shutter_speed", "_shutter_tc_driver", "_shutter_tc_index", "_shutter_bsa_call"):
                m.__dict__.pop(a, None)
            n += 1
    return n
