"""SSD weight streaming for the LongCat-Video-Avatar DiT (rungs ``stream_wide`` / ``stream_tight``).

The 48 DiT blocks (8-bit: ``weight``/``scales``/``biases`` per QuantizedLinear, plus the bf16 adaLN linears and
norms) are read from the DiT safetensors shards block by block through :class:`shutter.core.stream.BlockStreamer`;
everything else (embedders, audio projection, final layer: ~0.2 GB) stays resident. No model code is copied:

* The DiT is built the way ``generation.load_dit`` builds it (``from_config`` + ``nn.quantize`` at exactly the
  converted paths), but with ``depth=0``; its non-block tensors are loaded from the shards. One block "shell"
  (a real ``LongCatAvatarSingleStreamBlock``, quantized like block 0, with the active LoRAs attached as
  ``runtime_lora.LoRALinear`` wrappers) receives block i's tensors before it runs and placeholders after.
* ``dit.blocks`` becomes a :class:`StreamedBlocks` sequence. Everything that walks the blocks (the DiT forward,
  ``kv_cache.build_cache`` / ``forward_noise``) iterates it; the iterator loads block i into the shell, yields the
  shell, and when the caller asks for block i+1 it first evaluates block i's outputs (``mx.eval``), then drops
  block i's weights. The outputs are recorded by a subclass of the shell's mounted class (whole-block
  ``FastBlock``, 3-stage ``SplitBlock`` or the eager block) and by a wrapper around ``dit._compiled_fns``
  (the KV-cache block functions call ``f(block, ...)`` without calling the block).
* ``fast.apply`` / ``fast.apply_split`` are mounted on the skeleton while ``dit.blocks == [shell]``, through the
  generator's own ``_mount_fast``. ``fast.CompiledBlockFn`` already takes the block's weights as ARGUMENTS
  (``block.parameters()``) and shares one compiled function across all 48 blocks, so the streamed path runs the
  very same traced graph with the very same weight bytes: it is bit-identical to the production compiled path,
  not only to the eager one (asserted on CPU in tests/test_longcat_stream.py for compile on and off).
* Runtime LoRA: the active LoRA's per-block tensors are streamed with the block (keys are
  ``lora___lorahyphen___blocks___lorahyphen___{i}___...``, decoded with ``lora.decode_module_name``), cast to
  bf16 exactly as ``attach_lora`` does. Residency mirrors ``LC_LORA_SWAP=1``: only LoRAs with a non-zero
  multiplier are attached to the shell; when the step's multipliers change the active set (1.6: DMD -> OPSD at
  ``mix_steps``) the shell's LoRA structure changes, the compiled functions are re-mounted (as the tree's own swap
  does) and the streamer is rebuilt. ``alpha_scale`` values (python floats) are read once; the compiled template
  is cloned while the shell holds block 0's scales, exactly like production's template is cloned from block 0.
* KV cache (continuation segments): kept. The cached keys/values are activations, not weights; they are built
  and used through the same streamed blocks, so continuation stays exact (it costs 48 x 2 x cond tokens x 4096
  x 2 B, e.g. ~6 GB at 480p, ~14 GB at 704x1312: noted in the estimate for multi-segment clips).

Cost: one ``mx.eval`` per block instead of one per step (the split rung already does three per block for
+2.3% time in total, REPORT.md), one host->device copy per block on the compute thread, and SSD reads overlapped
with compute by the streamer's reader thread (0.4 GB per block vs ~1 s of compute per block at 480p).
"""

from __future__ import annotations

import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from shutter.core.stream import BlockStreamer, SafetensorsIndex, TensorEntry

__all__ = ["MultiIndex", "StreamedDiT", "StreamedBlocks", "build_streamed_dit", "install_lora_hook",
           "dit_stream_sizes", "BLOCK_RX"]

BLOCK_RX = re.compile(r"^blocks\.(\d+)\.(.+)$")
DIT_INDEX = "diffusion_pytorch_model.safetensors.index.json"


def _log(msg: str) -> None:
    sys.stderr.write(f"[shutter.stream] {msg}\n")
    sys.stderr.flush()


# ----------------------------------------------------------------------------- composite index


class MultiIndex:
    """Several SafetensorsIndex under namespaces (``"dit::"``, ``"lora:dmd::"``...), so tensor names that repeat
    across files (the two LoRA files have identical key names) can share one BlockStreamer. Implements what
    BlockStreamer uses: ``entries``, ``read``, ``nbytes``, ``set_nocache``."""

    SEP = "::"

    def __init__(self, parts: dict[str, SafetensorsIndex]):
        self.parts = dict(parts)
        self.entries: dict[str, TensorEntry] = {}
        self._owner: dict[str, tuple[SafetensorsIndex, str]] = {}
        for ns, idx in self.parts.items():
            for name, e in idx.entries.items():
                full = f"{ns}{self.SEP}{name}"
                self.entries[full] = e
                self._owner[full] = (idx, name)

    def read(self, name: str):
        idx, local = self._owner[name]
        return idx.read(local)

    def nbytes(self, names: Optional[Iterable[str]] = None) -> int:
        if names is None:
            names = self.entries
        return sum(self.entries[n].nbytes for n in names)

    def set_nocache(self, on: bool = True) -> None:
        for idx in self.parts.values():
            idx.set_nocache(on)

    def close(self) -> None:
        for idx in self.parts.values():
            idx.close()


# ----------------------------------------------------------------------------- helpers


def _dit_shards(folder: Path) -> list[Path]:
    idx = folder / DIT_INDEX
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        return [folder / s for s in sorted(set(wm.values()))]
    return sorted(folder.glob("*.safetensors"))


def dit_stream_sizes(folder: Path) -> Optional[dict[str, int]]:
    """Header-only byte counts of a DiT folder: total, non-block, per-block (max), number of blocks."""
    folder = Path(folder)
    shards = _dit_shards(folder) if folder.is_dir() else []
    if not shards:
        return None
    idx = SafetensorsIndex(shards)
    try:
        per: dict[int, int] = {}
        rest = 0
        for name, e in idx.entries.items():
            m = BLOCK_RX.match(name)
            if m:
                per[int(m.group(1))] = per.get(int(m.group(1)), 0) + e.nbytes
            else:
                rest += e.nbytes
        return {"total": rest + sum(per.values()), "nonblock": rest, "block_max": max(per.values()) if per else 0,
                "blocks": len(per)}
    finally:
        idx.close()


def _get(root, path: str):
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


class _Recorder:
    """Outputs of the block calls made since the last flush (evaluated before the block's weights go)."""

    def __init__(self):
        self.pending: list = []

    def add(self, out) -> None:
        self.pending.append(out)

    def flush(self, mx) -> None:
        if self.pending:
            mx.eval(self.pending)
            self.pending.clear()


_REC_CLASSES: dict[type, type] = {}


def _recording_class(base: type) -> type:
    cls = _REC_CLASSES.get(base)
    if cls is None:
        def __call__(self, *a, **k):
            out = base.__call__(self, *a, **k)
            rec = self.__dict__.get("_shutter_rec")
            if rec is not None:
                rec.add(out)
            return out

        cls = type(f"Streamed{base.__name__}", (base,), {"__call__": __call__, "_shutter_base": base})
        _REC_CLASSES[base] = cls
    return cls


class _RecordingFn:
    """Wraps a ``fast.CompiledBlockFn`` so calls on the streamed shell record their outputs."""

    def __init__(self, fn, rec: _Recorder, shell):
        self.fn, self.rec, self.shell = fn, rec, shell

    def __call__(self, block, *args, **kw):
        out = self.fn(block, *args, **kw)
        if block is self.shell:
            self.rec.add(out)
        return out

    def __getattr__(self, name):
        return getattr(self.fn, name)


class _RecordingFns(dict):
    """``dit._compiled_fns`` replacement: same dict, values handed out wrapped."""

    def __init__(self, base: dict, rec: _Recorder, shell):
        super().__init__(base)
        self._rec, self._shell = rec, shell

    def __getitem__(self, key):
        v = dict.__getitem__(self, key)
        if isinstance(v, _RecordingFn):
            return v
        return _RecordingFn(v, self._rec, self._shell)


class StreamedBlocks:
    """What ``dit.blocks`` becomes: ``len()`` = 48, ``[i]`` = the shell (structure only, used by
    ``fast.compiled`` to clone its template), iteration = stream the blocks through the shell."""

    def __init__(self, owner: "StreamedDiT"):
        self._owner = owner

    def __len__(self) -> int:
        return self._owner.n_blocks

    def __getitem__(self, i):
        if not isinstance(i, int):
            raise TypeError("streamed blocks support integer indexing only")
        n = self._owner.n_blocks
        if not -n <= i < n:
            raise IndexError(i)
        self._owner.set_scales(i % n)
        return self._owner.shell

    def __iter__(self):
        return self._owner.iterate()


# ----------------------------------------------------------------------------- the streamed DiT


class StreamedDiT:
    """Owns the shell, the streamer and the mount state of one streamed DiT (``dit._shutter_stream``)."""

    def __init__(self, dit, shell, n_blocks: int, dit_index: SafetensorsIndex, loras: dict[str, Path], *,
                 window: int, readahead: Optional[int] = None, nocache: bool = False,
                 mount: Callable[[], None], lora_dtype=None):
        import mlx.core as mx

        self._mx = mx
        self.dit = dit
        self.shell = shell
        self.n_blocks = n_blocks
        self.window = max(1, int(window))
        self.readahead = max(1, self.window - 1) if readahead is None else int(readahead)
        self.nocache = nocache
        self._mount_fn = mount
        self.lora_dtype = lora_dtype if lora_dtype is not None else mx.bfloat16
        self.rec = _Recorder()
        self._placeholder = mx.zeros((0,), dtype=mx.float32)
        self.base_cls: Optional[type] = None
        self.streamer: Optional[BlockStreamer] = None
        self.active: tuple[str, ...] = ()
        self._stats_done: dict[str, float] = {}
        self.swaps: list[dict[str, Any]] = []
        self.loaded_block: Optional[int] = None

        # --- DiT block tensors
        parts = {"dit": dit_index}
        self.dit_names: list[list[str]] = [[] for _ in range(n_blocks)]
        local_dit: dict[str, str] = {}
        for name in dit_index.entries:
            m = BLOCK_RX.match(name)
            if m:
                i = int(m.group(1))
                if i >= n_blocks:
                    raise ValueError(f"DiT shards have block {i} but config depth is {n_blocks}")
                full = f"dit{MultiIndex.SEP}{name}"
                self.dit_names[i].append(full)
                local_dit[full] = m.group(2)
        self._local: dict[str, tuple] = {}     # full name -> ("dit", local path) | ("lora", key, module, tail)
        expected = self._shell_shapes()
        ref = sorted(local_dit[n] for n in self.dit_names[0])
        for i in range(n_blocks):
            locs = sorted(local_dit[n] for n in self.dit_names[i])
            if locs != ref:
                raise ValueError(f"block {i} has a different tensor set from block 0: streaming needs identical blocks")
        missing = sorted(set(expected) - set(ref))
        extra = sorted(set(ref) - set(expected))
        if missing or extra:
            raise ValueError(f"block tensors do not match the shell (missing in shards: {missing[:5]}, "
                             f"unexpected: {extra[:5]})")
        for full, loc in local_dit.items():
            self._local[full] = ("dit", loc)
        self._expected = expected

        # --- LoRA tensors (header only) + alpha scales
        from longcat_video_avatar.lora import decode_module_name

        self.lora_paths = {k: Path(v) for k, v in (loras or {}).items()}
        self.lora_names: dict[str, list[list[str]]] = {}
        self.lora_modules: dict[str, dict[str, int]] = {}          # key -> {local module: number of ups}
        self.scales: dict[str, list[dict[str, float]]] = {}       # key -> per block {module: scale}
        for key, path in self.lora_paths.items():
            ns = f"lora:{key}"
            idx = SafetensorsIndex([path])
            parts[ns] = idx
            names: list[list[str]] = [[] for _ in range(n_blocks)]
            mods_per_block: list[dict[str, set]] = [dict() for _ in range(n_blocks)]
            scale_names: list[dict[str, str]] = [dict() for _ in range(n_blocks)]
            for name in idx.entries:
                mod, tail = decode_module_name(name)
                m = BLOCK_RX.match(mod + ".")
                if not m:
                    raise NotImplementedError(f"LoRA {key} targets {mod}, outside the DiT blocks: not streamable")
                i, local_mod = int(m.group(1)), m.group(2)[:-1]
                full = f"{ns}{MultiIndex.SEP}{name}"
                if tail == "alpha_scale":
                    scale_names[i][local_mod] = name
                    continue
                names[i].append(full)
                mods_per_block[i].setdefault(local_mod, set()).add(tail)
                self._local[full] = ("lora", key, local_mod, tail)
            ref_mods = {mname: sorted(t) for mname, t in mods_per_block[0].items()}
            for i in range(n_blocks):
                if {mname: sorted(t) for mname, t in mods_per_block[i].items()} != ref_mods:
                    raise ValueError(f"LoRA {key}: block {i} targets differ from block 0")
            self.lora_names[key] = names
            self.lora_modules[key] = {mname: sum(1 for t in tails if t.startswith("lora_up."))
                                      for mname, tails in ref_mods.items()}
            per_block = []
            for i in range(n_blocks):
                d = {}
                for mname in ref_mods:
                    sn = scale_names[i].get(mname)
                    # attach_lora: float(alpha_scale.item()) if present, else default_alpha_scale 0.5
                    d[mname] = float(idx.to_mlx(sn).item()) if sn else 0.5
                per_block.append(d)
            self.scales[key] = per_block
        self.index = MultiIndex(parts)
        self.block_bytes = max(self.index.nbytes(g) for g in self.dit_names) if n_blocks else 0

    # -- shell structure ---------------------------------------------------------------
    def _shell_shapes(self) -> dict[str, tuple]:
        from mlx.utils import tree_flatten

        out = {}
        for k, v in tree_flatten(self.shell.parameters()):
            if ".loras." in k:
                continue
            out[k.replace(".base.", ".")] = tuple(v.shape)
        return out

    def _module(self, local_mod: str):
        return _get(self.shell, local_mod)

    def _attach_key(self, key: str) -> None:
        from longcat_video_avatar.runtime_lora import LoRALinear, _set_module
        import mlx.nn as nn

        ph = self._placeholder
        for mname, n_ups in self.lora_modules[key].items():
            target = self._module(mname)
            if not isinstance(target, LoRALinear):
                if not isinstance(target, (nn.Linear, nn.QuantizedLinear)):
                    raise TypeError(f"LoRA target {mname} is {type(target).__name__}, expected Linear")
                target = LoRALinear(target)
                if "." in mname:
                    _set_module(self.shell, mname, target)
                else:
                    setattr(self.shell, mname, target)
            target.add(key, ph, [ph] * max(1, n_ups), self.scales[key][0][mname])

    def _detach_key(self, key: str) -> None:
        from longcat_video_avatar.runtime_lora import LoRALinear

        for _, m in self.shell.named_modules():
            if isinstance(m, LoRALinear) and key in m.loras:
                del m.loras[key]
                del m.mult[key]

    def set_scales(self, i: int) -> None:
        from longcat_video_avatar.runtime_lora import LoRALinear

        for key in self.active:
            for mname in self.lora_modules[key]:
                m = self._module(mname)
                if isinstance(m, LoRALinear) and key in m.loras:
                    m.loras[key]["scale"] = self.scales[key][i][mname]

    # -- mount -------------------------------------------------------------------------
    def _unmount(self) -> None:
        if self.base_cls is not None:
            self.shell.__class__ = self.base_cls
            self.base_cls = None
        d = self.dit.__dict__
        if "blocks" in d:
            object.__delattr__(self.dit, "blocks")
        self.dit["blocks"] = [self.shell]

    def mount(self, remount: bool = False) -> None:
        """(Re)mount the fast path through the generator's own ``_mount_fast`` with ``blocks == [shell]``."""
        from longcat_video_avatar import fast

        self._unmount()
        self.clear()
        self.set_scales(0)
        if remount:
            fast.remove(self.dit)
        self._mount_fn()
        base = type(self.shell)
        self.base_cls = base
        self.shell.__class__ = _recording_class(base)
        object.__setattr__(self.shell, "_shutter_rec", self.rec)
        fns = getattr(self.dit, "_compiled_fns", None)
        if isinstance(fns, dict):
            object.__setattr__(self.dit, "_compiled_fns", _RecordingFns(fns, self.rec, self.shell))
        self.dit.pop("blocks", None)
        object.__setattr__(self.dit, "blocks", StreamedBlocks(self))

    def set_active(self, keys: Iterable[str], remount: bool = True) -> None:
        keys = tuple(k for k in self.lora_paths if k in set(keys))
        if keys == self.active and self.streamer is not None:
            return
        t0 = time.time()
        self._close_streamer()
        for k in self.active:
            if k not in keys:
                self._detach_key(k)
        for k in keys:
            if k not in self.active:
                self._attach_key(k)
        old = self.active
        self.active = keys
        self._mx.clear_cache()
        if remount or self.base_cls is None:
            self.mount(remount=self.base_cls is not None)
        groups = [list(self.dit_names[i]) + [n for k in keys for n in self.lora_names[k][i]]
                  for i in range(self.n_blocks)]
        self.streamer = BlockStreamer(self.index, groups, window=self.window, readahead=self.readahead,
                                      cyclic=True, nocache=self.nocache)
        if old != keys:
            self.swaps.append({"from": list(old), "to": list(keys), "seconds": round(time.time() - t0, 2)})

    def on_mults(self, mults: dict[str, float]) -> None:
        """Called with each step's LoRA multipliers (before ``set_lora_multipliers``): keep exactly the LoRAs with a
        non-zero multiplier attached and streamed (LC_LORA_SWAP=1 residency)."""
        want = [k for k in self.lora_paths if float(mults.get(k, 0.0)) != 0.0]
        self.set_active(want)

    # -- per-block load / clear --------------------------------------------------------
    def _assign(self, full: str, arr) -> None:
        kind = self._local[full]
        if kind[0] == "dit":
            loc = kind[1]
            mod_path, attr = loc.rsplit(".", 1) if "." in loc else ("", loc)
            mod = self._module(mod_path) if mod_path else self.shell
            base = getattr(mod, "base", None) if hasattr(mod, "loras") else None
            target = base if base is not None else mod
            exp = self._expected.get(loc)
            if exp is not None and tuple(arr.shape) != exp:
                raise ValueError(f"{full}: shape {tuple(arr.shape)}, shell expects {exp}")
            target[attr] = arr
        else:
            _, key, mname, tail = kind
            lo = self._module(mname).loras[key]
            arr = arr.astype(self.lora_dtype)          # attach_lora: .astype(dtype), bf16 by default
            if tail == "lora_down.weight":
                lo["down"] = arr
            elif tail == "lora_up.weight":
                lo["ups"][0] = arr
            elif tail.startswith("lora_up.blocks."):
                lo["ups"][int(tail.split(".")[2])] = arr
            else:
                raise ValueError(f"unknown LoRA tensor {full}")

    def load(self, i: int, tensors: dict) -> None:
        for name, arr in tensors.items():
            self._assign(name, arr)
        self.set_scales(i)
        self.loaded_block = i

    def clear(self) -> None:
        """Placeholders everywhere a block tensor goes: no block's weights outlive its turn."""
        ph = self._placeholder
        names = self.dit_names[0] + [n for k in self.active for n in self.lora_names[k][0]] if self.n_blocks else []
        for full in names:
            kind = self._local[full]
            if kind[0] == "dit":
                loc = kind[1]
                mod_path, attr = loc.rsplit(".", 1) if "." in loc else ("", loc)
                mod = self._module(mod_path) if mod_path else self.shell
                target = mod.base if hasattr(mod, "loras") else mod
                target[attr] = ph
            else:
                _, key, mname, tail = kind
                m = self._module(mname)
                if key not in getattr(m, "loras", {}):
                    continue
                lo = m.loras[key]
                if tail == "lora_down.weight":
                    lo["down"] = ph
                elif tail == "lora_up.weight":
                    lo["ups"][0] = ph
                else:
                    lo["ups"][int(tail.split(".")[2])] = ph
        self.loaded_block = None

    def iterate(self):
        mx = self._mx
        if self.streamer is None:
            self.set_active(self.active)
        streamer = self.streamer
        for i in range(self.n_blocks):
            with streamer.block(i) as tensors:
                try:
                    self.load(i, tensors)
                    tensors = None
                    yield self.shell
                    self.rec.flush(mx)       # block i's outputs exist before its weights go
                finally:
                    self.rec.pending.clear()
                    self.clear()

    # -- lifetime / stats --------------------------------------------------------------
    def _close_streamer(self) -> None:
        if self.streamer is not None:
            st = self.streamer.stats()
            for k, v in st.items():
                if k.endswith("_peak"):
                    self._stats_done[k] = max(self._stats_done.get(k, 0), v)
                else:
                    self._stats_done[k] = self._stats_done.get(k, 0) + v
            self.streamer.close()
            self.streamer = None

    def stats(self) -> dict[str, Any]:
        out = dict(self._stats_done)
        if self.streamer is not None:
            for k, v in self.streamer.stats().items():
                out[k] = max(out.get(k, 0), v) if k.endswith("_peak") else out.get(k, 0) + v
        out.update({"window": self.window, "readahead": self.readahead, "block_bytes": self.block_bytes,
                    "lora_swaps": list(self.swaps)})
        for k in ("read_seconds", "wait_seconds", "convert_seconds"):
            if k in out:
                out[k] = round(out[k], 3)
        return out

    def close(self) -> None:
        self._close_streamer()
        self.rec.pending.clear()
        try:
            self.clear()
        except Exception:
            pass
        self.index.close()


# ----------------------------------------------------------------------------- build


def build_streamed_dit(dit_folder: Path, loras: dict[str, Path], *, window: int, mount: Callable[[Any], None],
                       initial_loras: Optional[Iterable[str]] = None, nocache: bool = False,
                       readahead: Optional[int] = None):
    """Build the streamed DiT. ``mount(dit)`` must mount the fast path on ``dit`` (the generator's
    ``_mount_fast`` with ``self.dit`` set to it). Returns the skeleton DiT with ``dit._shutter_stream``."""
    import mlx.core as mx
    import mlx.nn as nn
    from longcat_video_avatar.models.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel

    dit_folder = Path(dit_folder)
    cfg = json.loads((dit_folder / "config.json").read_text())
    depth = int(cfg.get("depth", 48))
    q = cfg.get("quantization")

    def quantize(model, prefix_filter: Callable[[str], bool]):
        if not q:
            return
        mods = {m for m in q["modules"] if prefix_filter(m)}
        mods4 = {m for m in q.get("modules4", []) if prefix_filter(m)}
        if mods4:   # same order as load_dit: 4-bit first, then 8-bit
            nn.quantize(model, group_size=int(q["group_size"]), bits=4,
                        class_predicate=lambda path, m: isinstance(m, nn.Linear) and path in mods4)
        nn.quantize(model, group_size=int(q["group_size"]), bits=int(q["bits"]),
                    class_predicate=lambda path, m: isinstance(m, nn.Linear) and path in mods)

    if q:   # every block must be quantized at the same local paths (one shell serves all)
        def local_set(key, i):
            return sorted(m[len(f"blocks.{i}."):] for m in q.get(key, []) if m.startswith(f"blocks.{i}."))
        for key in ("modules", "modules4"):
            ref = local_set(key, 0)
            for i in range(1, depth):
                if local_set(key, i) != ref:
                    raise ValueError(f"quantization {key}: block {i} differs from block 0; cannot share a shell")

    # skeleton: everything but the blocks
    dit = LongCatVideoAvatarTransformer3DModel.from_config({**cfg, "depth": 0})
    quantize(dit, lambda m: not m.startswith("blocks."))
    # shell: block 0's structure (lazy params, never evaluated)
    holder = LongCatVideoAvatarTransformer3DModel.from_config({**cfg, "depth": 1})
    quantize(holder, lambda m: m.startswith("blocks.0."))
    shell = holder.blocks[0]
    del holder

    index = SafetensorsIndex(_dit_shards(dit_folder))
    rest = [n for n in index.entries if not BLOCK_RX.match(n)]
    dit.load_weights([(n, index.to_mlx(n)) for n in rest], strict=False)
    dit.pop("blocks", None)
    mx.eval(dit.parameters())

    holder_ref: dict[str, Any] = {}

    def do_mount():
        mount(dit)

    sd = StreamedDiT(dit, shell, depth, index, loras, window=window, readahead=readahead, nocache=nocache,
                     mount=do_mount)
    holder_ref["sd"] = sd
    object.__setattr__(dit, "_shutter_stream", sd)
    keys = list(loras) if initial_loras is None else [k for k in loras if k in set(initial_loras)]
    sd.set_active(keys)
    return dit


# ----------------------------------------------------------------------------- LoRA multiplier hook


def install_lora_hook(gen_mod) -> None:
    """Route the generation module's ``set_lora_multipliers(self.dit, mults)`` to the streamed shell (idempotent;
    a plain DiT passes straight through)."""
    orig = getattr(gen_mod, "set_lora_multipliers", None)
    if orig is None or getattr(orig, "_shutter_wrapped", False):
        return

    def set_lora_multipliers(dit, mults):
        sd = dit.__dict__.get("_shutter_stream") if hasattr(dit, "__dict__") else None
        if sd is None:
            return orig(dit, mults)
        sd.on_mults(mults)
        return orig(sd.shell, mults)

    set_lora_multipliers._shutter_wrapped = True
    set_lora_multipliers._shutter_orig = orig
    gen_mod.set_lora_multipliers = set_lora_multipliers


def release_dit(dit) -> None:
    """Drop a (possibly streamed) DiT's resources; the caller drops its reference."""
    sd = dit.__dict__.get("_shutter_stream") if dit is not None and hasattr(dit, "__dict__") else None
    if sd is not None:
        sd.close()
    gc.collect()
