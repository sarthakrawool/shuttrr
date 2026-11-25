"""Fish Audio Drama 3 (text to speech) on Shuttrr.

The model code is TensorFold's MLX port (`tensorfold.families.drama3`); this adapter is the part that makes it
fit and stay fast on a smaller Mac:

* loaded once and kept warm in the model pool (the old flow started a process per take and reloaded ~41 GB);
* the lossless low-memory switches are always on (TF_D3_LOWMEM, 1 GB MLX cache, no unused lm_head, encoder freed
  after the reference is encoded, windowed codec decode): same audio, ~15 GB lower peak;
* the reference voice is encoded once and its codes are kept on disk, keyed by the clip's content;
* several takes (seeds) of one line run in one job, batched in one decode when the model code offers it
  (`Drama3Generator.synthesize_batch`), one after another otherwise;
* the model folder is an option, and `variants` maps memory steps to smaller builds (calibrated 5-bit / 4-bit or
  mixed precision), so Low memory or Fast's step-down picks the largest build that fits without swap.

Request: `prompt` is the text ([directions] and <|speaker:N|> tags as Drama 3 reads them), `inputs.audio` the
reference clip, `seed` the first seed. Params: ref_text (what the clip says; needed with a clip), ref_codes
(.npy [10, frames] instead of a clip), seeds (list) or takes (count, seeds seed..seed+takes-1), temperature
(0.75), top_k (30), top_p (0.9), greedy, max_chars (300), max_frames, verify (False), verify_retries (2),
stop_rule (True), save_codes (False), batch (False: decode the takes together; faster, same quality, but
each take differs at float level from the one-at-a-time take of the same seed), mode / efficiency / rung
(docs/MODES.md).

Options (registry config): tensorfold_src (folder that holds the `tensorfold` package; default
~/models/drama3-mem/code/src), variants ({step id: model folder}), lossless_variants (step ids whose build gives
the same audio as the main folder; default: only the main folder), lowmem (True), cache_gb (1), codec_window
(512), free_encoder (True), keep_warm (True), batch_takes (False), overhead_gb (2.0), peak_gb (override the estimate),
default_mode / default_efficiency.

Output: a 44.1 kHz mono WAV at `output`; more takes are written next to it as `<name>.s<seed>.wav`.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from shutter.adapters.base import Adapter, RunContext
from shutter.core import modes as M
from shutter.types import GB, Job, JobRequest, MemoryEstimate

DEFAULT_SRC = "~/models/drama3-mem/code/src"
FRAME_RATE = 44100 / 2048
#: env name -> (option, default); set before TensorFold is imported (it reads some at import time)
LOWMEM_ENV = {"TF_D3_LOWMEM": ("lowmem", True), "TF_D3_DROP_LMHEAD": ("lowmem", True),
              "TF_D3_FREE_ENCODER": ("free_encoder", True)}
#: bytes of a build's files that are not resident with the lossless switches on (full lm_head, codec encoder)
DROPPED_BYTES = int(1.8 * GB)
#: without TF_D3_LOWMEM the load holds the experts twice for a moment
LOAD_SPIKE = 0.53


def _truthy(v: Any) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on") if isinstance(v, str) else bool(v)


def build_bytes(model_dir: Path) -> int:
    """Size of a converted build's weight files (symlinks followed: small builds link the shared parts)."""
    total = 0
    for f in Path(model_dir).glob("*.safetensors"):
        try:
            total += f.stat().st_size
        except OSError:
            pass
    return total


def file_digest(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def take_seeds(request: JobRequest) -> list[int]:
    p = request.params
    if p.get("seeds"):
        seeds = [int(s) for s in p["seeds"]]
    else:
        seeds = [int(request.seed) + i for i in range(max(1, int(p.get("takes", 1))))]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be different from each other")
    return seeds


def take_path(output: Path, seeds: list[int], i: int) -> Path:
    """The first take is `output`; the others sit next to it as <name>.s<seed>.wav."""
    return output if i == 0 else output.with_name(f"{output.stem}.s{seeds[i]}{output.suffix}")


class Drama3Adapter(Adapter):
    family = "fish-drama3"
    kind = "audio"
    extension = "wav"

    # ---- options
    def _opt(self, key: str, default: Any) -> Any:
        return self.options.get(key, default)

    @property
    def source_dir(self) -> Path:
        return Path(str(self._opt("tensorfold_src", DEFAULT_SRC))).expanduser()

    def env(self) -> dict[str, str]:
        e = {name: "1" if _truthy(self._opt(opt, default)) else "0" for name, (opt, default) in LOWMEM_ENV.items()}
        e["TF_D3_CACHE_GB"] = str(self._opt("cache_gb", 1))
        e["TF_D3_CODEC_WINDOW"] = str(int(self._opt("codec_window", 512)))
        return e

    # ---- builds and memory steps
    def builds(self) -> dict[str, Path]:
        """Step id -> model folder. Steps without a variant of their own use the nearest larger one."""
        variants = {k: Path(str(v)).expanduser() for k, v in dict(self._opt("variants", {}) or {}).items()}
        unknown = set(variants) - set(M.RUNG_IDS)
        if unknown:
            raise ValueError(f"variants: unknown step {sorted(unknown)} (known: {', '.join(M.RUNG_IDS)})")
        out, current = {}, self.model_dir
        for rid in M.RUNG_IDS:
            current = variants.get(rid, current)
            out[rid] = current
        return out

    def is_lossless(self, rung_id: str) -> bool:
        build = self.builds()[rung_id]
        if build == self.model_dir:
            return True
        return rung_id in set(self._opt("lossless_variants", []) or [])

    def estimate_build(self, model_dir: Path, verify: bool = False) -> MemoryEstimate:
        if self._opt("peak_gb", None) is not None and Path(model_dir) == self.model_dir:
            peak = int(float(self.options["peak_gb"]) * GB)
            return MemoryEstimate(peak=peak, steady=peak, note="configured peak_gb")
        files = build_bytes(model_dir)
        if not files:
            # the folder is not on this machine (or not converted yet): the 8-bit build's size
            files = int(43.2 * GB)
        lowmem = _truthy(self._opt("lowmem", True))
        steady = files - (DROPPED_BYTES if lowmem else 0) + int(float(self._opt("cache_gb", 1)) * GB)
        overhead = int(float(self._opt("overhead_gb", 2.0)) * GB) + (int(1.6 * GB) if verify else 0)
        peak = steady + overhead + (0 if lowmem else int(files * LOAD_SPIKE))
        return MemoryEstimate(peak=peak, steady=steady,
                              note=f"{Path(model_dir).name}: files {files / GB:.1f} GB"
                                   + ("" if lowmem else ", load spike (lowmem off)"))

    def rung_estimates(self, request: JobRequest) -> dict[str, MemoryEstimate]:
        verify = _truthy(request.params.get("verify", False))
        return {rid: self.estimate_build(d, verify) for rid, d in self.builds().items()}

    def _budget(self, request: JobRequest, ctx: Optional[RunContext]) -> Optional[int]:
        b = request.params.get("budget_bytes")
        if b is not None:
            return int(b)
        if ctx is None:
            return None
        try:
            s = ctx.memory_sample() or {}
        except Exception:
            return None
        if not s.get("available"):
            return None
        return int(s["available"]) - int((ctx.settings or {}).get("headroom_bytes") or 0)

    def choose(self, request: JobRequest, ctx: Optional[RunContext] = None) -> dict[str, Any]:
        defaults = {k[len("default_"):]: v for k, v in self.options.items()
                    if k in ("default_mode", "default_efficiency")}
        if ctx is not None:
            for k in ("default_mode", "default_efficiency"):
                if (ctx.settings or {}).get(k) is not None:
                    defaults[k[len("default_"):]] = ctx.settings[k]
        choice = M.resolve_mode(request.params, defaults)
        ests = self.rung_estimates(request)
        budget = self._budget(request, ctx)
        rid = M.fit_to_budget(ests, budget, choice.rung) if choice.step_down else choice.rung
        return {"choice": choice, "rung": rid, "estimates": ests, "budget": budget, "build": self.builds()[rid],
                "stepped_down_from": choice.rung if rid != choice.rung else None}

    def estimate(self, request: JobRequest) -> MemoryEstimate:
        pick = self.choose(request)
        return pick["estimates"][pick["rung"]]

    def pool_key(self, build: Path) -> str:
        return f"{self.model_id}/drama3/{Path(build).name}"

    def pool_keys(self, request: JobRequest) -> list[str]:
        return [self.pool_key(self.choose(request)["build"])]

    def prepare(self, request: JobRequest) -> JobRequest:
        return request.copy()

    # ---- the model code
    def tensorfold(self) -> dict[str, Any]:
        """TensorFold's Drama 3 modules, imported once with the low-memory switches in the environment."""
        cached = getattr(self, "_tf", None)
        if cached is not None:
            return cached
        for k, v in self.env().items():
            os.environ[k] = v
        src = str(self.source_dir)
        if self.source_dir.is_dir() and src not in sys.path:
            sys.path.insert(0, src)
        names = {"generate": "tensorfold.families.drama3.generate", "prompt": "tensorfold.families.drama3.prompt",
                 "sampling": "tensorfold.families.drama3.sampling", "codec": "tensorfold.families.drama3.codec"}
        try:
            self._tf = {k: importlib.import_module(v) for k, v in names.items()}
        except ImportError as e:
            raise RuntimeError(f"Fish Drama 3 needs the TensorFold package; looked in {src} "
                               f"(option tensorfold_src): {e}") from e
        return self._tf

    def load(self, build: Path) -> Any:
        tf = self.tensorfold()
        started = time.perf_counter()
        gen = tf["generate"].Drama3Generator(build)
        gen.shuttr_load_s = time.perf_counter() - started
        return gen

    # ---- the reference voice
    def reference(self, gen: Any, request: JobRequest, ctx: RunContext) -> tuple[list, dict[str, Any]]:
        import numpy as np

        tf = self.tensorfold()
        p = request.params
        clip, codes_path = request.inputs.get("audio") or p.get("ref_audio"), p.get("ref_codes")
        if not clip and not codes_path:
            return [], {"reference": None}
        text = str(p.get("ref_text") or "").strip()
        if not text:
            raise ValueError("a reference voice needs params.ref_text (what the reference clip says)")
        info: dict[str, Any] = {}
        if codes_path:
            codes = np.load(os.path.expanduser(str(codes_path)))
            if codes.ndim == 2 and codes.shape[0] != 10 and codes.shape[1] == 10:
                codes = codes.T
            info["reference"] = "codes"
        else:
            clip = Path(os.path.expanduser(str(clip)))
            codec_file = Path(gen.model_dir) / "codec.safetensors"
            codec_id = [codec_file.stat().st_size if codec_file.exists() else 0]
            ran: list[float] = []

            def encode() -> dict[str, Any]:
                started = time.perf_counter()
                out = {"codes": np.array(self.encoder(gen).encode(tf["codec"].load_audio(clip)))}
                ran.append(time.perf_counter() - started)
                return out

            codes = np.asarray(ctx.cache.get_or_compute("drama3-ref", [file_digest(clip), *codec_id], encode)["codes"])
            info["reference"] = "encoded" if ran else "cached"
            if ran:
                info["reference_encode_s"] = round(ran[0], 3)
        info["reference_frames"] = int(codes.shape[1])
        if _truthy(self._opt("free_encoder", True)) and gen.codec is not None \
                and not getattr(gen.codec, "encoder_freed", False) and hasattr(gen.codec, "free_encoder"):
            gen.codec.free_encoder()
        return [tf["prompt"].Reference(transcript=text, codes=codes)], info

    def encoder(self, gen: Any) -> Any:
        """A codec that can encode: the warm one, or (its encoder was freed) a second copy for this one clip."""
        if gen.codec is not None and not getattr(gen.codec, "encoder_freed", False):
            return gen.codec
        return self.tensorfold()["codec"].Codec(Path(gen.model_dir) / "codec.safetensors")

    # ---- run
    def sampling(self, request: JobRequest) -> Any:
        p = request.params
        return self.tensorfold()["sampling"].Sampling(
            temperature=float(p.get("temperature", self._opt("temperature", 0.75))), top_k=int(p.get("top_k", 30)),
            top_p=float(p.get("top_p", 0.9)), greedy=_truthy(p.get("greedy", False)))

    def synthesize(self, gen: Any, text: str, seeds: list[int], kw: dict[str, Any],
                   between: Callable[[int], None]) -> tuple[list[tuple[Any, list]], bool]:
        """[(audio, results)] per seed, and whether the takes ran batched in one decode."""
        batch = getattr(gen, "synthesize_batch", None)
        if kw.pop("batch", False) and callable(batch) and len(seeds) > 1:
            return list(batch(text, seeds=seeds, **kw)), True
        takes = []
        for i, seed in enumerate(seeds):
            between(i)
            takes.append(gen.synthesize(text, seed=seed, **kw))
        return takes, False

    def run(self, job: Job, ctx: RunContext) -> dict[str, Any]:
        import numpy as np

        req = job.request
        p = req.params
        text = (req.prompt or "").strip()
        if not text:
            raise ValueError("prompt (the text to speak) is empty")
        seeds = take_seeds(req)
        pick = self.choose(req, ctx)
        rid, build = pick["rung"], Path(pick["build"])
        est = pick["estimates"][rid]
        tf = self.tensorfold()
        clock = time.perf_counter
        started = clock()

        ctx.report(0.0, "loading the voice model" + (f" (stepped down to {M.rung(rid)['label']})"
                                                     if pick["stepped_down_from"] else ""))
        loaded: list[float] = []

        def loader() -> Any:
            gen = self.load(build)
            loaded.append(getattr(gen, "shuttr_load_s", 0.0))
            return gen

        gen = ctx.models.get(self.pool_key(build), loader, est.steady or est.peak,
                             keep=_truthy(self._opt("keep_warm", True)))
        ctx.check_cancelled()
        ctx.report(0.05, "reading the reference voice")
        references, info = self.reference(gen, req, ctx)

        verify = _truthy(p.get("verify", False))
        done = [0]

        def log(message: str) -> None:
            ctx.check_cancelled()
            ctx.report(min(0.95, 0.1 + 0.85 * done[0] / len(seeds)), message.replace("[drama3] ", "")[:120])

        def between(i: int) -> None:
            ctx.check_cancelled()
            done[0] = i
            ctx.report(0.1 + 0.85 * i / len(seeds), f"take {i + 1} of {len(seeds)}")

        kw: dict[str, Any] = {"references": references, "sampling": self.sampling(req),
                              "max_chars": int(p.get("max_chars", 300)), "verify": verify,
                              "verify_retries": int(p.get("verify_retries", 2)),
                              "stop": _truthy(p.get("stop_rule", True)), "log": log,
                              "batch": _truthy(p.get("batch", self._opt("batch_takes", False)))}
        if p.get("max_frames"):
            kw["max_frames"] = int(p["max_frames"])
        decode_started = clock()
        takes, batched = self.synthesize(gen, text, seeds, kw, between)
        decode_wall = clock() - decode_started

        output = Path(req.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        outputs, rows = [], []
        for i, (audio, results) in enumerate(takes):
            path = take_path(output, seeds, i)
            tf["codec"].save_wav(path, audio)
            frames = sum(len(r.frames) for r in results)
            decode_s = sum(r.decode_s for r in results)
            row = {"seed": seeds[i], "output": str(path), "audio_s": round(len(audio) / 44100, 3), "frames": frames,
                   "decode_s": round(decode_s, 3), "prefill_s": round(sum(r.prefill_s for r in results), 3),
                   "frames_per_s": round(frames / max(decode_s, 1e-9), 2), "finish": [r.finish for r in results]}
            if verify:
                row["verify"] = [r.stats.get("verify") for r in results]
            if _truthy(p.get("save_codes", False)):
                codes = path.with_suffix(".codes.npy")
                np.save(codes, np.concatenate([r.frames for r in results]) if results else np.zeros((0, 11)))
                row["codes"] = str(codes)
            outputs.append(str(path))
            rows.append(row)
        audio_s = sum(r["audio_s"] for r in rows)
        result = {"output": outputs[0], "outputs": outputs, "kind": "audio", "takes": rows, "batched": batched,
                  "audio_s": round(audio_s, 3), "seconds": round(clock() - started, 3),
                  "decode_wall_s": round(decode_wall, 3),
                  "realtime_factor": round(audio_s / max(decode_wall, 1e-9), 2),
                  "warm": not loaded, "load_s": round(loaded[0], 2) if loaded else 0.0,
                  "mode": pick["choice"].mode, "rung": rid, "build": build.name,
                  "lossless": self.is_lossless(rid), "stepped_down_from": pick["stepped_down_from"],
                  "estimate_peak_bytes": est.peak, **info}
        try:
            import mlx.core as mx

            result["mlx_peak_bytes"] = int(mx.get_peak_memory())
            mx.clear_cache()
        except Exception:
            pass
        ctx.report(1.0, "done")
        return result

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["builds"] = {rid: str(path) for rid, path in self.builds().items()}
        return d
