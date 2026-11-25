"""ComfyUI V3 nodes: Shutter Generate (the primary, async node), Shutter Unload and Shutter Memory.

Importable without ComfyUI: `COMFY_AVAILABLE` is then False, `comfy_entrypoint` is not defined, and only
the pure helpers (`build_request`, `model_options`) are usable. Inside ComfyUI the schema API comes from
`comfy_api.latest` (docs/INTEGRATIONS.md section 1).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from . import runner
from .client import DEFAULT_URL, ShutterClient, ShutterError

try:  # inside ComfyUI
    from comfy_api.latest import ComfyExtension, io  # type: ignore
    COMFY_AVAILABLE = True
except ImportError:  # tests, plain Python
    ComfyExtension = object  # type: ignore
    io = None  # type: ignore
    COMFY_AVAILABLE = False

UNREACHABLE = "(Shuttrr not reachable: start it, then reload)"
MODES = ["default", "fast", "low_memory"]
PROGRESS_TOTAL = 1000


def model_options(url: Optional[str] = None, timeout: float = 2.0) -> List[str]:
    """Model ids from GET /v1/models for the combo, image and video models only."""
    try:
        rows = ShutterClient(url, timeout=timeout).models()
    except (ShutterError, ValueError):
        return [UNREACHABLE]
    ids = [m["id"] for m in rows if m.get("kind", "video") in ("image", "video")]
    return ids or [UNREACHABLE]


def build_request(model: str, prompt: str, negative_prompt: str = "", width: int = 0, height: int = 0,
                  frames: int = 0, fps: float = 0.0, steps: int = 0, guidance: float = 0.0, seed: int = -1,
                  memory_mode: str = "default", efficiency: float = 0.0, params_json: str = "",
                  model_id: str = "") -> Dict[str, Any]:
    """The node's widget values as a POST /v1/jobs body. Zero / empty / -1 means "model default"."""
    model = (model_id or "").strip() or model
    if not model or model == UNREACHABLE:
        raise ValueError("no Shuttrr model selected (is Shuttrr running at the configured URL?)")
    req: Dict[str, Any] = {"model": model, "prompt": prompt}
    if negative_prompt:
        req["negative_prompt"] = negative_prompt
    for k, v in (("width", width), ("height", height), ("frames", frames), ("steps", steps)):
        if v and int(v) > 0:
            req[k] = int(v)
    if fps and fps > 0:
        req["fps"] = float(fps)
    if guidance and guidance > 0:
        req["guidance"] = float(guidance)
    if seed is not None and int(seed) >= 0:
        req["seed"] = int(seed)
    params: Dict[str, Any] = {}
    if params_json and params_json.strip():
        try:
            extra = json.loads(params_json)
        except ValueError as e:
            raise ValueError(f"params_json is not valid JSON: {e}") from None
        if not isinstance(extra, dict):
            raise ValueError("params_json must be a JSON object")
        params.update(extra)
    if memory_mode and memory_mode != "default":
        params["mode"] = memory_mode
    if efficiency and efficiency > 0:
        params["efficiency"] = float(efficiency)
    if params:
        req["params"] = params
    return req


# ---------------------------------------------------------------- ComfyUI glue (only inside ComfyUI)

def _comfy_hooks(unique_id: Optional[str]) -> runner.Hooks:
    import comfy.model_management as mm  # type: ignore
    import comfy.utils  # type: ignore

    bar = comfy.utils.ProgressBar(PROGRESS_TOTAL, node_id=unique_id)

    def on_progress(p: float, step, steps, stage):
        bar.update_absolute(int(round(p * PROGRESS_TOTAL)), PROGRESS_TOTAL)

    def free_host_models():
        mm.unload_all_models()
        mm.soft_empty_cache()

    return runner.Hooks(on_progress=on_progress, is_interrupted=mm.processing_interrupted,
                        free_host_models=free_host_models)


def _output_dir() -> Optional[str]:
    try:
        import folder_paths  # type: ignore
        return os.path.join(folder_paths.get_temp_directory(), "shutter")
    except Exception:
        return None


if COMFY_AVAILABLE:
    HAS_VIDEO = hasattr(io, "Video")
    HAS_AUDIO = hasattr(io, "Audio")

    def _video_from_file(path: str):
        from comfy_api.latest import InputImpl  # type: ignore
        return InputImpl.VideoFromFile(path)

    class ShutterGenerate(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            inputs = [
                io.Combo.Input("model", options=model_options(os.environ.get("SHUTTER_URL"))),
                io.String.Input("prompt", multiline=True, default=""),
                io.String.Input("negative_prompt", multiline=True, default="", optional=True),
                io.Int.Input("width", default=0, min=0, max=8192, step=16, tooltip="0 = model default"),
                io.Int.Input("height", default=0, min=0, max=8192, step=16, tooltip="0 = model default"),
                io.Int.Input("frames", default=0, min=0, max=4096, tooltip="video models; 0 = model default"),
                io.Float.Input("fps", default=0.0, min=0.0, max=120.0, step=0.5, tooltip="0 = model default"),
                io.Int.Input("steps", default=0, min=0, max=1000, tooltip="0 = model default"),
                io.Float.Input("guidance", default=0.0, min=0.0, max=50.0, step=0.1, tooltip="0 = model default"),
                io.Int.Input("seed", default=-1, min=-1, max=2**31 - 1, tooltip="-1 = random (Shuttrr picks)"),
                io.Combo.Input("memory_mode", options=MODES, default="default",
                               tooltip="Shuttrr's memory mode (docs/MODES.md)"),
                io.Float.Input("efficiency", default=0.0, min=0.0, max=1.0, step=0.05,
                               tooltip="low_memory slider; 0 = Shuttrr's default"),
                io.Boolean.Input("free_comfy_models", default=True,
                                 tooltip="unload ComfyUI's models first when Shuttrr says room is short"),
                io.Boolean.Input("keep_loaded", default=True,
                                 tooltip="off: Shuttrr frees its idle models after this request"),
                io.String.Input("server_url", default=os.environ.get("SHUTTER_URL", DEFAULT_URL)),
                io.String.Input("model_id", default="", optional=True,
                                tooltip="overrides the model combo when set"),
                io.String.Input("params_json", default="", multiline=True, optional=True,
                                tooltip="extra adapter params as a JSON object"),
                io.Image.Input("image", optional=True),
                io.Mask.Input("mask", optional=True),
            ]
            outputs = [io.Image.Output(display_name="images")]
            if HAS_AUDIO:
                inputs.append(io.Audio.Input("audio", optional=True))
            if HAS_VIDEO:
                outputs.append(io.Video.Output(display_name="video"))
            if HAS_AUDIO:
                outputs.append(io.Audio.Output(display_name="audio"))
            return io.Schema(node_id="ShutterGenerate", display_name="Shuttrr Generate", category="Shuttrr",
                             description="Run a Shuttrr model (image or video) over Shuttrr's HTTP API.",
                             inputs=inputs, outputs=outputs, hidden=[io.Hidden.unique_id])

        @classmethod
        async def execute(cls, model, prompt, width, height, frames, fps, steps, guidance, seed, memory_mode,
                          efficiency, free_comfy_models, keep_loaded, server_url, negative_prompt="",
                          model_id="", params_json="", image=None, mask=None, audio=None) -> "io.NodeOutput":
            import comfy.model_management as mm  # type: ignore
            req = build_request(model, prompt, negative_prompt, width, height, frames, fps, steps, guidance,
                                seed, memory_mode, efficiency, params_json, model_id)
            client = ShutterClient(server_url)
            try:
                res = await runner.generate(client, req, image=image, mask=mask, audio=audio,
                                            hooks=_comfy_hooks(cls.hidden.unique_id),
                                            free_host_models=free_comfy_models, keep_loaded=keep_loaded,
                                            out_dir=_output_dir())
            except runner.Interrupted:
                mm.throw_exception_if_processing_interrupted()
                raise
            if res.images is None:
                raise RuntimeError(f"Shuttrr wrote {os.path.basename(res.output_path)}, which is neither an "
                                   "image nor a video")
            from .convert import to_tensor
            out = [to_tensor(res.images)]
            if HAS_VIDEO:
                out.append(_video_from_file(res.output_path) if res.kind == "video" else None)
            if HAS_AUDIO:
                a = res.audio
                out.append({"waveform": to_tensor(a["waveform"]), "sample_rate": a["sample_rate"]} if a else None)
            return io.NodeOutput(*out)

    class ShutterUnload(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="ShutterUnload", display_name="Shuttrr Unload", category="Shuttrr",
                             description="Free Shuttrr's idle model components (POST /v1/pool/release).",
                             inputs=[io.String.Input("server_url", default=os.environ.get("SHUTTER_URL", DEFAULT_URL))],
                             outputs=[io.String.Output(display_name="released")], is_output_node=True)

        @classmethod
        async def execute(cls, server_url) -> "io.NodeOutput":
            import asyncio
            r = await asyncio.to_thread(ShutterClient(server_url).release, None)
            return io.NodeOutput(json.dumps(r))

    class ShutterMemory(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="ShutterMemory", display_name="Shuttrr Memory", category="Shuttrr",
                             description="GET /v1/memory as JSON text.",
                             inputs=[io.String.Input("server_url", default=os.environ.get("SHUTTER_URL", DEFAULT_URL)),
                                     io.String.Input("model", default="", optional=True)],
                             outputs=[io.String.Output(display_name="memory")], is_output_node=True)

        @classmethod
        async def execute(cls, server_url, model="") -> "io.NodeOutput":
            import asyncio
            r = await asyncio.to_thread(ShutterClient(server_url).memory, model or None)
            return io.NodeOutput(json.dumps(r, indent=1))

    class ShutterExtension(ComfyExtension):
        async def get_node_list(self):
            return [ShutterGenerate, ShutterUnload, ShutterMemory]

    async def comfy_entrypoint() -> ShutterExtension:
        return ShutterExtension()
