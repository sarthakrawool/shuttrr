"""ComfyUI entry point for the Shutter node pack (V3 schema: the loader calls `comfy_entrypoint`).

Outside ComfyUI this file defines nothing; use the `shutter_comfy` package directly.
"""

try:
    from .shutter_comfy.nodes import COMFY_AVAILABLE
except ImportError:  # loaded as a plain directory (not a package): nothing to register
    COMFY_AVAILABLE = False

if COMFY_AVAILABLE:
    from .shutter_comfy.nodes import comfy_entrypoint  # noqa: F401

__all__ = ["comfy_entrypoint"] if COMFY_AVAILABLE else []
