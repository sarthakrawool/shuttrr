"""Model registry: model id -> (adapter family, model dir, options).

Models come from `config.models` ({id: {"adapter": family, "dir": ..., "options": {...}}})
plus, unless disabled, a built-in "dummy" model for trying the app. Adapter families are
imported lazily by dotted path, so a missing or broken adapter module never breaks the
engine: the model is listed with the import error instead.
"""

from __future__ import annotations

import importlib
import os
import threading
import traceback
from pathlib import Path
from typing import Any, Optional

from shutter.adapters.base import Adapter

__all__ = ["Registry", "UnknownModel", "AdapterUnavailable", "BUILTIN_FAMILIES"]

BUILTIN_FAMILIES: dict[str, str] = {
    "longcat-avatar": "shutter.adapters.longcat:LongCatAvatarAdapter",
    "fish-drama3": "shutter.adapters.drama3:Drama3Adapter",
    "wan": "shutter.adapters.wan:WanAdapter",
    "z-image": "shutter.adapters.zimage:ZImageAdapter",
    "dummy": "shutter.adapters.dummy:DummyAdapter",
    "dummy-image": "shutter.adapters.dummy:DummyImageAdapter",
}

DUMMY_MODEL_ID = "dummy"


class UnknownModel(KeyError):
    def __init__(self, model_id: str, known: list[str]):
        self.model_id = model_id
        self.known = known
        msg = f"unknown model {model_id!r}"
        msg += f"; known models: {', '.join(known)}" if known else "; no models are configured"
        super().__init__(msg)

    def __str__(self) -> str:
        return self.args[0]


class AdapterUnavailable(RuntimeError):
    pass


def _import(path: str) -> Any:
    mod, _, attr = path.partition(":")
    if not attr:
        mod, _, attr = path.rpartition(".")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


class Registry:
    def __init__(self, config: Any, families: Optional[dict[str, Any]] = None):
        self.config = config
        self.families: dict[str, Any] = dict(BUILTIN_FAMILIES)
        if families:
            self.families.update(families)
        self._classes: dict[str, type] = {}
        self._errors: dict[str, str] = {}
        self._adapters: dict[str, Adapter] = {}
        self._lock = threading.RLock()

    # -- families ---------------------------------------------------------------------

    def register(self, family: str, target: Any) -> None:
        """Add or replace a family: a class or a 'module:Class' dotted path."""
        with self._lock:
            self.families[family] = target
            self._classes.pop(family, None)
            self._errors.pop(family, None)
            for mid in [m for m, a in self._adapters.items() if self._family_of(m) == family]:
                self._adapters.pop(mid, None)

    def adapter_class(self, family: str) -> type:
        """Import the family's class (cached). Raises AdapterUnavailable with the reason."""
        with self._lock:
            if family in self._classes:
                return self._classes[family]
            target = self.families.get(family)
            if target is None and (":" in family):
                target = family  # a dotted path used directly as "adapter"
            if target is None:
                raise AdapterUnavailable(
                    f"no adapter family {family!r} (known: {', '.join(sorted(self.families))})")
            try:
                cls = _import(target) if isinstance(target, str) else target
                if not (isinstance(cls, type) and issubclass(cls, Adapter)):
                    raise TypeError(f"{target!r} is not an Adapter subclass")
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                self._errors[family] = err
                raise AdapterUnavailable(f"adapter {family!r} cannot be loaded: {err}") from e
            self._classes[family] = cls
            self._errors.pop(family, None)
            return cls

    # -- models -----------------------------------------------------------------------

    def _specs(self) -> dict[str, dict[str, Any]]:
        specs: dict[str, dict[str, Any]] = {}
        if getattr(self.config, "dummy_model", True):
            specs[DUMMY_MODEL_ID] = {"adapter": "dummy"}
        for mid, spec in (getattr(self.config, "models", None) or {}).items():
            specs[mid] = dict(spec or {})
        return specs

    def model_ids(self) -> list[str]:
        return list(self._specs())

    def spec(self, model_id: str) -> dict[str, Any]:
        specs = self._specs()
        if model_id not in specs:
            raise UnknownModel(model_id, list(specs))
        spec = dict(specs[model_id])
        spec.setdefault("adapter", model_id)
        spec["dir"] = self.config.model_dir(model_id)
        spec.setdefault("options", {})
        return spec

    def _family_of(self, model_id: str) -> Optional[str]:
        try:
            return self.spec(model_id)["adapter"]
        except UnknownModel:
            return None

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._specs()

    def adapter_for(self, model_id: str) -> Adapter:
        """The (cached) adapter instance for `model_id`. Raises UnknownModel or
        AdapterUnavailable."""
        with self._lock:
            if model_id in self._adapters:
                return self._adapters[model_id]
            spec = self.spec(model_id)
            cls = self.adapter_class(spec["adapter"])
            try:
                adapter = cls(model_id, Path(spec["dir"]), dict(spec.get("options") or {}))
            except Exception as e:
                raise AdapterUnavailable(
                    f"adapter {spec['adapter']!r} failed to start for {model_id!r}: "
                    f"{type(e).__name__}: {e}") from e
            self._adapters[model_id] = adapter
            return adapter

    def list(self) -> list[dict[str, Any]]:
        out = []
        for mid in self._specs():
            spec = self.spec(mid)
            family = spec["adapter"]
            row: dict[str, Any] = {
                "id": mid, "family": family, "kind": "", "dir": spec["dir"],
                "installed": False, "error": None, "options": spec.get("options") or {},
            }
            try:
                cls = self.adapter_class(family)
                row["kind"] = getattr(cls, "kind", "")
                needs_dir = getattr(cls, "needs_model_dir", True)
                row["installed"] = os.path.isdir(spec["dir"]) or not needs_dir
            except AdapterUnavailable as e:
                row["error"] = str(e)
            out.append(row)
        return out
