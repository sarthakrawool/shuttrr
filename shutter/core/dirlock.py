"""Directory locks shared with other GPU tools on the Mac (the `gpulock.sh` convention).

Other tools take the GPU by creating lock directories with `mkdir` (atomic): for example
`~/studio-gpu.lock.d` (shared by every GPU user, holds an `owner` file) and
`~/ImaginationStudio/queue/.render.lock` (the studio render queue, holds `owner_speed` and
`pid`). `DirLocks` takes the same directories in the same order, all or nothing, and only
ever removes directories it created itself. A lock that looks stale is reported, never
removed: the user decides.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Iterable, Optional, Union

__all__ = ["DirLocks", "read_owner"]

_OWNER_FILES = ("owner", "owner_speed", "pid")


def read_owner(path: str) -> dict[str, Optional[str]]:
    """What a lock dir says about its holder: {"owner", "owner_speed", "pid"} (None when
    the file is missing or unreadable)."""
    out: dict[str, Optional[str]] = {}
    for name in _OWNER_FILES:
        try:
            with open(os.path.join(path, name), "r", encoding="utf-8", errors="replace") as f:
                out[name] = f.read(200).strip() or None
        except OSError:
            out[name] = None
    return out


def _pid_alive(pid: int) -> Optional[bool]:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def describe_holder(path: str) -> str:
    """Human description of who holds the lock dir `path`."""
    info = read_owner(path)
    who = info.get("owner") or info.get("owner_speed")
    pid = info.get("pid")
    parts = [who or "an unknown owner"]
    if pid:
        parts.append(f"pid {pid}")
        try:
            alive = _pid_alive(int(pid))
        except ValueError:
            alive = None
        if alive is False:
            parts.append("that process is gone; the lock may be stale, check it and remove it "
                         "by hand if so")
    elif not who:
        parts.append("no owner file; it may be stale, check it before removing it by hand")
    return ", ".join(parts)


class DirLocks:
    """Take one or more `mkdir` locks in order, all or nothing.

    `try_acquire()` returns None on success or a human reason naming the holder. On
    success each created dir gets `owner` and `pid` files. `release()` removes only the
    dirs this instance created (in reverse order). Use as a context manager to block
    until acquired (polling `poll_s`).
    """

    def __init__(self, paths: Union[str, Iterable[str]], owner: str = "shutter",
                 poll_s: float = 15.0):
        if isinstance(paths, (str, os.PathLike)):
            paths = [paths]
        self.paths = [os.path.abspath(os.path.expanduser(os.fspath(p))) for p in paths]
        self.owner = owner
        self.poll_s = poll_s
        self._created: list[str] = []

    @property
    def held(self) -> bool:
        return bool(self._created)

    def _mark(self, path: str) -> None:
        for name, value in (("owner", self.owner), ("pid", str(os.getpid()))):
            try:
                with open(os.path.join(path, name), "w", encoding="utf-8") as f:
                    f.write(value + "\n")
            except OSError:
                pass

    def try_acquire(self) -> Optional[str]:
        """One attempt at every lock. None on success, else why not (and nothing held)."""
        if self._created and len(self._created) == len(self.paths):
            return None
        self.release()
        for path in self.paths:
            parent = os.path.dirname(path)
            try:
                if parent and not os.path.isdir(parent):
                    os.makedirs(parent, exist_ok=True)
                os.mkdir(path)
            except FileExistsError:
                reason = f"GPU lock {path} is held by {describe_holder(path)}"
                self.release()
                return reason
            except OSError as e:
                self.release()
                return f"cannot take GPU lock {path}: {e}"
            self._created.append(path)
            self._mark(path)
        return None

    def release(self) -> None:
        """Remove the dirs this instance created, newest first. Never touches others."""
        created, self._created = self._created, []
        for path in reversed(created):
            try:
                info = read_owner(path)
            except Exception:
                info = {}
            # Defensive: if someone replaced our dir with theirs, leave it alone.
            if info.get("pid") not in (None, str(os.getpid())):
                continue
            shutil.rmtree(path, ignore_errors=True)

    def holders(self) -> list[dict[str, Any]]:
        """Every configured lock that exists and is not ours, with its owner files."""
        out = []
        for path in self.paths:
            if path in self._created or not os.path.isdir(path):
                continue
            out.append({"path": path, **read_owner(path), "label": describe_holder(path)})
        return out

    def __enter__(self) -> "DirLocks":
        import time

        while True:
            if self.try_acquire() is None:
                return self
            time.sleep(self.poll_s)

    def __exit__(self, *exc: Any) -> None:
        self.release()

    def __repr__(self) -> str:  # pragma: no cover
        return f"DirLocks({self.paths!r}, owner={self.owner!r}, held={self.held})"
