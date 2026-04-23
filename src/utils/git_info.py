"""Helpers to capture git state at runtime for experiment reproducibility."""

from __future__ import annotations

import subprocess
import threading
from functools import lru_cache


def _run(args: list[str]) -> str | None:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=5, check=False)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


_lock = threading.Lock()


@lru_cache(maxsize=1)
def _get_git_info_locked() -> dict[str, str | bool | None]:
    commit = _run(["git", "rev-parse", "HEAD"])
    # --untracked-files=no skips walking large untracked dirs like outputs/
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    return {
        "commit": commit,
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status) if status is not None else None,
    }


def get_git_info() -> dict[str, str | bool | None]:
    """Return current repo git state: commit, branch, dirty flag.

    Cached per-process. Thread-safe on first call — benchmarks fan out games
    via ThreadPoolExecutor, so without the lock multiple workers would race
    and fork parallel git subprocesses. Returns None values outside a git repo.
    """
    with _lock:
        return _get_git_info_locked()
