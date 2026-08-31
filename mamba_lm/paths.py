"""Repository-root path resolution shared across packages."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    """Resolve a relative path against the CWD, then against the repo root.

    Defaults like ``data_dir="data"`` should mean the repo's ``data/`` whether
    the entry point was launched from the repo root, from ``forecast/``, or
    from an IDE with its own working directory.
    """
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    from_root = REPO_ROOT / candidate
    return from_root if from_root.exists() else candidate


def anchor_to_repo(path: str | Path) -> Path:
    """Anchor relative paths to the repo root (for checkpoints, logs, etc.)."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate
