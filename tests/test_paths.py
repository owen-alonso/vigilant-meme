"""Tests for repo-root path helpers."""

from __future__ import annotations

from pathlib import Path

from mamba_lm.paths import REPO_ROOT, anchor_to_repo, resolve_path


def test_repo_root_points_at_workspace():
    assert (REPO_ROOT / "mamba_lm").is_dir()
    assert (REPO_ROOT / "forecast").is_dir()


def test_anchor_to_repo_relative():
    assert anchor_to_repo("checkpoints/foo") == REPO_ROOT / "checkpoints/foo"


def test_resolve_path_prefers_existing_repo_path(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = REPO_ROOT / "data"
    if data_dir.exists():
        assert resolve_path("data") == data_dir
