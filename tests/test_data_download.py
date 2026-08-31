"""Tiny Shakespeare download integrity."""

from __future__ import annotations

import pytest

from mamba_lm.data import TINY_SHAKESPEARE_MIN_BYTES, _validate_tiny_shakespeare


def test_validate_rejects_small_payload():
    with pytest.raises(ValueError, match="too small"):
        _validate_tiny_shakespeare(b"tiny")


def test_validate_rejects_bad_hash():
    payload = b"x" * TINY_SHAKESPEARE_MIN_BYTES
    with pytest.raises(ValueError, match="SHA-256"):
        _validate_tiny_shakespeare(payload)
