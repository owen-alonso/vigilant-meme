"""Shared dataclass serialization helpers."""

from __future__ import annotations

import warnings
from dataclasses import fields
from typing import Any, TypeVar

T = TypeVar("T")


def filter_dataclass_fields(
    cls: type[T],
    data: dict[str, Any],
    *,
    warn_unknown: bool = True,
) -> dict[str, Any]:
    allowed = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - allowed)
    if unknown and warn_unknown:
        warnings.warn(
            f"{cls.__name__}.from_dict ignoring unknown keys: {unknown}",
            stacklevel=3,
        )
    return {k: v for k, v in data.items() if k in allowed}
