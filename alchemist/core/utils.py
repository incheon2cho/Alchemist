"""Shared utilities."""

from __future__ import annotations

from dataclasses import asdict
from enum import Enum
from typing import Any


def safe_asdict(obj: Any) -> dict[str, Any]:
    """Convert dataclass to dict, converting enum values to strings."""
    return _enum_to_str(asdict(obj))


def _enum_to_str(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _enum_to_str(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_enum_to_str(v) for v in d]
    if isinstance(d, Enum):
        return d.value
    return d
