"""Strict runtime validation helpers.

Static typing (mypy --strict) protects internal code paths, but values arriving from
users, files, C libraries (NVML, llama.cpp) and the database are untyped at runtime.
Every such boundary goes through these helpers.

Notable rules:
* ``bool`` is a subclass of ``int`` in Python; it is rejected wherever an int is expected,
  so ``max_tokens=True`` cannot silently become ``1``.
* Floats must be finite (NaN/inf would poison Q-values and comparisons).
* Strings are length-capped before use.
"""

from __future__ import annotations

import math
import os
from enum import Enum
from pathlib import Path
from typing import Final, TypeVar

from .errors import ValidationError

KiB: Final[int] = 1024
MiB: Final[int] = 1024 * KiB
GiB: Final[int] = 1024 * MiB

_E = TypeVar("_E", bound=Enum)


def require_int(
    name: str,
    value: object,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    """Return ``value`` as an int, rejecting bools, floats and out-of-range values."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} must be an int, got {type(value).__name__}")
    if min_value is not None and value < min_value:
        raise ValidationError(f"{name} must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ValidationError(f"{name} must be <= {max_value}, got {value}")
    return value


def require_float(
    name: str,
    value: object,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """Return ``value`` as a finite float. Ints are widened; bools are rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be finite, got {result}")
    if min_value is not None and result < min_value:
        raise ValidationError(f"{name} must be >= {min_value}, got {result}")
    if max_value is not None and result > max_value:
        raise ValidationError(f"{name} must be <= {max_value}, got {result}")
    return result


def require_bool(name: str, value: object) -> bool:
    """Return ``value`` only if it is a real bool (not 0/1, not a string)."""
    if not isinstance(value, bool):
        raise ValidationError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def require_str(name: str, value: object, *, max_len: int, allow_empty: bool = False) -> str:
    """Return ``value`` as a bounded string without NUL bytes."""
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a str, got {type(value).__name__}")
    if not allow_empty and not value:
        raise ValidationError(f"{name} must not be empty")
    if len(value) > max_len:
        raise ValidationError(f"{name} exceeds maximum length {max_len} (got {len(value)})")
    if "\x00" in value:
        raise ValidationError(f"{name} must not contain NUL bytes")
    return value


def require_enum(name: str, value: object, enum_type: type[_E]) -> _E:
    """Accept an enum member or its exact value; anything else is rejected."""
    if isinstance(value, enum_type):
        return value
    for member in enum_type:
        if type(member.value) is type(value) and member.value == value:
            return member
    allowed = ", ".join(repr(m.value) for m in enum_type)
    raise ValidationError(f"{name} must be one of {allowed}; got {value!r}")


def resolve_file(
    name: str,
    path: object,
    *,
    suffixes: tuple[str, ...],
    min_bytes: int,
    max_bytes: int,
) -> Path:
    """Resolve a user-provided path to an existing regular file with an allowed suffix.

    Symlinks are resolved first so the suffix/size checks apply to the real target.
    """
    if not isinstance(path, (str, os.PathLike)):
        raise ValidationError(f"{name} must be a str or os.PathLike, got {type(path).__name__}")
    raw = os.fspath(path)
    if not isinstance(raw, str):
        raise ValidationError(f"{name} must be a text path, got bytes")
    if "\x00" in raw:
        raise ValidationError(f"{name} must not contain NUL bytes")
    try:
        resolved = Path(raw).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValidationError(f"{name} does not exist or cannot be resolved: {raw!r}") from exc
    if not resolved.is_file():
        raise ValidationError(f"{name} is not a regular file: {resolved}")
    if resolved.suffix.lower() not in suffixes:
        raise ValidationError(f"{name} must have one of the suffixes {suffixes}: {resolved}")
    size = resolved.stat().st_size
    if size < min_bytes or size > max_bytes:
        raise ValidationError(f"{name} size {size} bytes outside [{min_bytes}, {max_bytes}]")
    return resolved


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    return low if value < low else high if value > high else value
