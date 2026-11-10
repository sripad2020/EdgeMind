from __future__ import annotations

import math
from pathlib import Path

import pytest

from air_sdk._validate import (
    require_bool,
    require_enum,
    require_float,
    require_int,
    require_str,
    resolve_file,
)
from air_sdk.analysis.quant import KVCacheType
from air_sdk.config import ExecutionConfig, Objective, SafetyPolicy
from air_sdk.errors import ValidationError


def test_int_rejects_bool_and_float() -> None:
    with pytest.raises(ValidationError):
        require_int("x", True)
    with pytest.raises(ValidationError):
        require_int("x", 3.0)
    assert require_int("x", 3, min_value=0, max_value=3) == 3
    with pytest.raises(ValidationError):
        require_int("x", 4, max_value=3)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, "1.0", None, False])
def test_float_rejects_non_finite_and_wrong_types(bad: object) -> None:
    with pytest.raises(ValidationError):
        require_float("x", bad)


def test_bool_str_enum() -> None:
    with pytest.raises(ValidationError):
        require_bool("b", 1)
    with pytest.raises(ValidationError):
        require_str("s", "a\x00b", max_len=10)
    with pytest.raises(ValidationError):
        require_str("s", "x" * 11, max_len=10)
    assert require_enum("o", "balanced", Objective) is Objective.BALANCED
    with pytest.raises(ValidationError):
        require_enum("o", "fastest", Objective)


def test_resolve_file(tmp_path: Path) -> None:
    good = tmp_path / "m.gguf"
    good.write_bytes(b"\0" * 64)
    assert resolve_file("p", str(good), suffixes=(".gguf",), min_bytes=24, max_bytes=100) == good
    with pytest.raises(ValidationError):
        resolve_file("p", tmp_path / "missing.gguf", suffixes=(".gguf",), min_bytes=1,
                      max_bytes=100)
    other = tmp_path / "m.bin"
    other.write_bytes(b"\0" * 64)
    with pytest.raises(ValidationError):
        resolve_file("p", other, suffixes=(".gguf",), min_bytes=1, max_bytes=100)
    with pytest.raises(ValidationError):
        resolve_file("p", str(good) + "\x00", suffixes=(".gguf",), min_bytes=1, max_bytes=100)
    with pytest.raises(ValidationError):
        resolve_file("p", 123, suffixes=(".gguf",), min_bytes=1, max_bytes=100)


def test_config_validation() -> None:
    # mypy accepts ``True`` for an ``int`` parameter (bool subclasses int); the runtime
    # check is what stops it - exactly why boundary validation exists.
    with pytest.raises(ValidationError):
        ExecutionConfig("m", "F16", KVCacheType.F16, n_gpu_layers=True, n_threads=4,
                        n_batch=512, n_ctx=2048, flash_attn=False)
    with pytest.raises(ValidationError):
        SafetyPolicy(max_quality_loss=1.5)
