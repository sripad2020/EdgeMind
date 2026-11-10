"""Deterministic catalogue of weight and KV-cache precisions.

Quality-loss figures are *priors*: approximate relative perplexity increase versus FP16,
in line with figures published by the llama.cpp project for ~7B models. They are used only
to enforce the hard quality constraint and as the ``Q`` term in the reward. They are
deliberately overridable (``PrecisionPolicy.quality_overrides``) because the real loss
depends on model and task; the SDK never claims these are measured values.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final


@dataclass(frozen=True, slots=True)
class WeightQuant:
    name: str
    bits_per_weight: float
    quality_loss: float


# llama.cpp ``general.file_type`` (LLAMA_FTYPE_*) -> descriptor.
FILE_TYPES: Final[dict[int, WeightQuant]] = {
    0: WeightQuant("F32", 32.0, 0.0),
    1: WeightQuant("F16", 16.0, 0.0),
    32: WeightQuant("BF16", 16.0, 0.0),
    7: WeightQuant("Q8_0", 8.5, 0.001),
    18: WeightQuant("Q6_K", 6.56, 0.004),
    17: WeightQuant("Q5_K_M", 5.69, 0.008),
    16: WeightQuant("Q5_K_S", 5.54, 0.010),
    9: WeightQuant("Q5_1", 6.0, 0.012),
    8: WeightQuant("Q5_0", 5.5, 0.015),
    15: WeightQuant("Q4_K_M", 4.85, 0.020),
    14: WeightQuant("Q4_K_S", 4.58, 0.030),
    3: WeightQuant("Q4_1", 5.0, 0.045),
    2: WeightQuant("Q4_0", 4.5, 0.050),
    13: WeightQuant("Q3_K_L", 4.27, 0.060),
    12: WeightQuant("Q3_K_M", 3.91, 0.080),
    11: WeightQuant("Q3_K_S", 3.5, 0.120),
    10: WeightQuant("Q2_K", 3.35, 0.250),
}

# Fallback when the file type is missing or unknown: assume a conservative, lossy format.
UNKNOWN_QUANT: Final[WeightQuant] = WeightQuant("UNKNOWN", 0.0, 0.10)

# Dominant ggml tensor type -> nearest file-type descriptor (used when general.file_type
# is absent).
_GGML_TYPE_TO_FTYPE: Final[dict[int, int]] = {
    0: 0, 1: 1, 30: 32, 8: 7, 14: 18, 13: 17, 7: 9, 6: 8, 12: 15, 3: 3, 2: 2, 11: 12, 10: 10
}


def quant_from_file_type(file_type: int | None) -> WeightQuant:
    if file_type is None:
        return UNKNOWN_QUANT
    return FILE_TYPES.get(file_type, UNKNOWN_QUANT)


def quant_from_dominant_ggml_type(ggml_type: int) -> WeightQuant:
    ftype = _GGML_TYPE_TO_FTYPE.get(ggml_type)
    return UNKNOWN_QUANT if ftype is None else FILE_TYPES[ftype]


class KVCacheType(str, Enum):
    """KV-cache element types that llama.cpp can select at load time."""

    F16 = "f16"
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"

    @property
    def ggml_type(self) -> int:
        return _KV_GGML[self]

    @property
    def bytes_per_element(self) -> float:
        return _KV_BYTES[self]

    @property
    def quality_loss(self) -> float:
        return _KV_QUALITY[self]

    @property
    def requires_flash_attention(self) -> bool:
        # llama.cpp requires flash attention for a quantized V cache.
        return self is not KVCacheType.F16


_KV_GGML: Final[dict[KVCacheType, int]] = {
    KVCacheType.F16: 1,
    KVCacheType.Q8_0: 8,
    KVCacheType.Q4_0: 2,
}
_KV_BYTES: Final[dict[KVCacheType, float]] = {
    KVCacheType.F16: 2.0,
    KVCacheType.Q8_0: 34 / 32,
    KVCacheType.Q4_0: 18 / 32,
}
_KV_QUALITY: Final[dict[KVCacheType, float]] = {
    KVCacheType.F16: 0.0,
    KVCacheType.Q8_0: 0.002,
    KVCacheType.Q4_0: 0.020,
}

KV_LADDER: Final[tuple[KVCacheType, ...]] = (KVCacheType.F16, KVCacheType.Q8_0, KVCacheType.Q4_0)


def combined_quality_loss(*losses: float) -> float:
    """Combine independent relative losses: 1 - prod(1 - l_i)."""
    keep = 1.0
    for loss in losses:
        keep *= 1.0 - loss
    return 1.0 - keep
