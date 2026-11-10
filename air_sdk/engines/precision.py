"""Engine 2: Adaptive Precision (deterministic).

llama.cpp fixes *weight* precision per GGUF file, so weight-precision candidates are the
validated variants of the same model (same architecture fingerprint) available on disk.
KV-cache precision is selectable at load time. The RL engine only ever chooses among the
candidates this engine emits; it never invents formats or alters numerics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from ..analysis.profile import ModelProfile
from ..analysis.quant import KV_LADDER, KVCacheType, combined_quality_loss
from ..errors import InsufficientResourcesError, ValidationError


@dataclass(frozen=True, slots=True)
class PrecisionCandidates:
    # Ordered from highest precision (largest) to lowest (smallest).
    variants: tuple[ModelProfile, ...]
    kv_types: tuple[KVCacheType, ...]
    max_quality_loss: float

    def quality_loss(self, variant: int, kv: int) -> float:
        return combined_quality_loss(
            self.variants[variant].quant.quality_loss, self.kv_types[kv].quality_loss
        )

    def is_quality_ok(self, variant: int, kv: int) -> bool:
        return within_quality(self.quality_loss(variant, kv), self.max_quality_loss)


# Absolute tolerance for comparing quality losses: combining losses in floating point
# (1 - (1-a)(1-b)) can exceed an exact limit by ~1e-17 (e.g. 0.05 -> 0.050000000000000044).
_QUALITY_TOL = 1e-9


def within_quality(loss: float, limit: float) -> bool:
    """Single comparison used by every quality check, so all checks agree."""
    return loss <= limit + _QUALITY_TOL


def build_precision_candidates(
    variants: tuple[ModelProfile, ...],
    *,
    max_quality_loss: float,
    pinned_precision: str | None = None,
    allowed_kv: tuple[KVCacheType, ...] = KV_LADDER,
    quality_overrides: Mapping[str, float] | None = None,
) -> PrecisionCandidates:
    """Filter variants by the hard quality constraint and order them by size.

    ``quality_overrides`` maps variant_key -> *measured* quality loss; a measurement always
    replaces the catalogue prior for that variant. O(V log V) for V variants.
    """
    if not variants:
        raise ValidationError("at least one model variant is required")
    family = variants[0].family_key
    if any(v.family_key != family for v in variants):
        raise ValidationError("all variants must be the same model (family fingerprint)")

    if quality_overrides:
        variants = tuple(
            replace(v, quant=replace(v.quant, quality_loss=quality_overrides[v.variant_key]))
            if v.variant_key in quality_overrides else v
            for v in variants
        )
    pool = variants
    if pinned_precision is not None:
        wanted = pinned_precision.upper()
        pool = tuple(v for v in variants if v.quant.name == wanted)
        if not pool:
            available = sorted({v.quant.name for v in variants})
            raise ValidationError(f"precision {pinned_precision!r} not available; have {available}")

    accepted = tuple(
        sorted(
            (v for v in pool if within_quality(v.quant.quality_loss, max_quality_loss)),
            key=lambda v: (-v.weight_bytes, v.quant.quality_loss, str(v.path)),
        )
    )
    if not accepted:
        raise InsufficientResourcesError(
            "no model variant satisfies max_quality_loss="
            f"{max_quality_loss}; variants: {[v.quant.name for v in pool]}"
        )
    best_weight_loss = min(v.quant.quality_loss for v in accepted)
    kv_types = tuple(
        kv for kv in KV_LADDER
        if kv in allowed_kv
        and within_quality(combined_quality_loss(best_weight_loss, kv.quality_loss),
                           max_quality_loss)
    )
    if not kv_types:
        raise InsufficientResourcesError(
            f"no KV-cache type supported by the backend ({[k.value for k in allowed_kv]})"
            f" satisfies max_quality_loss={max_quality_loss}"
        )
    return PrecisionCandidates(accepted, kv_types, max_quality_loss)
