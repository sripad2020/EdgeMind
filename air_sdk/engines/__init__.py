"""Deterministic optimisation engines (precision, placement, candidate space)."""

from .placement import estimate_memory, max_fitting_layers, placement_levels
from .precision import PrecisionCandidates, build_precision_candidates
from .space import CandidateSpace, MemoryBudget, Position, Rejection

__all__ = [
    "CandidateSpace",
    "MemoryBudget",
    "Position",
    "PrecisionCandidates",
    "Rejection",
    "build_precision_candidates",
    "estimate_memory",
    "max_fitting_layers",
    "placement_levels",
]
