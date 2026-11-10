"""Inference backends. ``LlamaCppBackend`` is imported lazily (optional dependency)."""

from .base import GenerationResult, InferenceBackend
from .simulated import SimulatedBackend, SimulatedHardware

__all__ = ["GenerationResult", "InferenceBackend", "SimulatedBackend", "SimulatedHardware"]
