"""Exception hierarchy for AIR-SDK.

Every error raised by the SDK derives from :class:`AirError` so callers can catch
SDK failures without catching unrelated exceptions.
"""

from __future__ import annotations

from enum import Enum


class AirError(Exception):
    """Base class for all AIR-SDK errors."""


class ValidationError(AirError, ValueError):
    """An argument or external value failed type/range validation."""


class SetupError(AirError):
    """The environment is not ready (e.g. Ollama not running). Carries a *how to fix* hint
    so users are never left guessing."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return self.message if not self.hint else f"{self.message}\n  -> How to fix: {self.hint}"


class ModelNotFoundError(SetupError):
    """The requested model is not installed / does not exist."""


class ModelFormatError(AirError):
    """A model file is malformed, truncated, or violates parser safety limits."""


class BackendUnavailableError(AirError):
    """The requested inference backend is not installed or cannot be used."""


class InsufficientResourcesError(AirError):
    """The model cannot realistically fit the available hardware under the safety policy."""


class UnsafeConfigurationError(AirError):
    """A configuration was rejected by the deterministic safety rules."""


class PersistenceError(AirError):
    """The RL experience store could not be read or written."""


class FailureKind(str, Enum):
    """Classification of runtime failures, used for RL penalties and rollback."""

    OUT_OF_MEMORY = "out_of_memory"
    LOAD_FAILED = "load_failed"
    RUNTIME_ERROR = "runtime_error"


class ModelLoadError(AirError):
    """The backend failed to load a configuration."""

    def __init__(self, message: str, kind: FailureKind = FailureKind.LOAD_FAILED) -> None:
        super().__init__(message)
        self.kind: FailureKind = kind


class GenerationError(AirError):
    """The backend failed during generation."""

    def __init__(self, message: str, kind: FailureKind = FailureKind.RUNTIME_ERROR) -> None:
        super().__init__(message)
        self.kind: FailureKind = kind
