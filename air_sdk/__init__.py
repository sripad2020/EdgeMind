"""AIR-SDK: Adaptive Intelligence Runtime.

Quick start::

    import air_sdk
    llm = air_sdk.load("qwen2.5:0.5b")      # any Ollama model, hf.co/... link, or .gguf file
    print(llm.ask("What is QUIC?"))

Hardware-aware LLM execution framework. Four deterministic engines (model analysis,
precision, CPU-GPU placement, adaptive inference) prepare and validate configurations; a
tabular Q-learning engine - the only learning component - chooses among them.
No neural networks, no supervised ML, no external AI services.
"""

from .analysis.profile import ModelProfile
from .analysis.quant import KVCacheType
from .config import DevicePlacement, ExecutionConfig, Objective, SafetyPolicy
from .errors import (
    AirError,
    BackendUnavailableError,
    GenerationError,
    InsufficientResourcesError,
    ModelFormatError,
    ModelLoadError,
    ModelNotFoundError,
    SetupError,
    ValidationError,
)
from .hardware.profiler import GPUInfo, HardwareProfile
from .planner import ExecutionPlan
from .rl.agent import QLearningParams
from .runtime import AdaptiveModel, AdaptiveRuntime, GenerationResponse

__version__ = "0.2.0"

# The simple API (imported last: it builds on everything above).
from .easy import LLM, Conversation, DoctorReport, Reply, doctor, list_models, load

__all__ = [
    "LLM",
    "AdaptiveModel",
    "AdaptiveRuntime",
    "AirError",
    "BackendUnavailableError",
    "Conversation",
    "DevicePlacement",
    "DoctorReport",
    "ExecutionConfig",
    "ExecutionPlan",
    "GPUInfo",
    "GenerationError",
    "GenerationResponse",
    "HardwareProfile",
    "InsufficientResourcesError",
    "KVCacheType",
    "ModelFormatError",
    "ModelLoadError",
    "ModelNotFoundError",
    "ModelProfile",
    "Objective",
    "QLearningParams",
    "Reply",
    "SafetyPolicy",
    "SetupError",
    "ValidationError",
    "__version__",
    "doctor",
    "list_models",
    "load",
]
