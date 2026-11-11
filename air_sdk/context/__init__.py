"""Sectioned, budgeted conversation context (rules only; no learning model).

See :mod:`.engine` for the flow, :mod:`.sections` for routing and lifecycle,
:mod:`.budget` for the token budget and safety ladder, :mod:`.facts` for extraction.
"""

from .budget import PromptPlan, build_prompt, estimate_tokens
from .calc import evaluate, expected_answer
from .detect import Detection, detect
from .engine import AUTO, ContextEngine, Prepared, SessionStore, is_auto
from .facts import Schema, extract
from .sections import Section, SessionMemory

__all__ = [
    "AUTO", "ContextEngine", "Detection", "Prepared", "PromptPlan", "Schema", "Section",
    "SessionMemory", "SessionStore", "build_prompt", "detect", "estimate_tokens", "evaluate",
    "expected_answer", "extract", "is_auto",
]
