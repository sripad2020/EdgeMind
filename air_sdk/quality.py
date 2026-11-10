"""Measured quality check for precision variants (no learning, no external judge model).

Method
  The highest-precision installed version is the reference. A fixed set of prompts is
  answered by the reference and by every other version with greedy decoding
  (temperature 0, fixed seed), so differences come only from the weights' precision.
  Each pair of answers is compared word by word (difflib longest-matching-blocks ratio):
  agreement = mean similarity in [0, 1].

  measured_loss = (1 - agreement) * LOSS_PER_DISAGREEMENT

  With LOSS_PER_DISAGREEMENT = 0.1, a version whose answers differ in half their words
  reaches 0.05 - the SDK's default quality limit - and is excluded. The measured value then
  replaces the catalogue prior for that version in planning and in the RL reward.

Complexity: O(V * P) generations for V versions and P prompts; similarity is
O(n*m) per pair in the worst case but answers are capped at ``max_tokens``.
"""

from __future__ import annotations

import difflib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

LOSS_PER_DISAGREEMENT: Final[float] = 0.1

EVAL_PROMPTS: Final[tuple[str, ...]] = (
    "What is the capital of France? Answer in one sentence.",
    "What is 12 multiplied by 7? Give only the number.",
    "Name the largest planet in our solar system.",
    "Explain in two sentences what a CPU does.",
    "Write a Python function that returns the square of a number.",
    "Translate 'thank you very much' into Spanish.",
    "List three primary colours.",
    "If a train travels 60 km in one hour, how far does it go in 3 hours?",
    "What is the boiling point of water at sea level in Celsius?",
    "Give one sentence describing the water cycle.",
    "What does HTTP stand for?",
    "Complete the sentence: The quick brown fox ...",
)


def word_similarity(a: str, b: str) -> float:
    """Similarity of two texts in [0, 1] on word sequences (1.0 = identical)."""
    wa, wb = a.split(), b.split()
    if not wa and not wb:
        return 1.0
    return difflib.SequenceMatcher(None, wa, wb, autojunk=False).ratio()


@dataclass(frozen=True, slots=True)
class QualityResult:
    model_name: str
    reference_name: str
    agreement: float
    measured_loss: float
    prompts: int
    seconds: float
    worst_prompt: str
    worst_similarity: float

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model_name, "reference": self.reference_name,
            "agreement": round(self.agreement, 4), "measured_loss": round(self.measured_loss, 4),
            "prompts": self.prompts, "seconds": round(self.seconds, 1),
            "worst_prompt": self.worst_prompt, "worst_similarity": round(self.worst_similarity, 3),
        }


Answerer = Callable[[str, str], str]  # (model_name, prompt) -> greedy answer


def evaluate_against_reference(
    answer: Answerer,
    reference: str,
    candidates: Sequence[str],
    prompts: Sequence[str] = EVAL_PROMPTS,
    *,
    release: Callable[[str], None] | None = None,
) -> list[QualityResult]:
    """Answer ``prompts`` with the reference and each candidate; return agreement scores.
    Models are evaluated one at a time (``release`` frees each one) to bound memory."""
    if not prompts:
        raise ValueError("at least one prompt is required")
    ref_answers = [answer(reference, p) for p in prompts]
    if release is not None:
        release(reference)
    results: list[QualityResult] = []
    for name in candidates:
        t0 = time.perf_counter()
        sims = [word_similarity(ref, answer(name, p)) for ref, p in zip(
            ref_answers, prompts, strict=True)]
        if release is not None:
            release(name)
        agreement = sum(sims) / len(sims)
        worst = min(range(len(sims)), key=sims.__getitem__)
        results.append(QualityResult(
            model_name=name, reference_name=reference, agreement=agreement,
            measured_loss=(1.0 - agreement) * LOSS_PER_DISAGREEMENT, prompts=len(prompts),
            seconds=time.perf_counter() - t0, worst_prompt=prompts[worst],
            worst_similarity=sims[worst],
        ))
    return results
