"""Verified skills: answer on the fastest precision, check, escalate only on failure.

For a task with a deterministic check (SQL, Python, JSON, Math):

1. **Route** - start on the fastest precision whose recorded pass rate for this model and
   skill is acceptable (untested precisions are tried; ones that keep failing are skipped).
2. **Answer** on that precision.
3. **Verify** the answer with the skill's check (see ``verify.py``).
4. **Escalate** on failure: retry on the next more accurate precision (e.g. Q4_0 -> Q8_0 ->
   F16), at most ``MAX_ATTEMPTS`` times in total.
5. **Remember** every pass/fail per (model family, skill, precision), so routing improves.

Result: the speed of the small precision when it is good enough, and the correctness of
the large one when it is not - decided per task, from evidence on this model.

Routing uses counts, not a learned model; precision pinning bypasses the RL policy for that
request (it is not learned from) but every safety check still applies.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from .._validate import require_str
from ..backends.base import TokenCallback
from ..rl.store import ExperienceStore
from ..runtime import AdaptiveModel, GenerationResponse, ResetCallback
from .profiles import SKILLS, Skill, get_skill
from .verify import MAX_CONTEXT, MAX_TEXT, bonus_check


class _Same:
    """Marker: check against the same context that is in the prompt."""


SAME: Final = _Same()

PASS_FLOOR: Final[float] = 0.8  # a precision is trusted for a skill at >= 80% passes
MIN_SAMPLES: Final[int] = 3  # ... once it has at least this many results
MAX_ATTEMPTS: Final[int] = 3

EscalateCallback = Callable[[str, str, str], None]  # (from_precision, to_precision, reason)


@dataclass(frozen=True, slots=True)
class Attempt:
    precision: str
    ok: bool | None  # None = this skill has no automatic check
    detail: str


@dataclass(frozen=True, slots=True)
class SkillOutcome:
    response: GenerationResponse
    skill: str
    verified: bool | None
    escalated: bool
    attempts: tuple[Attempt, ...]

    @property
    def detail(self) -> str:
        return self.attempts[-1].detail if self.attempts else ""


def start_precision(ladder: Sequence[str], stats: dict[str, tuple[int, int]]) -> str:
    """Fastest precision that is untested or passes >= PASS_FLOOR (ladder: highest first)."""
    for precision in reversed(ladder):  # fastest (smallest) first
        passes, fails = stats.get(precision, (0, 0))
        n = passes + fails
        if n < MIN_SAMPLES or passes / n >= PASS_FLOOR:
            return precision
    return ladder[0]


class SkillEngine:
    """Runs requests for a skill on an :class:`AdaptiveModel` with verify-and-escalate."""

    def __init__(self, model: AdaptiveModel, store: ExperienceStore) -> None:
        self._model = model
        self._store = store

    @property
    def family(self) -> str:
        return self._model.analyze().family_key

    # ---- prompts --------------------------------------------------------------------------

    @staticmethod
    def build_messages(skill: Skill, messages: Sequence[dict[str, object]],
                       context: str | None) -> list[dict[str, object]]:
        """Add the skill's fixed instruction (and its context, e.g. a schema) to the system
        message. A stable prefix also lets the backend reuse its prompt cache."""
        msgs = [dict(m) for m in messages]
        parts = [skill.instruction] if skill.instruction else []
        if context and skill.context_label:
            parts.append(f"{skill.context_label[:1].upper()}{skill.context_label[1:]}:\n{context}")
        if not parts:
            return msgs
        prefix = "\n\n".join(parts)
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = {"role": "system", "content": f"{prefix}\n\n{msgs[0].get('content', '')}"}
        else:
            msgs.insert(0, {"role": "system", "content": prefix})
        return msgs

    # ---- running --------------------------------------------------------------------------

    def run(
        self,
        messages: Sequence[dict[str, object]],
        skill_name: str | None,
        *,
        context: str | None = None,
        max_tokens: int = 512,
        temperature: float | None = None,
        seed: int | None = None,
        on_token: TokenCallback | None = None,
        on_reset: ResetCallback | None = None,
        on_escalate: EscalateCallback | None = None,
        check_context: str | _Same | None = SAME,
        bonus: bool = False,
        bonus_schema: str | None = None,
    ) -> SkillOutcome:
        """Answer, check, escalate. ``context`` goes into the prompt; ``check_context``
        (default: the same) is what the answer is checked against - the context engine
        sends a trimmed prompt but checks against the full stored facts. For skills without
        a check and ``bonus=True``, code blocks in the answer still get a *bonus* check (SQL
        against ``bonus_schema``, Python, JSON) - reported, never escalated or recorded."""
        skill = get_skill(skill_name)
        if context is not None:
            context = require_str("context", context, max_len=MAX_CONTEXT, allow_empty=True)
        verify_with = context if isinstance(check_context, _Same) else check_context
        if verify_with is not None:
            verify_with = require_str("check_context", verify_with, max_len=MAX_TEXT,
                                      allow_empty=True)
        msgs = self.build_messages(skill, messages, context)
        temp = skill.temperature if temperature is None else temperature

        if skill.verifier is None:
            r = self._model.chat(msgs, max_tokens=max_tokens, temperature=temp, seed=seed,
                                 on_token=on_token, on_reset=on_reset)
            extra = bonus_check(r.text, bonus_schema) if bonus else None
            if extra is None:
                return SkillOutcome(r, skill.name, None, False,
                                    (Attempt(r.config.precision, None, "no automatic check"),))
            return SkillOutcome(r, skill.name, extra.ok, False, (
                Attempt(r.config.precision, extra.ok, f"bonus check: {extra.detail}"),))

        ladder = self._model.precision_ladder()
        family = self.family
        index = ladder.index(start_precision(ladder, self._store.skill_results(family,
                                                                              skill.name)))
        attempts: list[Attempt] = []
        response: GenerationResponse | None = None
        while len(attempts) < MAX_ATTEMPTS:
            if attempts and on_escalate is not None:
                on_escalate(attempts[-1].precision, ladder[index], attempts[-1].detail)
            response = self._model.chat(msgs, max_tokens=max_tokens, temperature=temp,
                                        seed=seed, on_token=on_token, on_reset=on_reset,
                                        precision=ladder[index])
            used = response.config.precision  # may differ if the pin was not safe right now
            check = skill.verifier(response.text, verify_with)
            attempts.append(Attempt(used, check.ok, check.detail))
            used_index = ladder.index(used) if used in ladder else index
            if not check.retryable or check.ok or used_index == 0:
                break
            index = used_index - 1  # one step more accurate
        if response is None:  # pragma: no cover - MAX_ATTEMPTS >= 1
            raise RuntimeError("no attempt was made")
        self._record(family, skill.name, attempts, retryable=check.retryable)
        return SkillOutcome(response, skill.name, attempts[-1].ok, len(attempts) > 1,
                            tuple(attempts))

    def _record(self, family: str, skill: str, attempts: list[Attempt], *,
                retryable: bool) -> None:
        """Blame a precision only when the evidence supports it.

        * Final attempt passed: it gets a pass, and each earlier (less accurate) attempt
          gets a fail - a more accurate precision succeeded where it did not.
        * Every attempt failed: only the most accurate attempt gets a fail; the earlier
          failures are inconclusive (the task may simply be beyond this model).
        * Check not usable (e.g. invalid schema): nothing is recorded.
        """
        if not retryable or not attempts:
            return
        final = attempts[-1]
        self._store.record_skill_result(family, skill, final.precision, bool(final.ok))
        if final.ok:
            for earlier in attempts[:-1]:
                self._store.record_skill_result(family, skill, earlier.precision, False)

    def record_correction(self, skill_name: str, precision: str) -> bool:
        """The user said a *checked and passed* answer was wrong: count it as a fail for
        that precision, so routing starts higher next time. Returns True if recorded."""
        skill = get_skill(skill_name)
        if not skill.verified or precision not in self._model.precision_ladder():
            return False
        self._store.record_skill_result(self.family, skill.name, precision, False)
        return True

    # ---- insight --------------------------------------------------------------------------

    def report(self, skill_name: str | None) -> dict[str, object]:
        skill = get_skill(skill_name)
        ladder = self._model.precision_ladder()
        stats = self._store.skill_results(self.family, skill.name)
        rows = []
        for precision in ladder:
            passes, fails = stats.get(precision, (0, 0))
            n = passes + fails
            rows.append({"precision": precision, "passes": passes, "fails": fails,
                         "pass_rate": round(passes / n, 3) if n else None})
        return {
            "skill": skill.name, "title": skill.title, "description": skill.description,
            "verified": skill.verified, "context": skill.context_label or None,
            "starts_on": start_precision(ladder, stats) if skill.verified else None,
            "precisions": rows,
        }

    def catalogue(self) -> list[dict[str, object]]:
        return [self.report(name) for name in SKILLS]
