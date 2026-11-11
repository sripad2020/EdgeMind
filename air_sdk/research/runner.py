"""Profile a model on test items: same prompts, temperature 0, fixed seed, fixed max tokens.

For every item the answer is checked online (what EdgeMind sees) and scored against gold
(the truth, for evaluation only); both are stored with timings. Runs resume: items already
answered by a profile are skipped.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

from ..config import SafetyPolicy
from ..runtime import AdaptiveModel, AdaptiveRuntime
from ..skills.engine import SkillEngine
from ..skills.profiles import get_skill
from .gold import gold_score, online_check
from .items import Item
from .store import Result, ResultStore

TASK_SKILL: Final[dict[str, str]] = {"sql": "sql", "math": "math", "json": "json",
                                     "code": "python"}
MAX_TOKENS: Final[int] = 384
SEED: Final[int] = 0


@dataclass(frozen=True, slots=True)
class Generation:
    text: str
    seconds: float
    ttft_s: float
    completion_tokens: int
    prompt_tokens: int


class Generator(Protocol):
    def __call__(self, model: str, messages: list[dict[str, object]], *,
                 max_tokens: int) -> Generation: ...


def messages_for(item: Item) -> list[dict[str, object]]:
    """The fixed prompt: the skill's instruction (+ schema / keys), then the question."""
    skill = get_skill(TASK_SKILL[item.task])
    context = item.context if skill.context_label and item.task in ("sql", "json") else None
    return SkillEngine.build_messages(skill, [{"role": "user", "content": item.prompt}], context)


class EdgeMindGenerator:
    """Runs exact Ollama tags through EdgeMind's runtime: memory safety and measurement,
    but no adaptation (RL off, no precision switching) so results are repeatable."""

    def __init__(self, *, data_dir: Path, reserve_ram_gb: float = 0.5,
                 n_ctx: int = 2048, ollama_url: str = "http://127.0.0.1:11434") -> None:
        safety = SafetyPolicy(ram_headroom_min=int(reserve_ram_gb * (1 << 30)),
                              ram_headroom_fraction=0.0)
        self._rt = AdaptiveRuntime("balanced", rl_enabled=False, auto_adapt=False,
                                   backend="ollama", data_dir=data_dir, safety=safety,
                                   ollama_url=ollama_url)
        self._n_ctx = n_ctx
        self._model: AdaptiveModel | None = None
        self._name = ""

    def _load(self, name: str) -> AdaptiveModel:
        if self._model is not None and self._name == name:
            return self._model
        if self._model is not None:
            self._rt.release(self._model)  # one model in memory at a time
            self._model = None
        model = self._rt.load_ollama_model(name, discover=False)
        model.optimize(n_ctx=min(self._n_ctx, model.analyze().context_length),
                       auto_context=False, probe=False)
        self._model, self._name = model, name
        return model

    def __call__(self, model: str, messages: list[dict[str, object]], *,
                 max_tokens: int) -> Generation:
        r = self._load(model).chat(messages, max_tokens=max_tokens, temperature=0.0, seed=SEED)
        return Generation(r.text, r.total_s, r.ttft_s, r.completion_tokens, r.prompt_tokens)

    def close(self) -> None:
        self._rt.close()


ProgressCallback = Callable[[int, int, Result], None]


def profile_items(
    store: ResultStore,
    generate: Generator,
    model: str,
    items: Sequence[Item],
    *,
    data_dir: Path,
    sandbox: bool,
    max_tokens: int = MAX_TOKENS,
    on_result: ProgressCallback | None = None,
) -> int:
    """Answer and score every item not yet done for ``model``; returns how many ran."""
    done = store.done(model)
    todo = [i for i in items if i.id not in done]
    for n, item in enumerate(todo, 1):
        g = generate(model, messages_for(item), max_tokens=max_tokens)
        check = online_check(item, g.text)
        gold = gold_score(item, g.text, db_dir=data_dir, sandbox=sandbox)
        result = Result(model, item.id, item.task, item.split, bool(check.ok), gold.ok,
                        gold.detail, g.seconds, g.ttft_s, g.completion_tokens, g.prompt_tokens,
                        g.text)
        store.add_result(result)
        if on_result is not None:
            on_result(n, len(todo), result)
    return len(todo)


def rescore(store: ResultStore, items: Iterable[Item], *, data_dir: Path,
            sandbox: bool) -> int:
    """Re-run gold scoring on stored answers (e.g. after installing Docker for code)."""
    by_id = {i.id: i for i in items}
    n = 0
    for r in store.results():
        item = by_id.get(r.item_id)
        if item is None:
            continue
        gold = gold_score(item, r.answer, db_dir=data_dir, sandbox=sandbox)
        if gold.ok != r.gold_ok:
            store.add_result(Result(r.profile, r.item_id, r.task, r.split, r.check_ok, gold.ok,
                                    gold.detail, r.seconds, r.ttft_s, r.completion_tokens,
                                    r.prompt_tokens, r.answer))
            n += 1
    return n
