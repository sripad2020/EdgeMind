"""Research harness for "Verified Execution Profiles for Edge LLMs".

Measures, per task, how weight precision (uniform, per-layer-group, task-aware mixes) and
LoRA adapters change correctness and cost on the local device, with deterministic online
checks, offline gold scoring, Wilson intervals, and replayed selection policies (cascades,
thresholds, tabular Q-learning). Run with ``python -m air_sdk.research``.
"""

from .stats import bootstrap_mean, mcnemar_exact, qualified, wilson
from .store import Profile, ResultStore

__all__ = ["Profile", "ResultStore", "bootstrap_mean", "mcnemar_exact", "qualified", "wilson"]
