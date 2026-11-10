"""Discrete RL action space.

Actions are *relative moves* inside the validated candidate lattice built by the
deterministic engines. An action is only ever offered to the agent if the resulting
configuration exists and passes the safety rules (action masking), so exploration is safe
by construction.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final


class Action(IntEnum):
    MAINTAIN = 0
    LOWER_PRECISION = 1  # e.g. Q8_0 -> Q4_K_M (smaller, faster, slightly lossier)
    HIGHER_PRECISION = 2
    MORE_GPU_LAYERS = 3
    FEWER_GPU_LAYERS = 4
    MORE_THREADS = 5
    FEWER_THREADS = 6
    LARGER_BATCH = 7
    SMALLER_BATCH = 8
    LOWER_KV_PRECISION = 9  # lower-memory KV cache
    HIGHER_KV_PRECISION = 10


N_ACTIONS: Final[int] = len(Action)
ALL_ACTIONS: Final[tuple[Action, ...]] = tuple(Action)
