from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from trace_agent.contracts.scores import RiskProfile


class ActionKind(str, Enum):
    ADD = "add"
    STOP = "stop"


@dataclass(frozen=True)
class PlannerAction:
    kind: ActionKind
    module_index: int | None = None
    module_name: str | None = None

    def __post_init__(self) -> None:
        kind = ActionKind(self.kind)
        if kind is ActionKind.ADD:
            if self.module_index is None or int(self.module_index) < 0:
                raise ValueError("an add action requires a non-negative module_index")
            if not self.module_name:
                raise ValueError("an add action requires module_name")
        elif self.module_index is not None or self.module_name is not None:
            raise ValueError("a stop action cannot reference a module")
        object.__setattr__(self, "kind", kind)


@dataclass(frozen=True)
class PlannerStep:
    step_index: int
    action: PlannerAction
    candidate_module_indices: tuple[int, ...]
    q_values: np.ndarray
    accumulated_burden: float
    risk_profile: RiskProfile
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidates = tuple(int(value) for value in self.candidate_module_indices)
        q_values = np.asarray(self.q_values, dtype=np.float32).reshape(-1)
        if q_values.size != len(candidates) + 1:
            raise ValueError("q_values must contain one value per candidate plus STOP")
        if not np.isfinite(q_values).all():
            raise ValueError("q_values must be finite")
        object.__setattr__(self, "candidate_module_indices", candidates)
        object.__setattr__(self, "q_values", q_values)


@dataclass(frozen=True)
class PlanningResult:
    participant_id: object
    initial_used_modules: np.ndarray
    final_used_modules: np.ndarray
    accumulated_burden: float
    final_profile: RiskProfile
    trajectory: tuple[PlannerStep, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        initial = np.asarray(self.initial_used_modules, dtype=bool).reshape(-1)
        final = np.asarray(self.final_used_modules, dtype=bool).reshape(-1)
        if initial.shape != final.shape:
            raise ValueError("initial and final module masks must have the same shape")
        if np.any(initial & ~final):
            raise ValueError("planning cannot remove an initially observed module")
        object.__setattr__(self, "initial_used_modules", initial)
        object.__setattr__(self, "final_used_modules", final)
        object.__setattr__(self, "trajectory", tuple(self.trajectory))
