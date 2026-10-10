from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from trace_agent.contracts.consistency import ConsistencyResult
from trace_agent.contracts.planning import PlanningResult
from trace_agent.contracts.task import TaskType


@dataclass(frozen=True)
class EndpointAssessment:
    endpoint: str
    endpoint_index: int
    score: float
    raw_score: float
    retained_by_consistency: bool
    horizon_risks: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResult:
    participant_id: object
    task: TaskType
    planning: PlanningResult
    selected_modules: tuple[str, ...]
    prioritized_endpoints: tuple[EndpointAssessment, ...]
    rationale: tuple[str, ...]
    validation_checks: Mapping[str, bool]
    consistency: ConsistencyResult | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        task = TaskType.parse(self.task)
        checks = {str(name): bool(value) for name, value in self.validation_checks.items()}
        if not checks:
            raise ValueError("AgentResult requires validation checks")
        if self.planning.final_profile.task is not task:
            raise ValueError("AgentResult task does not match its final risk profile")
        if not np.array_equal(
            self.planning.final_profile.participant_ids,
            np.asarray([self.participant_id]),
        ):
            raise ValueError("AgentResult participant does not match its risk profile")
        object.__setattr__(self, "task", task)
        object.__setattr__(
            self,
            "selected_modules",
            tuple(str(value) for value in self.selected_modules),
        )
        object.__setattr__(
            self,
            "prioritized_endpoints",
            tuple(self.prioritized_endpoints),
        )
        object.__setattr__(self, "rationale", tuple(str(value) for value in self.rationale))
        object.__setattr__(self, "validation_checks", checks)
