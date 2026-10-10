from __future__ import annotations

from typing import Protocol

import numpy as np

from trace_agent.contracts.results import AgentResult
from trace_agent.contracts.scores import ModuleScoreBatch
from trace_agent.contracts.task import TaskType


class TaskAgent(Protocol):
    task: TaskType

    def assess(
        self,
        batch: ModuleScoreBatch,
        *,
        initial_used_modules: np.ndarray,
    ) -> AgentResult: ...
