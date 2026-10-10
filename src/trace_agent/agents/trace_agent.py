from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from trace_agent.agents.base import TaskAgent
from trace_agent.contracts.results import AgentResult
from trace_agent.contracts.scores import ModuleScoreBatch
from trace_agent.contracts.task import TaskType


class TRACEAgent:
    """Unified facade that routes a score batch to its task agent."""

    def __init__(self, *, task_agents: Iterable[TaskAgent] = ()) -> None:
        self._agents: dict[TaskType, TaskAgent] = {}
        for agent in task_agents:
            self.register(agent)

    def register(self, agent: TaskAgent) -> None:
        task = TaskType.parse(agent.task)
        if task in self._agents:
            raise ValueError(f"an agent is already registered for {task.value}")
        self._agents[task] = agent

    @property
    def supported_tasks(self) -> tuple[TaskType, ...]:
        return tuple(self._agents)

    def assess(
        self,
        batch: ModuleScoreBatch,
        *,
        initial_used_modules: np.ndarray,
    ) -> AgentResult:
        agent = self._agents.get(batch.task)
        if agent is None:
            supported = ", ".join(task.value for task in self.supported_tasks)
            raise ValueError(
                f"no agent registered for {batch.task.value}; " f"supported tasks: {supported or 'none'}"
            )
        return agent.assess(batch, initial_used_modules=initial_used_modules)
