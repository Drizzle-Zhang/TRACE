from __future__ import annotations

from trace_agent.agents.task_agent import TRACETaskAgent
from trace_agent.contracts.task import TaskType


class PrevalentAgent(TRACETaskAgent):
    def __init__(self, **kwargs) -> None:
        super().__init__(task=TaskType.PREVALENT, **kwargs)
