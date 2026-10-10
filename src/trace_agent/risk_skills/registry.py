from __future__ import annotations

from dataclasses import dataclass, field

from trace_agent.contracts.task import TaskType
from trace_agent.risk_skills.base import RiskSkill


@dataclass
class RiskSkillRegistry:
    _skills: dict[tuple[TaskType, str, str], RiskSkill] = field(default_factory=dict)

    def register(
        self,
        *,
        task: TaskType | str,
        module: str,
        endpoint: str,
        skill: RiskSkill,
    ) -> None:
        key = (TaskType.parse(task), str(module), str(endpoint))
        if key in self._skills:
            raise ValueError(f"risk skill is already registered for {key}")
        if not hasattr(skill, "predict"):
            raise TypeError("risk skill must expose predict(values)")
        self._skills[key] = skill

    def get(
        self,
        *,
        task: TaskType | str,
        module: str,
        endpoint: str,
    ) -> RiskSkill | None:
        return self._skills.get((TaskType.parse(task), str(module), str(endpoint)))
