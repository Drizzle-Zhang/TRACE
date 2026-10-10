from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from trace_agent.consistency.validator import ConsistencyValidator
from trace_agent.contracts.planning import ActionKind, PlanningResult
from trace_agent.contracts.results import AgentResult, EndpointAssessment
from trace_agent.contracts.scores import ModuleScoreBatch
from trace_agent.contracts.task import TaskType
from trace_agent.planning.engine import PlanningEngine

ContributionProvider = Callable[[ModuleScoreBatch, PlanningResult], np.ndarray]


class TRACETaskAgent:
    """Task-specific orchestration of planning and post-hoc validation."""

    def __init__(
        self,
        *,
        task: TaskType | str,
        planner: PlanningEngine,
        consistency_validator: ConsistencyValidator | None = None,
        contribution_provider: ContributionProvider | None = None,
        decision_strategy: str = "combined",
        artifact_provenance: Mapping[str, object] | None = None,
    ) -> None:
        self.task = TaskType.parse(task)
        self.planner = planner
        self.consistency_validator = consistency_validator
        self.contribution_provider = contribution_provider
        self.decision_strategy = str(decision_strategy)
        self.artifact_provenance = dict(artifact_provenance or {})

    def assess(
        self,
        batch: ModuleScoreBatch,
        *,
        initial_used_modules: np.ndarray,
    ) -> AgentResult:
        if batch.task is not self.task:
            raise ValueError(f"{self.task.value} agent cannot assess a " f"{batch.task.value} score batch")
        planning = self.planner.plan(batch, initial_used_modules=initial_used_modules)
        profile = planning.final_profile
        consistency = None
        retained = np.asarray(profile.valid_mask[0], dtype=bool)
        if self.consistency_validator is not None:
            contributions = (
                None if self.contribution_provider is None else self.contribution_provider(batch, planning)
            )
            consistency = self.consistency_validator.validate(
                batch=batch,
                profile=profile,
                contributions=contributions,
            )
            if self.decision_strategy not in consistency.strategy_masks:
                available = ", ".join(sorted(consistency.strategy_masks))
                raise ValueError(
                    f"unknown consistency strategy " f"{self.decision_strategy!r}; choose from {available}"
                )
            retained = np.asarray(
                consistency.strategy_masks[self.decision_strategy][0],
                dtype=bool,
            )

        prioritized = self._prioritize(profile, retained)
        selected = tuple(
            module for module, is_used in zip(batch.modules, planning.final_used_modules) if is_used
        )
        checks = self._validation_checks(
            batch=batch,
            planning=planning,
            consistency_shape=(None if consistency is None else consistency.combined_scores.shape),
        )
        rationale = self._build_rationale(
            selected_modules=selected,
            planning=planning,
            prioritized=prioritized,
            retained_count=int(np.sum(retained)),
            has_consistency=consistency is not None,
        )
        return AgentResult(
            participant_id=planning.participant_id,
            task=self.task,
            planning=planning,
            selected_modules=selected,
            prioritized_endpoints=prioritized,
            rationale=rationale,
            validation_checks=checks,
            consistency=consistency,
            metadata={
                "decision_strategy": (self.decision_strategy if consistency is not None else "risk_only"),
                "artifact_provenance": self.artifact_provenance,
            },
        )

    @staticmethod
    def _prioritize(profile, retained: np.ndarray) -> tuple[EndpointAssessment, ...]:
        scores = np.asarray(profile.decision_scores[0], dtype=np.float64)
        raw = np.asarray(profile.raw_scores[0], dtype=np.float64)
        valid = np.asarray(profile.valid_mask[0], dtype=bool)
        order = sorted(
            np.flatnonzero(valid).tolist(),
            key=lambda index: (
                not bool(retained[index]),
                -float(scores[index]),
                int(index),
            ),
        )
        assessments: list[EndpointAssessment] = []
        for index in order:
            horizon_values = {
                str(horizon): float(values[0, index])
                for horizon, values in profile.horizon_risks.items()
                if np.isfinite(values[0, index])
            }
            assessments.append(
                EndpointAssessment(
                    endpoint=profile.endpoints[index],
                    endpoint_index=index,
                    score=float(scores[index]),
                    raw_score=float(raw[index]),
                    retained_by_consistency=bool(retained[index]),
                    horizon_risks=horizon_values,
                )
            )
        return tuple(assessments)

    @staticmethod
    def _validation_checks(
        *,
        batch: ModuleScoreBatch,
        planning: PlanningResult,
        consistency_shape: tuple[int, ...] | None,
    ) -> dict[str, bool]:
        profile = planning.final_profile
        valid_values = profile.decision_scores[profile.valid_mask]
        return {
            "participant_alignment": bool(np.array_equal(batch.participant_ids, profile.participant_ids)),
            "endpoint_alignment": bool(tuple(batch.endpoints) == tuple(profile.endpoints)),
            "finite_valid_risks": bool(np.isfinite(valid_values).all()),
            "module_path_monotonic": bool(
                np.all(planning.initial_used_modules <= planning.final_used_modules)
            ),
            "selected_modules_available": bool(np.all(~planning.final_used_modules | batch.availability[0])),
            "terminal_stop_recorded": bool(
                planning.trajectory and planning.trajectory[-1].action.kind is ActionKind.STOP
            ),
            "consistency_alignment": bool(
                consistency_shape is None or consistency_shape == profile.raw_scores.shape
            ),
        }

    def _build_rationale(
        self,
        *,
        selected_modules: tuple[str, ...],
        planning: PlanningResult,
        prioritized: tuple[EndpointAssessment, ...],
        retained_count: int,
        has_consistency: bool,
    ) -> tuple[str, ...]:
        selected_text = ", ".join(selected_modules) if selected_modules else "none"
        statements = [
            f"Task mode: {self.task.value}.",
            (
                f"Selected modules: {selected_text}; accumulated burden: "
                f"{planning.accumulated_burden:.6g}."
            ),
        ]
        if prioritized:
            top = prioritized[0]
            statements.append(f"Highest-priority endpoint: {top.endpoint} " f"(score={top.score:.6g}).")
        if has_consistency:
            statements.append(
                f"Consistency strategy '{self.decision_strategy}' retained " f"{retained_count} endpoint(s)."
            )
        return tuple(statements)
