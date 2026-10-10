from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from trace_agent.contracts.planning import (
    ActionKind,
    PlannerAction,
    PlannerStep,
    PlanningResult,
)
from trace_agent.contracts.scores import ModuleScoreBatch, RiskProfile
from trace_agent.contracts.task import TaskType
from trace_agent.data.burden import BurdenProfile
from trace_agent.planning.actions import build_action_features
from trace_agent.planning.candidate_ranker import KnowledgeCandidateRanker
from trace_agent.planning.q_model import QModel
from trace_agent.planning.state import build_state_features


class RiskIntegrator(Protocol):
    def integrate(self, batch: ModuleScoreBatch, observed_modules: np.ndarray) -> RiskProfile: ...


@dataclass(frozen=True)
class PlannerConfig:
    max_steps: int = 0
    candidate_actions: int = 16
    stop_threshold: float = 0.0
    stop_value_mode: str = "fixed_zero"

    def __post_init__(self) -> None:
        if self.max_steps < 0:
            raise ValueError("max_steps cannot be negative")
        if self.candidate_actions <= 0:
            raise ValueError("candidate_actions must be positive")
        if not np.isfinite(self.stop_threshold):
            raise ValueError("stop_threshold must be finite")
        if self.stop_value_mode not in {"fixed_zero", "model"}:
            raise ValueError("stop_value_mode must be 'fixed_zero' or 'model'")


class PlanningEngine:
    def __init__(
        self,
        *,
        integrator: RiskIntegrator,
        q_model: QModel,
        candidate_ranker: KnowledgeCandidateRanker,
        burden_profile: BurdenProfile,
        config: PlannerConfig | None = None,
    ) -> None:
        self.integrator = integrator
        self.q_model = q_model
        self.candidate_ranker = candidate_ranker
        self.burden_profile = burden_profile
        self.config = config or PlannerConfig()

    def _validate(self, batch: ModuleScoreBatch, initial: np.ndarray) -> None:
        if batch.n_participants != 1:
            raise ValueError("PlanningEngine.plan expects exactly one participant")
        if tuple(batch.modules) != tuple(self.burden_profile.modules):
            raise ValueError("burden module order does not match score module order")
        if initial.shape != (len(batch.modules),):
            raise ValueError(f"initial_used_modules shape must be {(len(batch.modules),)}")
        if np.any(initial & ~batch.availability[0]):
            raise ValueError("initial_used_modules contains unavailable modules")

    def _profile(self, batch: ModuleScoreBatch, used: np.ndarray) -> RiskProfile:
        profile = self.integrator.integrate(batch, used[None, :].copy())
        if profile.n_participants != 1:
            raise ValueError("integrator must preserve the single participant")
        return profile

    def plan(
        self,
        batch: ModuleScoreBatch,
        *,
        initial_used_modules: np.ndarray,
    ) -> PlanningResult:
        initial = np.asarray(initial_used_modules, dtype=bool).reshape(-1)
        self._validate(batch, initial)
        used = initial.copy()
        trajectory: list[PlannerStep] = []
        max_additions = self.config.max_steps if self.config.max_steps > 0 else len(batch.modules)
        for step_index in range(max_additions + 1):
            profile = self._profile(batch, used)
            accumulated = self.burden_profile.accumulated(used)
            candidates = self.candidate_ranker.rank(
                profile=profile,
                pred_mask_md=batch.pred_mask[0],
                availability=batch.availability[0],
                used_modules=used,
                top_k=self.config.candidate_actions,
            )
            state = build_state_features(
                profile,
                used_modules=used,
                accumulated_burden=accumulated,
            )
            add_rows = [
                np.concatenate(
                    [
                        state,
                        build_action_features(
                            int(module_index),
                            n_modules=len(batch.modules),
                            burden=float(self.burden_profile.composite[int(module_index)]),
                        ),
                    ]
                )
                for module_index in candidates
            ]
            if add_rows:
                add_values = np.asarray(
                    self.q_model.predict(np.vstack(add_rows)),
                    dtype=np.float32,
                ).reshape(-1)
                if add_values.shape != (len(add_rows),):
                    raise ValueError("Q-model returned an unexpected prediction shape")
                if not np.isfinite(add_values).all():
                    raise ValueError("Q-model returned non-finite values")
            else:
                add_values = np.empty(0, dtype=np.float32)

            stop_features = build_action_features(None, n_modules=len(batch.modules), burden=0.0)
            if self.config.stop_value_mode == "model":
                stop_value = float(
                    np.asarray(self.q_model.predict(np.concatenate([state, stop_features])[None, :])).reshape(
                        -1
                    )[0]
                )
                if not np.isfinite(stop_value):
                    raise ValueError("Q-model returned a non-finite STOP value")
            else:
                stop_value = 0.0
            q_values = np.concatenate(
                [
                    add_values,
                    np.asarray([stop_value], dtype=np.float32),
                ]
            )

            best_add = int(np.argmax(add_values)) if add_values.size else None
            # The original model-STOP planners break ties by action order:
            # prevalent has STOP first, while longitudinal has ADDs first.
            prefer_add_tie = self.config.stop_value_mode == "model" and profile.task is TaskType.LONGITUDINAL
            stop_wins = best_add is not None and (
                stop_value > float(add_values[best_add])
                if prefer_add_tie
                else stop_value >= float(add_values[best_add])
            )
            at_capacity = step_index >= max_additions
            should_stop = (
                at_capacity
                or best_add is None
                or float(add_values[best_add]) <= float(self.config.stop_threshold)
                or stop_wins
            )
            if should_stop:
                if at_capacity:
                    stop_reason = "max_steps"
                elif best_add is None:
                    stop_reason = "no_candidates"
                elif float(add_values[best_add]) <= float(self.config.stop_threshold):
                    stop_reason = "threshold"
                else:
                    stop_reason = "stop_value"
                trajectory.append(
                    PlannerStep(
                        step_index=step_index,
                        action=PlannerAction(ActionKind.STOP),
                        candidate_module_indices=tuple(int(x) for x in candidates),
                        q_values=q_values,
                        accumulated_burden=accumulated,
                        risk_profile=profile,
                        metadata={"stop_reason": stop_reason},
                    )
                )
                break

            module_index = int(candidates[best_add])
            used[module_index] = True
            trajectory.append(
                PlannerStep(
                    step_index=step_index,
                    action=PlannerAction(
                        ActionKind.ADD,
                        module_index=module_index,
                        module_name=batch.modules[module_index],
                    ),
                    candidate_module_indices=tuple(int(x) for x in candidates),
                    q_values=q_values,
                    accumulated_burden=self.burden_profile.accumulated(used),
                    risk_profile=profile,
                )
            )

        final_profile = self._profile(batch, used)
        return PlanningResult(
            participant_id=batch.participant_ids[0],
            initial_used_modules=initial,
            final_used_modules=used,
            accumulated_burden=self.burden_profile.accumulated(used),
            final_profile=final_profile,
            trajectory=tuple(trajectory),
            metadata={"max_additions": max_additions},
        )
