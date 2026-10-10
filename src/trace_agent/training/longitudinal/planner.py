from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from trace_agent.contracts.scores import RiskProfile
from trace_agent.contracts.task import TaskType
from trace_agent.data.burden import BurdenProfile
from trace_agent.data.training_dataset import TrainingDataset, TrainingTargets
from trace_agent.integration.longitudinal_cox import LongitudinalRiskIntegrator
from trace_agent.planning.candidate_ranker import KnowledgeCandidateRanker
from trace_agent.planning.rewards.longitudinal import (
    build_horizon_labels,
    multi_horizon_bce_gain,
)
from trace_agent.training.planner import (
    PlannerTrainingConfig,
    PlannerTrainingResult,
    build_planner_examples,
    build_validation_planner_examples,
    evaluate_validation_trajectories,
    fit_action_value_model,
)


def _horizon_name(value: float) -> str:
    return f"{int(value)}y" if float(value).is_integer() else f"{value:g}y"


@dataclass(frozen=True)
class LongitudinalReward:
    horizons: tuple[float, ...] = (5.0, 10.0)
    horizon_weights: tuple[float, ...] = (0.5, 0.5)

    def __post_init__(self) -> None:
        horizons = tuple(float(value) for value in self.horizons)
        weights = tuple(float(value) for value in self.horizon_weights)
        if not horizons or len(horizons) != len(weights):
            raise ValueError("horizons and horizon_weights must have equal non-zero length")
        if np.any(~np.isfinite(horizons)) or np.any(np.asarray(horizons) <= 0):
            raise ValueError("horizons must be finite and positive")
        if len(set(horizons)) != len(horizons):
            raise ValueError("horizons must be unique")
        if np.any(~np.isfinite(weights)) or np.any(np.asarray(weights) < 0):
            raise ValueError("horizon_weights must be finite and non-negative")
        if float(np.sum(weights)) <= 0:
            raise ValueError("at least one horizon weight must be positive")
        object.__setattr__(self, "horizons", horizons)
        object.__setattr__(self, "horizon_weights", weights)

    def _logits(self, profile: RiskProfile) -> np.ndarray:
        probabilities: list[np.ndarray] = []
        for horizon in self.horizons:
            name = _horizon_name(horizon)
            if name not in profile.horizon_risks:
                raise ValueError(f"risk profile does not contain horizon {name!r}")
            values = np.asarray(profile.horizon_risks[name], dtype=np.float64)
            if values.shape != (1, len(profile.endpoints)):
                raise ValueError("planner rewards require one participant at a time")
            probabilities.append(values[0])
        probability = np.stack(probabilities, axis=-1)
        probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
        return np.log(probability / (1.0 - probability))

    def gain(
        self,
        *,
        current: RiskProfile,
        updated: RiskProfile,
        targets: TrainingTargets,
        row: int,
    ) -> float:
        if targets.event_time is None:
            raise ValueError("longitudinal rewards require event_time targets")
        labels, known = build_horizon_labels(
            event=targets.labels[row],
            time=targets.event_time[row],
            label_mask=targets.label_mask[row],
            horizons=np.asarray(self.horizons, dtype=np.float64),
        )
        return multi_horizon_bce_gain(
            current_logits=self._logits(current),
            updated_logits=self._logits(updated),
            labels=labels,
            known_mask=known,
            horizon_weights=np.asarray(self.horizon_weights, dtype=np.float64),
        )


def train_longitudinal_planner(
    dataset: TrainingDataset,
    *,
    fusion: Any,
    burden: BurdenProfile,
    ranker: KnowledgeCandidateRanker,
    reward: LongitudinalReward,
    config: PlannerTrainingConfig,
) -> PlannerTrainingResult:
    if dataset.scores.task is not TaskType.LONGITUDINAL:
        raise ValueError("longitudinal planner training requires longitudinal data")
    integrator = LongitudinalRiskIntegrator(fusion=fusion)
    train_examples = build_planner_examples(
        dataset.view("train"),
        integrator=integrator,
        burden=burden,
        ranker=ranker,
        reward=reward,
        config=config,
    )
    validation = dataset.view("validation")
    validation_examples = build_validation_planner_examples(
        validation,
        integrator=integrator,
        burden=burden,
        ranker=ranker,
        reward=reward,
        config=config,
    )
    result = fit_action_value_model(
        train_examples,
        validation_examples,
        config=config,
    )
    return replace(
        result,
        metrics={
            **result.metrics,
            "validation_trajectories": evaluate_validation_trajectories(
                validation,
                model=result.model,
                integrator=integrator,
                burden=burden,
                ranker=ranker,
                config=config,
            ),
        },
    )
