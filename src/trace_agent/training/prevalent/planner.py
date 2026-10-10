from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from trace_agent.contracts.scores import RiskProfile
from trace_agent.contracts.task import TaskType
from trace_agent.data.burden import BurdenProfile
from trace_agent.data.training_dataset import TrainingDataset, TrainingTargets
from trace_agent.integration.prevalent_lgbm import PrevalentRiskIntegrator
from trace_agent.planning.candidate_ranker import KnowledgeCandidateRanker
from trace_agent.planning.rewards.prevalent import prevalent_bce_gain
from trace_agent.training.planner import (
    PlannerTrainingConfig,
    PlannerTrainingResult,
    build_planner_examples,
    build_validation_planner_examples,
    evaluate_validation_trajectories,
    fit_action_value_model,
)
from trace_agent.training.prevalent.calibration import PrevalentCalibrationModel


@dataclass(frozen=True)
class PrevalentReward:
    def gain(
        self,
        *,
        current: RiskProfile,
        updated: RiskProfile,
        targets: TrainingTargets,
        row: int,
    ) -> float:
        return prevalent_bce_gain(
            current_probability=current.decision_scores[0],
            updated_probability=updated.decision_scores[0],
            labels=targets.labels[row],
            label_mask=targets.label_mask[row],
        )


def train_prevalent_planner(
    dataset: TrainingDataset,
    *,
    fusion: Any,
    calibration: PrevalentCalibrationModel | None,
    burden: BurdenProfile,
    ranker: KnowledgeCandidateRanker,
    config: PlannerTrainingConfig,
) -> PlannerTrainingResult:
    if dataset.scores.task is not TaskType.PREVALENT:
        raise ValueError("prevalent planner training requires prevalent data")
    integrator = PrevalentRiskIntegrator(
        fusion=fusion,
        calibrators=({} if calibration is None else dict(calibration.calibrators)),
    )
    reward = PrevalentReward()
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
