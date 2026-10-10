from __future__ import annotations

import numpy as np
import pytest

from trace_agent.contracts.scores import ModuleScoreBatch, RiskProfile
from trace_agent.contracts.task import SplitRole, TaskType
from trace_agent.data.burden import BurdenProfile
from trace_agent.data.training_dataset import TrainingTargets, TrainingView
from trace_agent.planning.candidate_ranker import KnowledgeCandidateRanker
from trace_agent.training.planner import (
    PlannerTrainingConfig,
    build_planner_examples,
    build_validation_planner_examples,
    fit_action_value_model,
)


class MeanIntegrator:
    def integrate(self, batch: ModuleScoreBatch, observed_modules: np.ndarray) -> RiskProfile:
        observed = np.asarray(observed_modules, dtype=bool)
        active = batch.pred_mask & observed[:, :, None]
        totals = np.sum(np.where(active, batch.scores, 0.0), axis=1)
        counts = np.sum(active, axis=1)
        probabilities = np.full(totals.shape, 0.5, dtype=np.float32)
        np.divide(totals, counts, out=probabilities, where=counts > 0)
        logits = np.log(np.clip(probabilities, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - probabilities, 1e-6, 1.0))
        return RiskProfile(
            participant_ids=batch.participant_ids,
            endpoints=batch.endpoints,
            raw_scores=probabilities,
            valid_mask=np.any(active, axis=1),
            observed_modules=observed,
            task=TaskType.PREVALENT,
            metadata={"logits": logits},
        )


class ProbabilityGain:
    def gain(self, *, current, updated, targets, row):
        del targets, row
        return float(np.mean(updated.raw_scores - current.raw_scores))


def _view(role: SplitRole) -> TrainingView:
    scores = np.array(
        [
            [[0.2, 0.3], [0.8, 0.7]],
            [[0.7, 0.6], [0.3, 0.4]],
            [[0.4, 0.2], [0.9, 0.8]],
        ],
        dtype=np.float32,
    )
    batch = ModuleScoreBatch(
        participant_ids=np.array([10, 11, 12]),
        modules=("M1", "M2"),
        endpoints=("D1", "D2"),
        scores=scores,
        pred_mask=np.ones_like(scores, dtype=bool),
        availability=np.ones((3, 2), dtype=bool),
        task=TaskType.PREVALENT,
    )
    return TrainingView(
        role=role,
        row_indices=np.array([100, 101, 102]),
        scores=batch,
        targets=TrainingTargets(
            labels=np.array([[0, 0], [1, 1], [0, 1]], dtype=np.float32),
            label_mask=np.ones((3, 2), dtype=bool),
        ),
    )


def _burden() -> BurdenProfile:
    return BurdenProfile(
        modules=("M1", "M2"),
        monetary=np.array([1.0, 2.0], dtype=np.float32),
        time=np.zeros(2, dtype=np.float32),
    )


def _ranker(burden: BurdenProfile) -> KnowledgeCandidateRanker:
    return KnowledgeCandidateRanker(
        kg_matrix=np.ones((2, 2), dtype=np.float32),
        burden=burden.composite,
    )


def test_planner_examples_match_runtime_feature_width() -> None:
    view = _view(SplitRole.TRAIN)
    burden = _burden()

    examples = build_planner_examples(
        view,
        integrator=MeanIntegrator(),
        burden=burden,
        ranker=_ranker(burden),
        reward=ProbabilityGain(),
        config=PlannerTrainingConfig(
            seed=2,
            states_per_participant=1,
            candidate_actions=2,
            stop_value_mode="model",
            n_estimators=8,
            min_child_samples=1,
        ),
    )

    expected_width = len(view.scores.endpoints) + 10 + 2 * len(view.scores.modules) + 4
    assert examples.features.shape[1] == expected_width
    assert examples.features.shape[0] == examples.targets.shape[0]
    assert set(examples.participant_rows.tolist()) == {100, 101, 102}


@pytest.mark.parametrize("role", [SplitRole.VALIDATION, SplitRole.TEST])
def test_training_example_builder_rejects_nontrain_views(role: SplitRole) -> None:
    burden = _burden()

    with pytest.raises(ValueError, match="train view"):
        build_planner_examples(
            _view(role),
            integrator=MeanIntegrator(),
            burden=burden,
            ranker=_ranker(burden),
            reward=ProbabilityGain(),
            config=PlannerTrainingConfig(states_per_participant=1),
        )


def test_action_value_model_fits_train_and_scores_validation_examples() -> None:
    burden = _burden()
    config = PlannerTrainingConfig(
        seed=3,
        states_per_participant=2,
        candidate_actions=2,
        n_estimators=8,
        min_child_samples=1,
        n_jobs=1,
    )
    train_examples = build_planner_examples(
        _view(SplitRole.TRAIN),
        integrator=MeanIntegrator(),
        burden=burden,
        ranker=_ranker(burden),
        reward=ProbabilityGain(),
        config=config,
    )
    validation_examples = build_validation_planner_examples(
        _view(SplitRole.VALIDATION),
        integrator=MeanIntegrator(),
        burden=burden,
        ranker=_ranker(burden),
        reward=ProbabilityGain(),
        config=config,
    )

    result = fit_action_value_model(
        train_examples,
        validation_examples,
        config=config,
    )

    predictions = result.model.predict(validation_examples.features)
    assert predictions.shape == validation_examples.targets.shape
    assert result.example_counts == {
        "train": len(train_examples.targets),
        "validation": len(validation_examples.targets),
    }
    assert set(result.metrics) == {"mean_squared_error", "mean_absolute_error"}
