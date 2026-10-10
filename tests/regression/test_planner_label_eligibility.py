"""Q supervision skips unknown labels without discarding known censoring."""

from dataclasses import replace

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
)


class TrackingIntegrator:
    def __init__(self):
        self.participants = []

    def integrate(self, batch, observed_modules):
        self.participants.extend(batch.participant_ids.tolist())
        active = batch.pred_mask & observed_modules[:, :, None]
        return RiskProfile(
            participant_ids=batch.participant_ids,
            endpoints=batch.endpoints,
            raw_scores=np.full((1, 2), 0.5, dtype=np.float32),
            valid_mask=np.any(active, axis=1),
            observed_modules=observed_modules,
            task=batch.task,
        )


class LabelGuardReward:
    def __init__(self):
        self.rows = []

    def gain(self, *, current, updated, targets, row):
        assert targets.label_mask[row].any(), "reward must never see an entirely unknown target"
        assert updated.observed_modules[0].tolist() == [True, False]
        assert current.observed_modules[0].tolist() == [False, False]
        self.rows.append(row)
        return 0.25


def _mixed_view(task, role):
    availability = np.array([[True, False], [True, False], [True, False], [False, False]])
    pred_mask = np.zeros((4, 2, 2), dtype=bool)
    pred_mask[:3, 0, :] = True
    pred_mask[1, 0, 1] = False
    scores = np.where(pred_mask, 0.3, np.nan)
    return TrainingView(
        role=role,
        row_indices=np.array([20, 31, 47, 55]),
        scores=ModuleScoreBatch(
            participant_ids=np.array([100, 101, 102, 103]),
            modules=("M1", "M2"),
            endpoints=("D1", "D2"),
            scores=scores,
            pred_mask=pred_mask,
            availability=availability,
            task=task,
        ),
        targets=TrainingTargets(
            labels=np.array([[np.nan, np.nan], [0, np.nan], [0, 0], [np.nan, np.nan]]),
            label_mask=np.array([[False, False], [True, False], [True, True], [False, False]]),
            # The labeled longitudinal rows are all censored. Row 1 is known
            # event-free through 12 years; row 2 has positive follow-up too.
            event_time=(
                np.array([[np.nan, np.nan], [12, np.nan], [2, 12], [np.nan, np.nan]])
                if task is TaskType.LONGITUDINAL
                else None
            ),
        ),
    )


def _build(view, mode, *, reward, integrator):
    burden = BurdenProfile(modules=("M1", "M2"), monetary=np.array([1.0, 4.0]), time=np.zeros(2))
    builder = build_planner_examples if view.role is SplitRole.TRAIN else build_validation_planner_examples
    return builder(
        view,
        integrator=integrator,
        burden=burden,
        ranker=KnowledgeCandidateRanker(kg_matrix=np.ones((2, 2)), burden=burden.composite),
        reward=reward,
        config=PlannerTrainingConfig(states_per_participant=2, stop_value_mode=mode),
    )


@pytest.mark.parametrize("task", [TaskType.PREVALENT, TaskType.LONGITUDINAL])
@pytest.mark.parametrize("role", [SplitRole.TRAIN, SplitRole.VALIDATION])
@pytest.mark.parametrize("mode", ["fixed_zero", "model"])
def test_unknown_target_rows_never_enter_add_or_stop_supervision(task, role, mode):
    reward = LabelGuardReward()
    integrator = TrackingIntegrator()
    examples = _build(_mixed_view(task, role), mode, reward=reward, integrator=integrator)

    assert set(examples.participant_rows.tolist()) == {31, 47}
    assert reward.rows == [1, 1, 2, 2]
    assert set(integrator.participants) == {101, 102}
    stop = examples.features[:, -1] == 1
    assert examples.features.shape[0] == (8 if mode == "model" else 4)
    assert int(stop.sum()) == (4 if mode == "model" else 0)
    np.testing.assert_allclose(examples.targets[~stop], 0.2475)
    np.testing.assert_allclose(examples.targets[stop], 0)
    np.testing.assert_array_equal(examples.features[~stop, -4:-2], np.tile([1, 0], (4, 1)))


@pytest.mark.parametrize("task", [TaskType.PREVALENT, TaskType.LONGITUDINAL])
@pytest.mark.parametrize("mode", ["fixed_zero", "model"])
def test_entirely_unknown_training_cohort_rejects_without_reward_calls(task, mode):
    view = _mixed_view(task, SplitRole.TRAIN)
    view = replace(view, targets=replace(view.targets, label_mask=np.zeros((4, 2), dtype=bool)))
    reward = LabelGuardReward()
    integrator = TrackingIntegrator()
    with pytest.raises(ValueError, match="produced no candidate actions"):
        _build(view, mode, reward=reward, integrator=integrator)
    assert reward.rows == []
    assert integrator.participants == []
