from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trace_agent.contracts.scores import ModuleScoreBatch, RiskProfile
from trace_agent.contracts.task import SplitRole, TaskType
from trace_agent.data.burden import BurdenProfile
from trace_agent.data.split_table import DatasetSplits
from trace_agent.data.training_dataset import TrainingDataset, TrainingTargets, TrainingView
from trace_agent.planning.candidate_ranker import KnowledgeCandidateRanker
from trace_agent.training.planner import PlannerTrainingConfig, PlannerTrainingResult


class AddFirstModule:
    def predict(self, rows):
        # Action columns are [M1, M2, burden, is_stop].
        return np.where(rows[:, -4] == 1, 1.0, np.where(rows[:, -1] == 1, 0.0, -1.0))


class SnapshotIntegrator:
    def integrate(self, batch, observed_modules):
        active = batch.pred_mask & observed_modules[:, :, None]
        valid = np.any(active, axis=1)
        probability = np.where(valid, np.nan_to_num(batch.scores[:, 0, :], nan=0.5), 0.5)
        return RiskProfile(
            participant_ids=batch.participant_ids,
            endpoints=batch.endpoints,
            raw_scores=1.0 - probability,
            calibrated_scores=probability if batch.task is TaskType.PREVALENT else None,
            valid_mask=valid,
            observed_modules=observed_modules,
            task=batch.task,
            horizon_risks=(
                {"5y": probability, "10y": probability} if batch.task is TaskType.LONGITUDINAL else {}
            ),
        )


def _view(task=TaskType.PREVALENT, role=SplitRole.VALIDATION):
    scores = np.full((3, 2, 2), np.nan, dtype=np.float32)
    scores[:2, :, 0] = np.array([[0.8, 0.7], [0.2, 0.3]])
    return TrainingView(
        role=role,
        row_indices=np.array([7, 9, 11]),
        scores=ModuleScoreBatch(
            participant_ids=np.array([101, 102, 103]),
            modules=("M1", "M2"),
            endpoints=("D1", "D2"),
            scores=scores,
            pred_mask=np.isfinite(scores),
            availability=np.array([[True, True], [True, True], [False, False]]),
            task=task,
        ),
        targets=TrainingTargets(
            labels=np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32),
            label_mask=np.ones((3, 2), dtype=bool),
            event_time=np.array([[3, 12], [7, 2], [6, 12]], dtype=np.float32),
        ),
    )


def _burden():
    return BurdenProfile(modules=("M1", "M2"), monetary=np.array([2.0, 5.0]), time=np.zeros(2))


def _evaluate(view, *, mode="fixed_zero"):
    from trace_agent.training.planner import evaluate_validation_trajectories

    burden = _burden()
    return evaluate_validation_trajectories(
        view,
        model=AddFirstModule(),
        integrator=SnapshotIntegrator(),
        burden=burden,
        ranker=KnowledgeCandidateRanker(kg_matrix=np.ones((2, 2)), burden=burden.composite),
        config=PlannerTrainingConfig(candidate_actions=1, stop_value_mode=mode),
    )


@pytest.mark.parametrize("mode", ["fixed_zero", "model"])
def test_rollouts_report_real_cost_actions_and_final_calibrated_risk(mode):
    metrics = _evaluate(_view(), mode=mode)

    assert metrics["selection_source"] == "validation_only"
    assert metrics["rollout_config"] == {
        "initial_modules": [],
        "candidate_actions": 1,
        "stop_value_mode": mode,
        "stop_threshold": 0.0,
        "max_steps": 0,
    }
    rows = metrics["participants"]
    assert [row["row_index"] for row in rows] == [7, 9, 11]
    assert [row["participant_id"] for row in rows] == [101, 102, 103]
    assert [row["final_burden"] for row in rows] == [2.0, 2.0, 0.0]
    assert [row["added_modules"] for row in rows] == [["M1"], ["M1"], []]
    assert [row["steps"] for row in rows] == [2, 2, 1]
    assert [row["stop_reason"] for row in rows] == ["threshold", "threshold", "no_candidates"]
    assert metrics["summary"]["mean_final_burden"] == pytest.approx(4 / 3)
    assert metrics["summary"]["mean_added_modules"] == pytest.approx(2 / 3)
    assert metrics["summary"]["stop_action_fraction"] == pytest.approx(3 / 5)
    assert metrics["summary"]["participant_stop_rate"] == 1.0
    risk = metrics["final_risk"]
    assert risk["endpoint_order"] == ["D1", "D2"]
    assert risk["classification"]["n_valid"] == 2
    assert risk["classification"]["masked_log_loss"] == pytest.approx(-np.log(0.8))
    assert risk["classification"]["endpoints"][1]["n"] == 0
    json.dumps(metrics, allow_nan=False)


def test_longitudinal_rollouts_use_horizon_known_labels_and_final_valid_mask():
    metrics = _evaluate(_view(TaskType.LONGITUDINAL))

    horizons = metrics["final_risk"]["horizons"]
    assert horizons["5y"]["n_valid"] == 2
    assert horizons["10y"]["n_valid"] == 1  # Row 102 is censored at seven years.
    assert horizons["5y"]["masked_log_loss"] == pytest.approx(-np.log(0.8))
    assert horizons["10y"]["masked_log_loss"] == pytest.approx(-np.log(0.8))
    assert horizons["5y"]["endpoints"][1]["n"] == 0


@pytest.mark.parametrize("role", [SplitRole.TRAIN, SplitRole.TEST])
def test_rollout_metrics_reject_any_nonvalidation_view(role):
    with pytest.raises(ValueError, match="validation view"):
        _evaluate(_view(role=role))


@pytest.mark.parametrize("task", [TaskType.PREVALENT, TaskType.LONGITUDINAL])
def test_task_trainers_attach_rollouts_without_test_access_or_refitting(tmp_path, monkeypatch, task):
    from trace_agent.training.longitudinal import planner as longitudinal
    from trace_agent.training.prevalent import planner as prevalent

    module = prevalent if task is TaskType.PREVALENT else longitudinal
    view = _view(task)
    rows = {
        SplitRole.TRAIN: np.array([0]),
        SplitRole.VALIDATION: np.array([1]),
        SplitRole.TEST: np.array([2]),
    }
    dataset = TrainingDataset(
        scores=view.scores,
        targets=view.targets,
        splits=DatasetSplits(
            table_path=Path(tmp_path) / "split.csv",
            row_indices=rows,
            eid_values={
                role: view.scores.participant_ids[indices].astype(str) for role, indices in rows.items()
            },
            file_sha256="a" * 64,
            canonical_sha256="b" * 64,
        ),
    )
    accessed = []
    original_view = TrainingDataset.view

    def guarded_view(self, role):
        resolved = SplitRole.parse(role)
        assert resolved is not SplitRole.TEST
        accessed.append(resolved)
        return original_view(self, resolved)

    monkeypatch.setattr(TrainingDataset, "view", guarded_view)
    monkeypatch.setattr(module, "build_planner_examples", lambda *args, **kwargs: "train")
    monkeypatch.setattr(module, "build_validation_planner_examples", lambda *args, **kwargs: "validation")
    model = AddFirstModule()
    fits = []

    def fit(train_examples, validation_examples, *, config):
        fits.append((train_examples, validation_examples))
        return PlannerTrainingResult(
            model=model, metrics={"mean_squared_error": 0.25}, example_counts={"train": 1, "validation": 1}
        )

    monkeypatch.setattr(module, "fit_action_value_model", fit)
    integrator_name = (
        "PrevalentRiskIntegrator" if task is TaskType.PREVALENT else "LongitudinalRiskIntegrator"
    )
    monkeypatch.setattr(module, integrator_name, lambda **kwargs: SnapshotIntegrator())
    burden = _burden()
    kwargs = {
        "fusion": object(),
        "burden": burden,
        "ranker": KnowledgeCandidateRanker(kg_matrix=np.ones((2, 2)), burden=burden.composite),
        "config": PlannerTrainingConfig(candidate_actions=1),
    }
    if task is TaskType.PREVALENT:
        kwargs["calibration"] = None
        result = prevalent.train_prevalent_planner(dataset, **kwargs)
    else:
        kwargs["reward"] = longitudinal.LongitudinalReward()
        result = longitudinal.train_longitudinal_planner(dataset, **kwargs)

    assert fits == [("train", "validation")]
    assert result.model is model
    assert result.metrics["mean_squared_error"] == 0.25
    assert result.metrics["validation_trajectories"]["participants"][0]["participant_id"] == 102
    assert set(accessed) == {SplitRole.TRAIN, SplitRole.VALIDATION}
