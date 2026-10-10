from __future__ import annotations

from pathlib import Path

import numpy as np

from trace_agent.contracts.scores import ModuleScoreBatch
from trace_agent.contracts.task import SplitRole, TaskType
from trace_agent.data.burden import BurdenProfile
from trace_agent.data.split_table import DatasetSplits
from trace_agent.data.training_dataset import TrainingDataset, TrainingTargets
from trace_agent.planning.candidate_ranker import KnowledgeCandidateRanker
from trace_agent.training.longitudinal.planner import (
    LongitudinalReward,
    train_longitudinal_planner,
)
from trace_agent.training.planner import PlannerTrainingConfig
from trace_agent.training.prevalent.planner import train_prevalent_planner


class MeanPrevalentFusion:
    def fuse(self, scores, used, pred_mask_md=None):
        active = np.asarray(pred_mask_md, dtype=bool) & np.asarray(used, dtype=bool)[:, None]
        totals = np.sum(np.where(active, scores, 0.0), axis=0)
        counts = np.sum(active, axis=0)
        probability = np.full(totals.shape, 0.5, dtype=np.float32)
        np.divide(totals, counts, out=probability, where=counts > 0)
        return probability

    def fuse_logits(self, scores, used, pred_mask_md=None):
        probability = np.clip(self.fuse(scores, used, pred_mask_md), 1e-6, 1.0 - 1e-6)
        return np.log(probability / (1.0 - probability))


class MeanLongitudinalFusion:
    def __init__(self) -> None:
        self.baseline_survival = {
            "5y": np.array([0.9], dtype=np.float32),
            "10y": np.array([0.8], dtype=np.float32),
        }

    def predict(self, scores, *, pred_mask_nmd, used_mask_nm, standardized=False):
        active = (
            np.asarray(pred_mask_nmd, dtype=bool)
            & np.asarray(
                used_mask_nm,
                dtype=bool,
            )[:, :, None]
        )
        totals = np.sum(np.where(active, scores, 0.0), axis=1)
        counts = np.sum(active, axis=1)
        risk = np.zeros(totals.shape, dtype=np.float32)
        np.divide(totals, counts, out=risk, where=counts > 0)
        return self.standardize(risk) if standardized else risk

    def standardize(self, risk):
        return np.asarray(risk, dtype=np.float32)


def _dataset(tmp_path: Path) -> TrainingDataset:
    labels = np.array(
        [[0], [1], [0], [1], [0], [1], [0], [1], [0]],
        dtype=np.float32,
    )
    scores = np.empty((9, 2, 1), dtype=np.float32)
    scores[:, 0, 0] = 0.2 + 0.6 * labels[:, 0]
    scores[:, 1, 0] = 0.3 + 0.4 * labels[:, 0]
    batch = ModuleScoreBatch(
        participant_ids=np.arange(100, 109),
        modules=("M1", "M2"),
        endpoints=("D1",),
        scores=scores,
        pred_mask=np.ones_like(scores, dtype=bool),
        availability=np.ones((9, 2), dtype=bool),
        task=TaskType.PREVALENT,
    )
    split_path = tmp_path / "splits.csv"
    split_path.write_text("row_index,eid,split\n", encoding="utf-8")
    row_indices = {
        SplitRole.TRAIN: np.arange(0, 5),
        SplitRole.VALIDATION: np.arange(5, 7),
        SplitRole.TEST: np.arange(7, 9),
    }
    eid_values = {role: batch.participant_ids[rows].astype(str) for role, rows in row_indices.items()}
    splits = DatasetSplits(
        table_path=split_path,
        row_indices=row_indices,
        eid_values=eid_values,
        file_sha256="a" * 64,
        canonical_sha256="b" * 64,
    )
    return TrainingDataset(
        scores=batch,
        targets=TrainingTargets(
            labels=labels,
            label_mask=np.ones_like(labels, dtype=bool),
        ),
        splits=splits,
    )


def _burden() -> BurdenProfile:
    return BurdenProfile(
        modules=("M1", "M2"),
        monetary=np.array([1.0, 2.0]),
        time=np.zeros(2),
    )


def _longitudinal_dataset(tmp_path: Path) -> TrainingDataset:
    event = np.array(
        [[0], [1], [0], [1], [0], [1], [0], [1], [0]],
        dtype=np.float32,
    )
    time = np.array(
        [[12], [3], [11], [7], [15], [4], [13], [9], [14]],
        dtype=np.float32,
    )
    scores = np.empty((9, 2, 1), dtype=np.float32)
    scores[:, 0, 0] = 0.2 + 0.6 * event[:, 0]
    scores[:, 1, 0] = 0.3 + 0.4 * event[:, 0]
    batch = ModuleScoreBatch(
        participant_ids=np.arange(200, 209),
        modules=("M1", "M2"),
        endpoints=("D1",),
        scores=scores,
        pred_mask=np.ones_like(scores, dtype=bool),
        availability=np.ones((9, 2), dtype=bool),
        task=TaskType.LONGITUDINAL,
    )
    split_path = tmp_path / "longitudinal_splits.csv"
    split_path.write_text("row_index,eid,split\n", encoding="utf-8")
    row_indices = {
        SplitRole.TRAIN: np.arange(0, 5),
        SplitRole.VALIDATION: np.arange(5, 7),
        SplitRole.TEST: np.arange(7, 9),
    }
    eid_values = {role: batch.participant_ids[rows].astype(str) for role, rows in row_indices.items()}
    return TrainingDataset(
        scores=batch,
        targets=TrainingTargets(
            labels=event,
            label_mask=np.ones_like(event, dtype=bool),
            event_time=time,
            signed_time=np.where(event > 0.5, time, -time),
        ),
        splits=DatasetSplits(
            table_path=split_path,
            row_indices=row_indices,
            eid_values=eid_values,
            file_sha256="c" * 64,
            canonical_sha256="d" * 64,
        ),
    )


def test_prevalent_planner_service_uses_train_and_validation_but_not_test(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    burden = _burden()
    test_labels_before = dataset.view("test").targets.labels.copy()

    result = train_prevalent_planner(
        dataset,
        fusion=MeanPrevalentFusion(),
        calibration=None,
        burden=burden,
        ranker=KnowledgeCandidateRanker(
            kg_matrix=np.ones((2, 1), dtype=np.float32),
            burden=burden.composite,
        ),
        config=PlannerTrainingConfig(
            seed=5,
            states_per_participant=1,
            candidate_actions=2,
            n_estimators=8,
            min_child_samples=1,
            n_jobs=1,
        ),
    )

    assert set(result.example_counts) == {"train", "validation"}
    assert result.example_counts["train"] > 0
    np.testing.assert_array_equal(
        dataset.view("test").targets.labels,
        test_labels_before,
    )


def test_longitudinal_planner_service_uses_train_and_validation_but_not_test(
    tmp_path: Path,
) -> None:
    dataset = _longitudinal_dataset(tmp_path)
    burden = _burden()
    test_labels_before = dataset.view("test").targets.labels.copy()

    result = train_longitudinal_planner(
        dataset,
        fusion=MeanLongitudinalFusion(),
        burden=burden,
        ranker=KnowledgeCandidateRanker(
            kg_matrix=np.ones((2, 1), dtype=np.float32),
            burden=burden.composite,
        ),
        reward=LongitudinalReward(
            horizons=(5.0, 10.0),
            horizon_weights=(0.4, 0.6),
        ),
        config=PlannerTrainingConfig(
            seed=5,
            states_per_participant=1,
            candidate_actions=2,
            n_estimators=8,
            min_child_samples=1,
            n_jobs=1,
        ),
    )

    assert set(result.example_counts) == {"train", "validation"}
    assert result.example_counts["train"] > 0
    assert result.model.feature_width > 0
    np.testing.assert_array_equal(
        dataset.view("test").targets.labels,
        test_labels_before,
    )
