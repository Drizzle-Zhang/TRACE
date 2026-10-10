from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trace_agent.contracts.scores import ModuleScoreBatch
from trace_agent.contracts.task import SplitRole, TaskType
from trace_agent.data.split_table import DatasetSplits
from trace_agent.data.training_dataset import TrainingDataset, TrainingTargets
from trace_agent.training.manifests import split_provenance, write_training_manifest


def _dataset(tmp_path: Path) -> TrainingDataset:
    split_path = tmp_path / "splits.csv"
    split_path.write_text(
        "row_index,eid,split\n0,10,train\n1,11,validation\n2,12,test\n",
        encoding="utf-8",
    )
    roles = {
        SplitRole.TRAIN: np.array([0], dtype=np.int64),
        SplitRole.VALIDATION: np.array([1], dtype=np.int64),
        SplitRole.TEST: np.array([2], dtype=np.int64),
    }
    eids = {
        SplitRole.TRAIN: np.array(["10"]),
        SplitRole.VALIDATION: np.array(["11"]),
        SplitRole.TEST: np.array(["12"]),
    }
    splits = DatasetSplits(
        table_path=split_path,
        row_indices=roles,
        eid_values=eids,
        file_sha256="a" * 64,
        canonical_sha256="b" * 64,
    )
    scores = ModuleScoreBatch(
        participant_ids=np.array([10, 11, 12]),
        modules=("M1",),
        endpoints=("D1",),
        scores=np.array([[[0.1]], [[0.2]], [[0.3]]], dtype=np.float32),
        pred_mask=np.ones((3, 1, 1), dtype=bool),
        availability=np.ones((3, 1), dtype=bool),
        task=TaskType.PREVALENT,
    )
    targets = TrainingTargets(
        labels=np.array([[0], [1], [0]], dtype=np.float32),
        label_mask=np.ones((3, 1), dtype=bool),
    )
    return TrainingDataset(scores=scores, targets=targets, splits=splits)


def test_split_provenance_records_each_role_without_exposing_eids(tmp_path: Path) -> None:
    provenance = split_provenance(_dataset(tmp_path))

    assert provenance["split_table_file_sha256"] == "a" * 64
    assert provenance["split_table_canonical_sha256"] == "b" * 64
    assert provenance["splits"]["train"]["count"] == 1
    assert len(provenance["splits"]["test"]["eids_sha256"]) == 64
    assert "eids" not in provenance["splits"]["test"]


def test_write_training_manifest_is_atomic_json(tmp_path: Path) -> None:
    path = write_training_manifest(
        tmp_path / "training_manifest.json",
        {"component": "integration", "count": np.int64(3)},
    )

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "component": "integration",
        "count": 3,
    }
    assert not (tmp_path / "training_manifest.json.tmp").exists()
