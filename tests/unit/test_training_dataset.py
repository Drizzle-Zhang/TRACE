from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trace_agent.contracts.task import SplitRole, TaskType
from trace_agent.data.split_table import SplitColumns
from trace_agent.data.training_dataset import load_training_dataset


def _write_score_bundle(
    root: Path,
    *,
    task: TaskType,
    eids: np.ndarray,
) -> None:
    prefix = "cohort"
    n = int(eids.size)
    np.save(root / f"{prefix}_eids.npy", eids)
    np.save(root / f"{prefix}_modules.npy", np.array(["M1", "M2"]))
    np.save(root / f"{prefix}_diseases.npy", np.array(["D1"]))
    scores = np.linspace(0.1, 0.8, n * 2, dtype=np.float32).reshape(n, 2, 1)
    suffix = "oof_probs" if task is TaskType.PREVALENT else "scores"
    np.save(root / f"{prefix}_{suffix}.npy", scores)
    np.save(root / f"{prefix}_pred_mask.npy", np.ones_like(scores, dtype=bool))
    np.save(root / f"{prefix}_avail.npy", np.ones((n, 2), dtype=bool))


def _write_split_table(root: Path, records: list[tuple[int, int, str]]) -> Path:
    path = root / "splits.csv"
    pd.DataFrame(records, columns=["row_index", "eid", "split"]).to_csv(path, index=False)
    return path


def _write_prevalent_bundle(root: Path) -> Path:
    eids = np.array([10, 11, 12, 13])
    _write_score_bundle(root, task=TaskType.PREVALENT, eids=eids)
    np.save(root / "cohort_y.npy", np.array([[0], [1], [1], [0]], dtype=np.float32))
    np.save(root / "cohort_y_mask.npy", np.ones((4, 1), dtype=bool))
    return _write_split_table(
        root,
        [(2, 12, "train"), (0, 10, "train"), (1, 11, "validation"), (3, 13, "test")],
    )


def _write_longitudinal_bundle(root: Path) -> Path:
    eids = np.array([20, 21, 22])
    _write_score_bundle(root, task=TaskType.LONGITUDINAL, eids=eids)
    event = np.array([[1], [0], [1]], dtype=np.float32)
    time = np.array([[2.0], [8.0], [4.0]], dtype=np.float32)
    np.save(root / "cohort_event.npy", event)
    np.save(root / "cohort_time.npy", time)
    np.save(root / "cohort_signed_time.npy", np.where(event > 0, time, -time))
    np.save(root / "cohort_label_mask.npy", np.ones((3, 1), dtype=bool))
    return _write_split_table(
        root,
        [(0, 20, "train"), (1, 21, "validation"), (2, 22, "test")],
    )


@pytest.mark.parametrize("include_row_index", [False, True])
def test_prevalent_training_view_slices_all_participant_arrays_together(
    tmp_path: Path,
    include_row_index: bool,
) -> None:
    split_path = _write_prevalent_bundle(tmp_path)
    if not include_row_index:
        pd.read_csv(split_path).drop(columns="row_index").to_csv(split_path, index=False)

    dataset = load_training_dataset(
        tmp_path,
        prefix="cohort",
        task="prevalent",
        split_table=split_path,
    )
    train = dataset.view(SplitRole.TRAIN)

    assert train.role is SplitRole.TRAIN
    assert train.row_indices.tolist() == [0, 2]
    assert train.scores.participant_ids.tolist() == [10, 12]
    np.testing.assert_array_equal(
        train.targets.labels,
        np.load(tmp_path / "cohort_y.npy")[[0, 2]],
    )
    np.testing.assert_array_equal(
        train.targets.label_mask,
        np.load(tmp_path / "cohort_y_mask.npy")[[0, 2]],
    )


def test_longitudinal_training_view_loads_survival_targets(tmp_path: Path) -> None:
    split_path = _write_longitudinal_bundle(tmp_path)

    dataset = load_training_dataset(
        tmp_path,
        prefix="cohort",
        task="longitudinal",
        split_table=split_path,
    )
    validation = dataset.view("validation")

    assert validation.scores.task is TaskType.LONGITUDINAL
    assert validation.scores.participant_ids.tolist() == [21]
    np.testing.assert_array_equal(validation.targets.labels, [[0]])
    np.testing.assert_array_equal(validation.targets.event_time, [[8.0]])
    np.testing.assert_array_equal(validation.targets.signed_time, [[-8.0]])


def test_longitudinal_training_rejects_misaligned_target_shape(tmp_path: Path) -> None:
    split_path = _write_longitudinal_bundle(tmp_path)
    np.save(tmp_path / "cohort_time.npy", np.ones((2, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="time shape"):
        load_training_dataset(
            tmp_path,
            prefix="cohort",
            task="longitudinal",
            split_table=split_path,
        )


def test_prevalent_training_rejects_nonbinary_observed_labels(tmp_path: Path) -> None:
    split_path = _write_prevalent_bundle(tmp_path)
    labels = np.load(tmp_path / "cohort_y.npy")
    labels[0, 0] = 0.5
    np.save(tmp_path / "cohort_y.npy", labels)

    with pytest.raises(ValueError, match="binary"):
        load_training_dataset(
            tmp_path,
            prefix="cohort",
            task="prevalent",
            split_table=split_path,
        )


def test_training_dataset_passes_explicit_split_column_names(tmp_path: Path) -> None:
    split_path = _write_prevalent_bundle(tmp_path)
    frame = pd.read_csv(split_path).rename(columns={"row_index": "row", "eid": "participant", "split": "set"})
    frame.to_csv(split_path, index=False)

    dataset = load_training_dataset(
        tmp_path,
        prefix="cohort",
        task="prevalent",
        split_table=split_path,
        split_columns=SplitColumns(row_index="row", eid="participant", split="set"),
    )

    assert dataset.view("test").scores.participant_ids.tolist() == [13]


@pytest.mark.parametrize(
    ("name", "row", "value", "message"),
    [
        ("signed_time", 1, 8.0, "signed_time.*event"),
        ("signed_time", 0, -2.0, "signed_time.*event"),
        ("signed_time", 0, 3.0, "signed_time.*time"),
        ("time", 0, 0.0, "time.*positive"),
    ],
)
def test_survival_targets_reject_inconsistent_censoring(tmp_path, name, row, value, message):
    split_path = _write_longitudinal_bundle(tmp_path)
    path = tmp_path / f"cohort_{name}.npy"
    values = np.load(path)
    values[row, 0] = value
    np.save(path, values)
    with pytest.raises(ValueError, match=message):
        load_training_dataset(tmp_path, prefix="cohort", task="longitudinal", split_table=split_path)


def test_survival_validation_ignores_unobserved_placeholders(tmp_path):
    split_path = _write_longitudinal_bundle(tmp_path)
    mask = np.ones((3, 1), dtype=bool)
    mask[0] = False
    np.save(tmp_path / "cohort_label_mask.npy", mask)
    for name in ("event", "time", "signed_time"):
        values = np.load(tmp_path / f"cohort_{name}.npy")
        values[0] = np.nan
        np.save(tmp_path / f"cohort_{name}.npy", values)
    dataset = load_training_dataset(tmp_path, prefix="cohort", task="longitudinal", split_table=split_path)
    assert not dataset.targets.label_mask[0, 0]
