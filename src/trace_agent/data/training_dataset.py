from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trace_agent.contracts.scores import ModuleScoreBatch
from trace_agent.contracts.task import SplitRole, TaskType
from trace_agent.data.arrays import load_module_score_bundle, select_participants
from trace_agent.data.split_table import (
    DatasetSplits,
    SplitColumns,
    load_split_table,
)
from trace_agent.data.training_contracts import validate_training_bundle


def _load_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False, mmap_mode="r")


def _validate_shape(
    values: np.ndarray,
    *,
    expected: tuple[int, int],
    label: str,
) -> None:
    if values.shape != expected:
        raise ValueError(f"{label} shape must be {expected}, got {values.shape}")


def _validate_binary(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    label: str,
) -> None:
    observed = np.asarray(mask, dtype=bool)
    if np.any(observed & ~np.isfinite(values)):
        raise ValueError(f"observed {label} values must be finite")
    selected = np.asarray(values)[observed]
    if selected.size and not np.isin(selected, (0, 1)).all():
        raise ValueError(f"observed {label} values must be binary")


@dataclass(frozen=True)
class TrainingTargets:
    labels: np.ndarray
    label_mask: np.ndarray
    event_time: np.ndarray | None = None
    signed_time: np.ndarray | None = None

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels, dtype=np.float32)
        label_mask = np.asarray(self.label_mask, dtype=bool)
        event_time = None if self.event_time is None else np.asarray(self.event_time, dtype=np.float32)
        signed_time = None if self.signed_time is None else np.asarray(self.signed_time, dtype=np.float32)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "label_mask", label_mask)
        object.__setattr__(self, "event_time", event_time)
        object.__setattr__(self, "signed_time", signed_time)


@dataclass(frozen=True)
class TrainingView:
    role: SplitRole
    row_indices: np.ndarray
    scores: ModuleScoreBatch
    targets: TrainingTargets

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", SplitRole.parse(self.role))
        object.__setattr__(
            self,
            "row_indices",
            np.asarray(self.row_indices, dtype=np.int64).reshape(-1),
        )


@dataclass(frozen=True)
class TrainingDataset:
    scores: ModuleScoreBatch
    targets: TrainingTargets
    splits: DatasetSplits

    def view(self, role: SplitRole | str) -> TrainingView:
        resolved = SplitRole.parse(role)
        rows = self.splits.indices(resolved)
        return TrainingView(
            role=resolved,
            row_indices=rows,
            scores=select_participants(self.scores, rows),
            targets=_select_targets(self.targets, rows),
        )


def _select_targets(targets: TrainingTargets, rows: np.ndarray) -> TrainingTargets:
    return TrainingTargets(
        labels=targets.labels[rows],
        label_mask=targets.label_mask[rows],
        event_time=(None if targets.event_time is None else targets.event_time[rows]),
        signed_time=(None if targets.signed_time is None else targets.signed_time[rows]),
    )


def _load_prevalent_targets(
    root: Path,
    *,
    prefix: str,
    expected: tuple[int, int],
) -> TrainingTargets:
    labels = np.asarray(
        _load_array(root / f"{prefix}_y.npy"),
        dtype=np.float32,
    )
    label_mask = np.asarray(
        _load_array(root / f"{prefix}_y_mask.npy"),
        dtype=bool,
    )
    _validate_shape(labels, expected=expected, label="y")
    _validate_shape(label_mask, expected=expected, label="y_mask")
    _validate_binary(labels, label_mask, label="labels")
    return TrainingTargets(labels=labels, label_mask=label_mask)


def _load_longitudinal_targets(
    root: Path,
    *,
    prefix: str,
    expected: tuple[int, int],
) -> TrainingTargets:
    event = np.asarray(
        _load_array(root / f"{prefix}_event.npy"),
        dtype=np.float32,
    )
    time = np.asarray(
        _load_array(root / f"{prefix}_time.npy"),
        dtype=np.float32,
    )
    signed_time = np.asarray(
        _load_array(root / f"{prefix}_signed_time.npy"),
        dtype=np.float32,
    )
    label_mask = np.asarray(
        _load_array(root / f"{prefix}_label_mask.npy"),
        dtype=bool,
    )
    for label, values in (
        ("event", event),
        ("time", time),
        ("signed_time", signed_time),
        ("label_mask", label_mask),
    ):
        _validate_shape(values, expected=expected, label=label)
    _validate_binary(event, label_mask, label="event")
    if np.any(label_mask & (~np.isfinite(time) | (time <= 0))):
        raise ValueError("observed time values must be finite and positive")
    if np.any(label_mask & ~np.isfinite(signed_time)):
        raise ValueError("observed signed_time values must be finite")
    if np.any(label_mask & (signed_time != np.where(event == 1, time, -time))):
        raise ValueError("observed signed_time must equal time for event=1 and -time for event=0")
    return TrainingTargets(
        labels=event,
        label_mask=label_mask,
        event_time=time,
        signed_time=signed_time,
    )


def load_training_targets(
    input_dir: str | Path,
    *,
    prefix: str,
    task: TaskType | str,
    expected: tuple[int, int],
) -> TrainingTargets:
    """Load validated targets without requiring precomputed module scores."""
    root = Path(input_dir).expanduser().resolve()
    if TaskType.parse(task) is TaskType.PREVALENT:
        return _load_prevalent_targets(root, prefix=prefix, expected=expected)
    return _load_longitudinal_targets(root, prefix=prefix, expected=expected)


def load_training_dataset(
    input_dir: str | Path,
    *,
    prefix: str,
    task: TaskType | str,
    split_table: str | Path,
    split_columns: SplitColumns | None = None,
    allow_legacy_pickle_names: bool = False,
    require_training_scores: bool = True,
) -> TrainingDataset:
    root = Path(input_dir).expanduser().resolve()
    resolved_task = TaskType.parse(task)
    scores = load_module_score_bundle(
        root,
        prefix=prefix,
        task=resolved_task,
        allow_legacy_pickle_names=allow_legacy_pickle_names,
    )
    splits = load_split_table(
        split_table,
        participant_ids=scores.participant_ids,
        columns=split_columns,
        require_complete=True,
    )
    validate_training_bundle(
        root,
        prefix=prefix,
        scores=scores,
        splits=splits,
        require_training_scores=require_training_scores,
    )
    targets = load_training_targets(
        root,
        prefix=prefix,
        task=resolved_task,
        expected=(scores.n_participants, len(scores.endpoints)),
    )
    return TrainingDataset(scores=scores, targets=targets, splits=splits)
