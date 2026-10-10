"""Offline outcome alignment and task-specific module training targets."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from trace_agent.contracts.task import TaskType
from trace_agent.data.module_features import ModuleCatalog
from trace_agent.data.split_table import normalize_eids, read_eid_table
from trace_agent.data.training_dataset import TrainingTargets


def encode_outcomes(
    target_y: np.ndarray,
    time: np.ndarray,
    *,
    task: TaskType | str,
) -> TrainingTargets:
    """Encode one endpoint; missing/ineligible outcomes retain their row."""
    target = np.asarray(target_y, dtype=float).reshape(-1)
    years = np.asarray(time, dtype=float).reshape(-1)
    if target.shape != years.shape:
        raise ValueError("target and time shape mismatch")
    if np.isinf(target).any() or np.isinf(years).any():
        raise ValueError("outcomes contain infinity")
    known = np.isfinite(target)
    if not np.isin(target[known], [0.0, 1.0]).all():
        raise ValueError("target_y must be 0 or 1")
    event = known & (target == 1)
    if np.any(event & (years == 0)):
        raise ValueError("event time must not be zero")
    resolved = TaskType.parse(task)
    if resolved is TaskType.PREVALENT:
        mask = (known & (target == 0)) | (event & (years < 0))
        labels = np.where(mask, event.astype(float), np.nan)[:, None]
        return TrainingTargets(labels, mask[:, None])
    if np.any(known & (target == 0) & np.isfinite(years) & (years <= 0)):
        raise ValueError("censoring follow-up must be positive")
    mask = known & np.isfinite(years) & (years > 0)
    labels = np.where(mask, target, np.nan)
    times = np.where(mask, years, np.nan)
    signed = np.where(labels == 1, times, -times)
    return TrainingTargets(labels[:, None], mask[:, None], times[:, None], signed[:, None])


def load_module_outcomes(
    catalog: ModuleCatalog,
    participant_ids: np.ndarray,
    *,
    task: TaskType | str,
) -> TrainingTargets:
    """Explicitly align unique outcome participant IDs; this is not a scoring input."""
    ids = normalize_eids(participant_ids)
    parts: list[TrainingTargets] = []
    for endpoint in catalog.endpoints:
        frame = read_eid_table(endpoint.outcome_path, eid_column=endpoint.eid_column)
        required = {endpoint.eid_column, endpoint.target_column, endpoint.time_column}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"outcome {endpoint.name!r} missing columns: {missing}")
        outcome_ids = normalize_eids(frame[endpoint.eid_column].to_numpy(), label=endpoint.name)
        if np.unique(outcome_ids).size != outcome_ids.size:
            raise ValueError(f"duplicate outcome participant ID: {endpoint.name}")
        if not set(outcome_ids).issubset(set(ids)):
            raise ValueError(f"unknown outcome participant ID: {endpoint.name}")
        frame.index = outcome_ids
        aligned = frame.reindex(ids)
        target = pd.to_numeric(aligned[endpoint.target_column], errors="raise").to_numpy(dtype=float)
        time = pd.to_numeric(aligned[endpoint.time_column], errors="raise").to_numpy(dtype=float)
        parts.append(encode_outcomes(target, time, task=task))
    labels = np.concatenate([part.labels for part in parts], axis=1)
    masks = np.concatenate([part.label_mask for part in parts], axis=1)
    if TaskType.parse(task) is TaskType.PREVALENT:
        return TrainingTargets(labels, masks)
    # Survival encoders always provide these arrays; keep that invariant explicit.
    times = []
    signed = []
    for part in parts:
        if part.event_time is None or part.signed_time is None:
            raise ValueError("longitudinal outcomes require event_time and signed_time")
        times.append(part.event_time)
        signed.append(part.signed_time)
    return TrainingTargets(labels, masks, np.concatenate(times, axis=1), np.concatenate(signed, axis=1))


def write_target_arrays(
    root: str | Path,
    *,
    prefix: str,
    targets: TrainingTargets,
    task: TaskType | str,
) -> dict[str, Path]:
    """Atomically materialize target files, separately from feature scores."""
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be a non-empty filename stem")
    destination = Path(root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    arrays = {"y": targets.labels, "y_mask": targets.label_mask}
    if TaskType.parse(task) is TaskType.LONGITUDINAL:
        if targets.event_time is None or targets.signed_time is None:
            raise ValueError("longitudinal targets require time and signed_time")
        arrays = {
            "event": targets.labels,
            "time": targets.event_time,
            "signed_time": targets.signed_time,
            "label_mask": targets.label_mask,
        }
    paths: dict[str, Path] = {}
    for name, values in arrays.items():
        path = destination / f"{prefix}_{name}.npy"
        temporary = path.with_name(path.name + ".tmp")
        try:
            with temporary.open("wb") as handle:
                np.save(handle, values, allow_pickle=False)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        paths[name] = path
    return paths
