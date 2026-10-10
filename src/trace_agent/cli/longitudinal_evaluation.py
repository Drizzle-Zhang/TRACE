"""Evaluate frozen absolute risks at named horizons, never raw Cox log-risk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from trace_agent.contracts.task import SplitRole
from trace_agent.data.split_table import canonical_eids_equal
from trace_agent.data.training_dataset import TrainingDataset
from trace_agent.planning.rewards.longitudinal import build_horizon_labels
from trace_agent.runtime.provenance import array_sha256
from trace_agent.training.common import masked_classification_metrics


def horizon_years(name: str) -> float:
    """Require explicit year units, e.g. 5y, 10y or 2.5y."""
    try:
        value = float(name[:-1]) if name.endswith("y") else float("nan")
    except ValueError as exc:
        raise ValueError(f"invalid year-based risk horizon: {name!r}") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"invalid year-based risk horizon: {name!r}")
    return value


def evaluate_horizon_predictions(dataset: TrainingDataset, path: str | Path) -> dict[str, Any]:
    test = dataset.view(SplitRole.TEST)
    if test.targets.event_time is None:
        raise ValueError("incident evaluation requires event times and censoring information")
    expected = (dataset.scores.n_participants, len(dataset.scores.endpoints))
    with np.load(path, allow_pickle=False) as archive:
        metadata = {"participant_ids", "endpoints", "valid_mask"}
        if not metadata.issubset(archive.files):
            raise ValueError("horizon predictions require participant_ids, endpoints and valid_mask")
        eids = archive["participant_ids"]
        if eids.shape != (expected[0],) or not canonical_eids_equal(eids, dataset.scores.participant_ids):
            raise ValueError("horizon prediction participant IDs do not match the score bundle order")
        endpoints = archive["endpoints"]
        if endpoints.shape != (expected[1],) or tuple(endpoints.tolist()) != dataset.scores.endpoints:
            raise ValueError("horizon prediction endpoint order does not match the score bundle")
        valid = archive["valid_mask"]
        if valid.shape != expected or valid.dtype.kind != "b":
            raise ValueError("horizon prediction valid_mask must be a boolean participant-by-endpoint array")
        names = sorted(set(archive.files) - metadata, key=horizon_years)
        if not names:
            raise ValueError("horizon predictions contain no absolute-risk probabilities")
        years = [horizon_years(name) for name in names]
        if len(set(years)) != len(years):
            raise ValueError("horizon predictions contain duplicate year horizons")
        reports = {}
        for name, horizon in zip(names, years):
            predictions = np.asarray(archive[name], dtype=np.float64)
            if predictions.shape != expected:
                raise ValueError(f"horizon {name} prediction shape must be {expected}")
            if np.any(valid & (~np.isfinite(predictions) | (predictions < 0) | (predictions > 1))):
                raise ValueError(f"horizon {name} valid probabilities must be finite and in [0, 1]")
            labels, known = build_horizon_labels(
                event=test.targets.labels,
                time=test.targets.event_time,
                label_mask=test.targets.label_mask,
                horizons=np.asarray([horizon]),
            )
            labels = labels.reshape(test.targets.labels.shape)
            known = known.reshape(test.targets.label_mask.shape)
            metrics = masked_classification_metrics(
                predictions[test.row_indices], labels, known & valid[test.row_indices]
            )
            metrics["horizon_years"] = horizon
            metrics["n_known_outcomes"] = int(known.sum())
            metrics["n_unknown_outcomes"] = int((test.targets.label_mask & ~known).sum())
            reports[name] = metrics
    return {
        "task": "longitudinal",
        "split": SplitRole.TEST.value,
        "participants": test.scores.n_participants,
        "row_indices_sha256": array_sha256(test.row_indices),
        "eids_sha256": array_sha256(dataset.splits.eids(SplitRole.TEST)),
        "predictions_path": str(Path(path).resolve()),
        "metric_method": "known_outcomes_at_horizon (complete-case, not IPCW)",
        "horizons": reports,
    }
