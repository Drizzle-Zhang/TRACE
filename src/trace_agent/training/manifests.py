from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from trace_agent.contracts.task import SplitRole
from trace_agent.data.training_dataset import TrainingDataset
from trace_agent.runtime.provenance import array_sha256, row_array_sha256

__all__ = ["array_sha256", "split_provenance", "write_training_manifest"]


def split_provenance(dataset: TrainingDataset) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for role in (SplitRole.TRAIN, SplitRole.VALIDATION, SplitRole.TEST):
        rows = dataset.splits.indices(role)
        eids = dataset.splits.eids(role)
        splits[role.value] = {
            "count": int(rows.size),
            "row_indices_sha256": array_sha256(rows),
            "eids_sha256": array_sha256(eids),
        }
    return {
        "split_table_path": str(dataset.splits.table_path),
        "split_table_file_sha256": dataset.splits.file_sha256,
        "split_table_canonical_sha256": dataset.splits.canonical_sha256,
        "splits": splits,
    }


def supervised_input_provenance(dataset: TrainingDataset) -> dict[str, dict[str, str]]:
    """Bind fitted inputs to train/validation values without depending on test labels."""
    arrays = {
        "scores": dataset.scores.scores,
        "pred_mask": dataset.scores.pred_mask,
        "availability": dataset.scores.availability,
        "labels": dataset.targets.labels,
        "label_mask": dataset.targets.label_mask,
    }
    for key in ("event_time", "signed_time"):
        values = getattr(dataset.targets, key)
        if values is not None:
            arrays[key] = values
    return {
        role.value: {
            name: row_array_sha256(values, dataset.splits.indices(role)) for name, values in arrays.items()
        }
        for role in (SplitRole.TRAIN, SplitRole.VALIDATION)
    }


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")


def write_training_manifest(
    path: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                dict(payload),
                default=_json_default,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
