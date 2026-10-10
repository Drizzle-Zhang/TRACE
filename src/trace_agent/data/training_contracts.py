"""Native label identity and OOF provenance checks at the supervised boundary.

Inference never calls this module. Legacy externally prepared score bundles
without manifests retain their existing shape/value/split checks; their target
identity and OOF provenance cannot be certified by TRACE.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trace_agent.contracts.scores import ModuleScoreBatch
from trace_agent.contracts.task import TaskType
from trace_agent.data.split_table import DatasetSplits
from trace_agent.runtime.provenance import array_sha256, file_provenance, row_array_sha256


def target_row_fingerprints(root: Path, *, prefix: str, task: TaskType, rows: np.ndarray) -> dict[str, str]:
    """Fingerprint fitting targets without including validation/test labels."""
    names = ("y", "y_mask") if task is TaskType.PREVALENT else ("event", "time", "signed_time", "label_mask")
    return {
        name: row_array_sha256(
            np.load(root / f"{prefix}_{name}.npy", allow_pickle=False, mmap_mode="r"), rows
        )
        for name in names
    }


def validate_training_bundle(
    root: Path,
    *,
    prefix: str,
    scores: ModuleScoreBatch,
    splits: DatasetSplits,
    require_training_scores: bool,
) -> None:
    target_path = root / f"{prefix}_module_targets_manifest.json"
    score_path = root / f"{prefix}_score_bundle_manifest.json"
    if target_path.is_file() or score_path.is_file():
        target = json.loads(target_path.read_text(encoding="utf-8"))
        expected = (
            {"y", "y_mask"}
            if scores.task is TaskType.PREVALENT
            else {"event", "time", "signed_time", "label_mask"}
        )
        checks = (
            target.get("schema_version") == 1,
            target.get("task") == scores.task.value,
            target.get("eids_sha256") == array_sha256(scores.participant_ids),
            target.get("modules") == list(scores.modules),
            target.get("endpoints") == list(scores.endpoints),
            target.get("split_sha256") == splits.canonical_sha256,
            set(target.get("files", {})) == expected,
        )
        if not all(checks):
            raise ValueError("prepared target identity/split contract mismatch")
        for name in expected:
            record = target["files"][name]
            filename = f"{prefix}_{name}.npy"
            if record.get("file") != filename:
                raise ValueError(f"prepared target filename mismatch: {name}")
            if record.get("sha256") != file_provenance(root / filename)["sha256"]:
                raise ValueError(f"prepared target hash mismatch: {filename}")
        # A native target bundle must not silently fall back to legacy scores.
        if not score_path.is_file():
            raise ValueError("native target bundle requires a score manifest; rerun score-modules")

    if require_training_scores and score_path.is_file():
        manifest = json.loads(score_path.read_text(encoding="utf-8"))
        provenance = manifest.get("provenance", {})
        if provenance.get("score_mode") != "training":
            raise ValueError("training requires OOF scores with score_mode=training, not inference scores")
        train_rows = splits.indices("train")
        expected_identity = {
            "split_sha256": splits.canonical_sha256,
            "train_rows_sha256": array_sha256(train_rows),
            "train_eids_sha256": array_sha256(scores.participant_ids[train_rows]),
        }
        for name, expected_value in expected_identity.items():
            if provenance.get(name) != expected_value:
                raise ValueError(f"OOF {name} mismatch or missing; rerun score-modules in training mode")
        fitted_targets = provenance.get("train_target_fingerprints")
        if fitted_targets is not None:
            current_targets = target_row_fingerprints(root, prefix=prefix, task=scores.task, rows=train_rows)
            if fitted_targets != current_targets:
                raise ValueError("module OOF train target mismatch; rerun train-modules and score-modules")
