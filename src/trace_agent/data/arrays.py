from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trace_agent.contracts.scores import ModuleScoreBatch
from trace_agent.contracts.task import TaskType
from trace_agent.runtime.provenance import file_provenance


def validate_score_bundle_manifest(root: Path, *, prefix: str, task: TaskType) -> None:
    """Check native score files when a manifest exists; retain legacy compatibility."""
    path = root / f"{prefix}_score_bundle_manifest.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    suffix = "oof_probs" if task is TaskType.PREVALENT else "scores"
    expected = {"eids", "modules", "diseases", suffix, "pred_mask", "avail"}
    if manifest.get("schema_version") != 1 or manifest.get("task") != task.value:
        raise ValueError("score bundle manifest schema/task mismatch")
    if set(manifest["files"]) != expected:
        raise ValueError("score bundle manifest file set mismatch")
    for name in expected:
        record = manifest["files"][name]
        if record["file"] != f"{prefix}_{name}.npy":
            raise ValueError("score bundle manifest filename mismatch")
        if file_provenance(root / record["file"])["sha256"] != record["sha256"]:
            raise ValueError("score bundle hash mismatch")


def _load_required(path: Path, *, memory_map: bool = False) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(
        path,
        allow_pickle=False,
        mmap_mode="r" if memory_map else None,
    )


def _load_names(path: Path, *, allow_legacy_pickle_names: bool) -> tuple[str, ...]:
    try:
        values = _load_required(path).reshape(-1)
    except ValueError as exc:
        if "Object arrays cannot be loaded" not in str(exc):
            raise
        if not allow_legacy_pickle_names:
            raise ValueError(
                f"{path} is a legacy pickled object array; set "
                "allow_legacy_pickle_names=true only for a trusted bundle"
            ) from exc
        values = np.load(path, allow_pickle=True).reshape(-1)
        if not all(isinstance(value, (str, np.str_)) for value in values):
            raise ValueError(f"legacy name array contains non-string objects: {path}")
    return tuple(str(value) for value in values.tolist())


def load_module_score_bundle(
    input_dir: str | Path,
    *,
    prefix: str,
    task: TaskType | str,
    allow_legacy_pickle_names: bool = False,
) -> ModuleScoreBatch:
    root = Path(input_dir).expanduser().resolve()
    resolved_task = TaskType.parse(task)
    validate_score_bundle_manifest(root, prefix=prefix, task=resolved_task)
    score_suffix = "oof_probs" if resolved_task is TaskType.PREVALENT else "scores"
    return ModuleScoreBatch(
        participant_ids=_load_required(root / f"{prefix}_eids.npy", memory_map=True),
        modules=_load_names(
            root / f"{prefix}_modules.npy",
            allow_legacy_pickle_names=allow_legacy_pickle_names,
        ),
        endpoints=_load_names(
            root / f"{prefix}_diseases.npy",
            allow_legacy_pickle_names=allow_legacy_pickle_names,
        ),
        scores=_load_required(
            root / f"{prefix}_{score_suffix}.npy",
            memory_map=True,
        ),
        pred_mask=_load_required(root / f"{prefix}_pred_mask.npy", memory_map=True),
        availability=_load_required(root / f"{prefix}_avail.npy", memory_map=True),
        task=resolved_task,
        metadata={
            "input_dir": str(root),
            "prefix": prefix,
            "memory_mapped": True,
        },
    )


def select_participant(batch: ModuleScoreBatch, index: int) -> ModuleScoreBatch:
    """Return one participant while preserving all named axes."""

    resolved = int(index)
    if resolved < 0:
        resolved += batch.n_participants
    if resolved < 0 or resolved >= batch.n_participants:
        raise IndexError(index)
    return ModuleScoreBatch(
        participant_ids=batch.participant_ids[resolved : resolved + 1],
        modules=batch.modules,
        endpoints=batch.endpoints,
        scores=batch.scores[resolved : resolved + 1],
        pred_mask=batch.pred_mask[resolved : resolved + 1],
        availability=batch.availability[resolved : resolved + 1],
        task=batch.task,
        metadata={
            **batch.metadata,
            "source_participant_index": resolved,
        },
    )


def select_participants(
    batch: ModuleScoreBatch,
    indices: np.ndarray,
) -> ModuleScoreBatch:
    """Select participant rows while preserving all named axes."""

    rows = np.asarray(indices, dtype=np.int64).reshape(-1)
    if rows.size and (int(np.min(rows)) < 0 or int(np.max(rows)) >= batch.n_participants):
        raise IndexError("participant indices are outside the score bundle")
    return ModuleScoreBatch(
        participant_ids=batch.participant_ids[rows],
        modules=batch.modules,
        endpoints=batch.endpoints,
        scores=batch.scores[rows],
        pred_mask=batch.pred_mask[rows],
        availability=batch.availability[rows],
        task=batch.task,
        metadata={
            **batch.metadata,
            "source_participant_indices": rows.copy(),
        },
    )
