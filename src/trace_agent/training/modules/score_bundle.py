"""Hash-checked module banks, train OOF caches and score bundle publication."""

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from trace_agent.contracts.scores import ModuleScoreBatch
from trace_agent.contracts.task import TaskType
from trace_agent.data.module_features import ModuleFeatures
from trace_agent.data.split_table import normalize_eids
from trace_agent.risk_skills.module_model import ModuleModel
from trace_agent.runtime.model_artifact import TraceModelArtifact, load_trace_artifact
from trace_agent.runtime.provenance import file_provenance
from trace_agent.training.manifests import write_training_manifest

_BANK_IDENTITY_FIELDS = {
    "feature_names",
    "feature_sources",
    "eids_sha256",
    "split_sha256",
    "folds_sha256",
    "fold_source",
    "settings",
    "oof_cache",
    "training",
}


def _has_bank_training_identity(manifest: Mapping[str, Any]) -> bool:
    return bool(_BANK_IDENTITY_FIELDS.intersection(manifest) or "train_target_fingerprints" in manifest)


def _check_bank_identity_fields(manifest: Mapping[str, Any]) -> None:
    missing = _BANK_IDENTITY_FIELDS.difference(manifest)
    if missing:
        raise ValueError(f"module bank training identity is incomplete: missing {sorted(missing)}")


def _check_bank_model_identity(
    manifest: Mapping[str, Any], record: Mapping[str, Any], artifact: TraceModelArtifact
) -> None:
    # Minimal compatibility envelopes have no training identity. A bank/model
    # declaring this identity must retain and agree on the complete record.
    model_has_identity = "inputs" in artifact.provenance or "settings" in artifact.training
    if not (_has_bank_training_identity(manifest) or model_has_identity):
        return
    _check_bank_identity_fields(manifest)
    inputs = artifact.provenance.get("inputs")
    source_hashes = (
        {name: value.get("sha256") for name, value in inputs.items() if isinstance(value, Mapping)}
        if isinstance(inputs, Mapping)
        else None
    )
    if source_hashes != manifest["feature_sources"]:
        raise ValueError("module bank model input fingerprint mismatch")
    for bank_field, model_field in (
        ("split_sha256", "split_table_canonical_sha256"),
        ("folds_sha256", "folds_sha256"),
        ("fold_source", "fold_source"),
    ):
        if model_field not in artifact.provenance or artifact.provenance[model_field] != manifest[bank_field]:
            raise ValueError(f"module bank model {bank_field} mismatch")
    if artifact.training.get("settings") != manifest["settings"]:
        raise ValueError("module bank model training settings mismatch")
    if artifact.training.get("train_rows_used") != record["metadata"]["eligible_train_count"]:
        raise ValueError("module bank model eligible train count mismatch")
    for field in ("validation_rows_used", "test_rows_used"):
        if artifact.training.get(field) != manifest["training"][field]:
            raise ValueError(f"module bank model {field} mismatch")
    key = (str(record["module"]), str(record["endpoint"]))
    selected = {
        (str(item["module"]), str(item["endpoint"])): tuple(item["names"])
        for item in manifest["feature_names"]
    }
    if artifact.model.feature_names != selected.get(key):
        raise ValueError("module bank model selected feature order mismatch")
    if record.get("score_scale") != artifact.model.score_scale:
        raise ValueError("module bank model score scale mismatch")


def pair_filename(module: str, endpoint: str) -> str:
    payload = json.dumps([module, endpoint], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest() + ".joblib"


def checked_file(root: Path, record: Mapping[str, Any]) -> Path:
    name = str(record["file"])
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("artifact file must be a local filename")
    path = root / name
    if file_provenance(path)["sha256"] != record["sha256"]:
        raise ValueError(f"artifact hash mismatch: {name}")
    return path


def local_record(path: Path) -> dict[str, Any]:
    provenance = file_provenance(path)
    return {"file": path.name, "sha256": provenance["sha256"], "size": provenance["size"]}


def write_oof_cache(
    path: Path,
    *,
    train_rows: np.ndarray,
    train_eids: np.ndarray,
    folds: np.ndarray,
    scores: np.ndarray,
    pred_mask: np.ndarray,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                train_rows=train_rows,
                train_eids=np.asarray(train_eids, dtype=np.str_),
                folds=folds,
                scores=scores,
                pred_mask=pred_mask,
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_module_bank(
    root: str | Path, *, task: TaskType | str
) -> tuple[dict[str, Any], dict[tuple[str, str], ModuleModel]]:
    directory = Path(root).expanduser().resolve()
    manifest_path = directory / "module_bank_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or TaskType.parse(manifest["task"]) is not TaskType.parse(task):
        raise ValueError("module bank schema/task mismatch")
    if _has_bank_training_identity(manifest):
        _check_bank_identity_fields(manifest)
    models: dict[tuple[str, str], ModuleModel] = {}
    seen: set[tuple[str, str]] = set()
    for record in manifest["pairs"]:
        key = (str(record["module"]), str(record["endpoint"]))
        if key in seen:
            raise ValueError("duplicate module bank pair")
        seen.add(key)
        if key[0] not in manifest["modules"] or key[1] not in manifest["endpoints"]:
            raise ValueError("unknown module bank pair reference")
        if record["status"] == "skipped":
            continue
        if record["status"] != "trained":
            raise ValueError("unknown module bank pair status")
        artifact = load_trace_artifact(
            checked_file(directory, record["artifact"]), component="module", task=task
        )
        if artifact.modules != (key[0],) or artifact.endpoints != (key[1],):
            raise ValueError("module bank pair identity mismatch")
        if not isinstance(artifact.model, ModuleModel):
            raise TypeError("module bank requires native ModuleModel")
        _check_bank_model_identity(manifest, record, artifact)
        if (
            "train_target_fingerprints" in manifest
            and artifact.provenance.get("train_target_fingerprints") != manifest["train_target_fingerprints"]
        ):
            raise ValueError("module bank model train target fingerprint mismatch")
        models[key] = artifact.model
    if "feature_names" in manifest:
        declared = {(str(item["module"]), str(item["endpoint"])) for item in manifest["feature_names"]}
        if len(declared) != len(manifest["feature_names"]) or seen != declared:
            raise ValueError("module bank configured pair coverage mismatch")
    return manifest, models


def load_oof_cache(root: Path, manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = checked_file(root, manifest["oof_cache"])
    with np.load(path, allow_pickle=False) as archive:
        required = {"train_rows", "train_eids", "folds", "scores", "pred_mask"}
        if set(archive.files) != required:
            raise ValueError("OOF cache fields mismatch")
        cache = {key: archive[key].copy() for key in required}
    n = cache["train_rows"].size
    expected = (n, len(manifest["modules"]), len(manifest["endpoints"]))
    if any(cache[key].shape != (n,) for key in ("train_rows", "train_eids", "folds")):
        raise ValueError("OOF cache row axis mismatch")
    if cache["scores"].shape != expected or cache["pred_mask"].shape != expected:
        raise ValueError("OOF cache score axes mismatch")
    mask = cache["pred_mask"].astype(bool)
    if not np.isfinite(cache["scores"][mask]).all() or not np.isnan(cache["scores"][~mask]).all():
        raise ValueError("OOF cache score/mask mismatch")
    if "pairs" in manifest:
        trained = {
            (str(record["module"]), str(record["endpoint"]))
            for record in manifest["pairs"]
            if record["status"] == "trained"
        }
        for m, module in enumerate(manifest["modules"]):
            for d, endpoint in enumerate(manifest["endpoints"]):
                if (module, endpoint) not in trained and mask[:, m, d].any():
                    raise ValueError("OOF cache contains predictions for an untrained pair")
        for record in manifest["pairs"]:
            if record["status"] != "trained":
                continue
            count = record.get("metadata", {}).get("oof_valid_count")
            if count is None and not _has_bank_training_identity(manifest):
                continue
            m = manifest["modules"].index(record["module"])
            d = manifest["endpoints"].index(record["endpoint"])
            if count is None or count != int(mask[:, m, d].sum()):
                raise ValueError("OOF cache recorded coverage mismatch")
    return cache


def score_module_features(
    features: ModuleFeatures,
    models: Mapping[tuple[str, str], ModuleModel],
    *,
    task: TaskType | str,
    predict_rows: np.ndarray | None = None,
) -> ModuleScoreBatch:
    modules = tuple(item.name for item in features.catalog.modules)
    endpoints = tuple(item.name for item in features.catalog.endpoints)
    scores = np.full((features.participant_ids.size, len(modules), len(endpoints)), np.nan, dtype=np.float32)
    mask = np.zeros(scores.shape, dtype=bool)
    allowed = np.ones(features.participant_ids.size, dtype=bool)
    if predict_rows is not None:
        allowed[:] = False
        allowed[np.asarray(predict_rows, dtype=np.int64)] = True
    for key, model in models.items():
        if (
            key not in features.matrices
            or (model.module, model.endpoint) != key
            or model.task is not TaskType.parse(task)
        ):
            raise ValueError("module scoring model identity/task mismatch")
        if model.feature_names != features.feature_names[key]:
            raise ValueError("module scoring feature order mismatch")
        matrix = features.matrices[key]
        selected = allowed & np.isfinite(matrix).any(axis=1)
        if not selected.any():
            continue
        predicted = model.predict(matrix[selected])
        positions = np.flatnonzero(selected)
        valid = np.isfinite(predicted)
        m, d = modules.index(key[0]), endpoints.index(key[1])
        scores[positions[valid], m, d] = predicted[valid]
        mask[positions[valid], m, d] = True
    return ModuleScoreBatch(
        participant_ids=features.participant_ids,
        modules=modules,
        endpoints=endpoints,
        scores=scores,
        pred_mask=mask,
        availability=features.availability,
        task=TaskType.parse(task),
    )


def validate_oof_identity(
    cache: Mapping[str, np.ndarray],
    train_rows: np.ndarray,
    participant_ids: np.ndarray,
) -> None:
    if not np.array_equal(cache["train_rows"], train_rows):
        raise ValueError("OOF train row mismatch")
    if not np.array_equal(normalize_eids(cache["train_eids"]), normalize_eids(participant_ids)[train_rows]):
        raise ValueError("OOF participant mismatch")


def apply_train_oof(
    batch: ModuleScoreBatch,
    cache: Mapping[str, np.ndarray],
    train_rows: np.ndarray,
) -> ModuleScoreBatch:
    validate_oof_identity(cache, train_rows, batch.participant_ids)
    if cache["scores"].shape != (train_rows.size, len(batch.modules), len(batch.endpoints)):
        raise ValueError("OOF score axes mismatch")
    scores = batch.scores.copy()
    mask = batch.pred_mask.copy()
    scores[train_rows] = cache["scores"]
    mask[train_rows] = cache["pred_mask"]
    return ModuleScoreBatch(
        participant_ids=batch.participant_ids,
        modules=batch.modules,
        endpoints=batch.endpoints,
        scores=scores,
        pred_mask=mask,
        availability=batch.availability,
        task=batch.task,
    )


def write_score_bundle(
    root: str | Path,
    *,
    prefix: str,
    batch: ModuleScoreBatch,
    provenance: Mapping[str, Any],
) -> Path:
    directory = Path(root).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be a non-empty filename stem")
    suffix = "oof_probs" if batch.task is TaskType.PREVALENT else "scores"
    arrays = {
        "participant_ids": np.asarray(batch.participant_ids, dtype=np.str_),
        "modules": np.asarray(batch.modules, dtype=np.str_),
        "diseases": np.asarray(batch.endpoints, dtype=np.str_),
        suffix: batch.scores,
        "pred_mask": batch.pred_mask,
        "avail": batch.availability,
    }
    files = {}
    for name, values in arrays.items():
        path = directory / f"{prefix}_{name}.npy"
        temporary = path.with_name(path.name + ".tmp")
        try:
            with temporary.open("wb") as handle:
                np.save(handle, values, allow_pickle=False)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        files[name] = local_record(path)
    return write_training_manifest(
        directory / f"{prefix}_score_bundle_manifest.json",
        {
            "schema_version": 1,
            "task": batch.task.value,
            "modules": list(batch.modules),
            "endpoints": list(batch.endpoints),
            "files": files,
            "provenance": dict(provenance),
        },
    )
