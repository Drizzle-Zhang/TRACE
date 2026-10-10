"""Configuration entry points for preparing targets and training native modules."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from trace_agent import __version__
from trace_agent.contracts.task import TaskType
from trace_agent.data.module_features import ModuleFeatures, load_catalog, load_module_features
from trace_agent.data.module_outcomes import load_module_outcomes, write_target_arrays
from trace_agent.data.split_table import DatasetSplits, SplitColumns, id_column_from_config, load_split_table
from trace_agent.data.training_contracts import target_row_fingerprints
from trace_agent.data.training_dataset import load_training_targets
from trace_agent.identity import framework_metadata
from trace_agent.runtime.model_artifact import TraceModelArtifact, dump_trace_artifact
from trace_agent.runtime.provenance import file_provenance
from trace_agent.training.manifests import array_sha256, write_training_manifest
from trace_agent.training.modules.crossfit import crossfit_pair
from trace_agent.training.modules.folds import build_outer_folds
from trace_agent.training.modules.score_bundle import (
    apply_train_oof,
    checked_file,
    load_module_bank,
    load_oof_cache,
    local_record,
    pair_filename,
    score_module_features,
    validate_oof_identity,
    write_oof_cache,
    write_score_bundle,
)
from trace_agent.training.modules.settings import ModuleTrainingSettings


def section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"config.{name} must be a mapping")
    return value


def module_context(config: Mapping[str, Any]) -> tuple[ModuleFeatures, Path, str, TaskType]:
    module_data = section(config, "module_data")
    data = section(config, "data")
    if not module_data.get("catalog_path") or not data.get("input_dir") or not data.get("prefix"):
        raise ValueError("module_data.catalog_path and data.input_dir/prefix are required")
    features = load_module_features(load_catalog(str(module_data["catalog_path"])))
    return (
        features,
        Path(str(data["input_dir"])).expanduser().resolve(),
        str(data["prefix"]),
        TaskType.parse(config["task"]),
    )


def module_splits(config: Mapping[str, Any], participant_ids: np.ndarray) -> DatasetSplits:
    data = section(config, "data")
    split = section(data, "split_table")
    if not split.get("path"):
        raise ValueError("data.split_table.path is required for module training")
    columns = SplitColumns(
        str(split.get("row_index_column", "row_index")),
        id_column_from_config(split),
        str(split.get("split_column", "split")),
    )
    return load_split_table(str(split["path"]), participant_ids=participant_ids, columns=columns)


def pair_feature_records(features: ModuleFeatures) -> list[dict[str, Any]]:
    return [
        {"module": key[0], "endpoint": key[1], "names": list(names)}
        for key, names in features.feature_names.items()
    ]


def feature_source_hashes(features: ModuleFeatures) -> dict[str, str]:
    return {str(key): str(value["sha256"]) for key, value in features.provenance.items()}


def prepare_module_data(config: Mapping[str, Any]) -> Path:
    features, root, prefix, task = module_context(config)
    splits = module_splits(config, features.participant_ids)
    targets = load_module_outcomes(features.catalog, features.participant_ids, task=task)
    files = write_target_arrays(root, prefix=prefix, targets=targets, task=task)
    return write_training_manifest(
        root / f"{prefix}_module_targets_manifest.json",
        {
            "schema_version": 1,
            "framework": framework_metadata(),
            "task": task.value,
            "eids_sha256": array_sha256(features.participant_ids),
            "modules": [item.name for item in features.catalog.modules],
            "endpoints": [item.name for item in features.catalog.endpoints],
            "split_sha256": splits.canonical_sha256,
            "train_target_fingerprints": target_row_fingerprints(
                root, prefix=prefix, task=task, rows=splits.indices("train")
            ),
            "feature_names": pair_feature_records(features),
            "feature_sources": feature_source_hashes(features),
            "outcomes": {
                item.name: file_provenance(item.outcome_path) for item in features.catalog.endpoints
            },
            "valid_labels": targets.label_mask.sum(axis=0).tolist(),
            "missing_or_ineligible_labels": (~targets.label_mask).sum(axis=0).tolist(),
            "files": {key: local_record(path) for key, path in files.items()},
        },
    )


def check_prepared_targets(
    root: Path,
    prefix: str,
    features: ModuleFeatures,
    splits: DatasetSplits,
    task: TaskType,
) -> dict[str, Any]:
    manifest = json.loads((root / f"{prefix}_module_targets_manifest.json").read_text(encoding="utf-8"))
    expected = (
        {"y", "y_mask"} if task is TaskType.PREVALENT else {"event", "time", "signed_time", "label_mask"}
    )
    checks = (
        manifest.get("schema_version") == 1,
        manifest.get("task") == task.value,
        manifest.get("eids_sha256") == array_sha256(features.participant_ids),
        manifest.get("modules") == [item.name for item in features.catalog.modules],
        manifest.get("endpoints") == [item.name for item in features.catalog.endpoints],
        manifest.get("split_sha256") == splits.canonical_sha256,
        manifest.get("feature_names") == pair_feature_records(features),
        manifest.get("feature_sources") == feature_source_hashes(features),
        set(manifest.get("files", {})) == expected,
    )
    if not all(checks):
        raise ValueError("prepared module target contract mismatch")
    for name in expected:
        record = manifest["files"][name]
        if record["file"] != f"{prefix}_{name}.npy":
            raise ValueError("prepared target filename mismatch")
        checked_file(root, record)
    return manifest


def train_modules(config: Mapping[str, Any]) -> Path:
    features, root, prefix, task = module_context(config)
    splits = module_splits(config, features.participant_ids)
    prepared = check_prepared_targets(root, prefix, features, splits, task)
    targets = load_training_targets(
        root,
        prefix=prefix,
        task=task,
        expected=(features.participant_ids.size, len(features.catalog.endpoints)),
    )
    rows = splits.indices("train")
    train_target_fingerprints = target_row_fingerprints(root, prefix=prefix, task=task, rows=rows)
    if (
        prepared.get("train_target_fingerprints") is not None
        and prepared["train_target_fingerprints"] != train_target_fingerprints
    ):
        raise ValueError("prepared module train target fingerprint mismatch")
    values = dict(section(section(config, "training"), "modules"))
    fold_values = values.pop("fold_table", {})
    if not isinstance(fold_values, Mapping):
        raise TypeError("training.modules.fold_table must be a mapping")
    if set(fold_values).difference(
        {"path", "row_index_column", "participant_id_column", "eid_column", "fold_column"}
    ):
        raise ValueError("unknown fold_table settings")
    if fold_values and not fold_values.get("path"):
        raise ValueError("configured fold_table.path must not be empty")
    settings = ModuleTrainingSettings.from_mapping(values, task=task)
    effective_settings = asdict(settings)
    training_metadata = {
        "command": "train-modules",
        "package_version": __version__,
        "random_seed": settings.seed,
    }
    fold_source = file_provenance(str(fold_values["path"])) if fold_values.get("path") else None
    folds = build_outer_folds(
        features.participant_ids,
        rows,
        seed=settings.seed,
        path=fold_values.get("path"),
        row_column=str(fold_values.get("row_index_column", "row_index")),
        eid_column=id_column_from_config(fold_values),
        fold_column=str(fold_values.get("fold_column", "fold")),
    )
    artifacts = section(config, "artifacts")
    if not artifacts.get("module_model_dir"):
        raise ValueError("artifacts.module_model_dir is required")
    bank = Path(str(artifacts["module_model_dir"])).expanduser().resolve()
    bank.mkdir(parents=True, exist_ok=True)
    modules = tuple(item.name for item in features.catalog.modules)
    endpoints = tuple(item.name for item in features.catalog.endpoints)
    scores = np.full((rows.size, len(modules), len(endpoints)), np.nan, dtype=np.float32)
    mask = np.zeros(scores.shape, dtype=bool)
    records = []
    for pair in features.catalog.pairs:
        key = (pair.module, pair.endpoint)
        m, d = modules.index(pair.module), endpoints.index(pair.endpoint)
        result = crossfit_pair(
            task=task,
            module=pair.module,
            endpoint=pair.endpoint,
            feature_names=features.feature_names[key],
            matrix=features.matrices[key][rows],
            labels=targets.labels[rows, d],
            label_mask=targets.label_mask[rows, d],
            signed_time=None if targets.signed_time is None else targets.signed_time[rows, d],
            row_indices=rows,
            folds=folds,
            settings=settings,
        )
        if result.status == "trained":
            expected = np.isfinite(features.matrices[key][rows]).any(axis=1)
            if not np.array_equal(result.oof_mask, expected):
                raise ValueError(f"OOF prediction coverage mismatch for {pair.module}/{pair.endpoint}")
        scores[:, m, d] = result.oof_scores
        mask[:, m, d] = result.oof_mask
        record = {
            "module": pair.module,
            "endpoint": pair.endpoint,
            "status": result.status,
            "metadata": dict(result.metadata),
            "score_scale": "probability" if task is TaskType.PREVALENT else "log_risk",
        }
        if result.model is not None:
            path = bank / pair_filename(*key)
            artifact = TraceModelArtifact(
                component="module",
                task=task,
                model=result.model,
                modules=(pair.module,),
                endpoints=(pair.endpoint,),
                training={
                    **training_metadata,
                    "settings": effective_settings,
                    "train_rows_used": result.metadata["eligible_train_count"],
                    "validation_rows_used": 0,
                    "test_rows_used": 0,
                },
                provenance={
                    "split_table_canonical_sha256": splits.canonical_sha256,
                    "folds_sha256": array_sha256(folds),
                    "fold_source": fold_source,
                    "inputs": dict(features.provenance),
                    "prepared_targets": prepared["files"],
                    "train_target_fingerprints": train_target_fingerprints,
                },
            )
            dump_trace_artifact(artifact, path)
            record["artifact"] = local_record(path)
        records.append(record)
    cache = write_oof_cache(
        bank / "train_oof.npz",
        train_rows=rows,
        train_eids=features.participant_ids[rows],
        folds=folds,
        scores=scores,
        pred_mask=mask,
    )
    return write_training_manifest(
        bank / "module_bank_manifest.json",
        {
            "schema_version": 1,
            "framework": framework_metadata(),
            "task": task.value,
            "modules": list(modules),
            "endpoints": list(endpoints),
            "feature_names": pair_feature_records(features),
            "feature_sources": feature_source_hashes(features),
            "eids_sha256": array_sha256(features.participant_ids),
            "split_sha256": splits.canonical_sha256,
            "train_target_fingerprints": train_target_fingerprints,
            "folds_sha256": array_sha256(folds),
            "fold_source": fold_source,
            "settings": effective_settings,
            "pairs": records,
            "oof_cache": local_record(cache),
            "training": {
                **training_metadata,
                "train_rows": int(rows.size),
                "validation_rows_used": 0,
                "test_rows_used": 0,
            },
        },
    )


def score_modules(config: Mapping[str, Any]) -> Path:
    """Score without labels; use the exact train OOF cache in training mode."""
    features, root, prefix, task = module_context(config)
    mode = section(config, "module_data").get("score_mode")
    if mode not in {"training", "inference"}:
        raise ValueError("module_data.score_mode must be training or inference")
    artifacts = section(config, "artifacts")
    if not artifacts.get("module_model_dir"):
        raise ValueError("artifacts.module_model_dir is required")
    bank = Path(str(artifacts["module_model_dir"])).expanduser().resolve()
    manifest, models = load_module_bank(bank, task=task)
    if manifest["modules"] != [item.name for item in features.catalog.modules]:
        raise ValueError("module bank module axis mismatch")
    if manifest["endpoints"] != [item.name for item in features.catalog.endpoints]:
        raise ValueError("module bank endpoint axis mismatch")
    if "feature_names" in manifest:
        if manifest["feature_names"] != pair_feature_records(features):
            raise ValueError("module bank selected feature mismatch")
    elif mode == "inference":
        # Inference exports retain saved feature order without claiming train
        # splits or OOF identity. Missing fitted models may have no features.
        records = {(str(record["module"]), str(record["endpoint"])): record for record in manifest["pairs"]}
        configured = set(features.feature_names)
        if not set(models).issubset(configured) or not configured.issubset(records):
            raise ValueError("module bank configured pair coverage mismatch")
        for key in configured:
            declared = records[key].get("selected_features")
            if declared is not None and tuple(declared) != features.feature_names[key]:
                raise ValueError("module bank selected feature mismatch")
        for key, model in models.items():
            if model.feature_names != features.feature_names[key]:
                raise ValueError("module bank selected feature mismatch")
    else:
        raise ValueError("module bank selected feature records are required for training")
    oof_identity: dict[str, Any] = {}
    if mode == "training":
        splits = module_splits(config, features.participant_ids)
        if manifest["eids_sha256"] != array_sha256(features.participant_ids):
            raise ValueError("module bank training participant mismatch")
        if manifest["split_sha256"] != splits.canonical_sha256:
            raise ValueError("module bank split mismatch")
        if manifest["feature_sources"] != feature_source_hashes(features):
            raise ValueError("module bank training feature sources mismatch")
        rows = splits.indices("train")
        cache = load_oof_cache(bank, manifest)
        validate_oof_identity(cache, rows, features.participant_ids)
        if manifest["folds_sha256"] != array_sha256(cache["folds"]):
            raise ValueError("OOF fold fingerprint mismatch")
        for key in models:
            m, d = manifest["modules"].index(key[0]), manifest["endpoints"].index(key[1])
            expected = np.isfinite(features.matrices[key][rows]).any(axis=1)
            if not np.array_equal(cache["pred_mask"][:, m, d], expected):
                raise ValueError(f"OOF prediction coverage mismatch for {key[0]}/{key[1]}")
        other_rows = np.concatenate([splits.indices("validation"), splits.indices("test")])
        batch = score_module_features(features, models, task=task, predict_rows=other_rows)
        batch = apply_train_oof(batch, cache, rows)
        oof_identity = {
            "split_sha256": splits.canonical_sha256,
            "train_rows_sha256": array_sha256(rows),
            "train_eids_sha256": array_sha256(batch.participant_ids[rows]),
            "module_training_target_identity": (
                "verified" if "train_target_fingerprints" in manifest else "legacy_unverified"
            ),
        }
        if "train_target_fingerprints" in manifest:
            oof_identity["train_target_fingerprints"] = manifest["train_target_fingerprints"]
    else:
        batch = score_module_features(features, models, task=task)
    return write_score_bundle(
        root,
        prefix=prefix,
        batch=batch,
        provenance={
            "score_mode": mode,
            **oof_identity,
            "module_bank": local_record(bank / "module_bank_manifest.json"),
            "features": dict(features.provenance),
            "score_scale": "probability" if task is TaskType.PREVALENT else "log_risk",
        },
    )
