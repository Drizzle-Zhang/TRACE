"""Validation-only rule fitting, label-free application and test-only reporting."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from trace_agent.cli.longitudinal_evaluation import horizon_years
from trace_agent.consistency.frozen import (
    apply_frozen_rules,
    classification_counts,
    load_frozen_rules,
    select_frozen_rules,
    write_frozen_rules,
)
from trace_agent.consistency.survival_policies import (
    apply_survival_policies,
    evaluate_survival_policies,
    load_survival_policies,
    select_survival_policies,
    write_survival_policies,
)
from trace_agent.contracts.task import SplitRole, TaskType
from trace_agent.data.split_table import canonical_eids_equal, normalize_eids
from trace_agent.data.training_dataset import TrainingDataset
from trace_agent.planning.rewards.longitudinal import build_horizon_labels
from trace_agent.runtime.provenance import array_sha256, file_provenance
from trace_agent.training.service import dataset_from_config


def _settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    settings = config.get("consistency", {})
    if not isinstance(settings, Mapping):
        raise TypeError("config.consistency must be a mapping")
    return settings


def _output(config: Mapping[str, Any]) -> Path:
    output = config.get("output", {})
    if not isinstance(output, Mapping) or not output.get("directory"):
        raise ValueError("output.directory is required")
    return Path(str(output["directory"])).expanduser().resolve()


def _path(config: Mapping[str, Any], key: str, default: str) -> Path:
    return Path(str(_settings(config).get(key) or (_output(config) / default))).expanduser().resolve()


def _horizon(config: Mapping[str, Any], task: TaskType) -> str | None:
    name = _settings(config).get("horizon")
    if task is TaskType.PREVALENT:
        if name is not None:
            raise ValueError("consistency.horizon is only valid for longitudinal tasks")
        return None
    if name is None:
        raise ValueError("longitudinal consistency requires consistency.horizon, e.g. 5y")
    if not isinstance(name, str):
        raise TypeError("consistency.horizon must be a year-based string, e.g. 5y")
    horizon_years(name)
    return name


def _axes(eids: np.ndarray, endpoints: np.ndarray, expected_eids: np.ndarray, names: tuple[str, ...]) -> None:
    if eids.shape != expected_eids.shape or not canonical_eids_equal(eids, expected_eids):
        raise ValueError("consistency prediction participant IDs do not match inference order")
    if endpoints.shape != (len(names),) or tuple(endpoints.tolist()) != names:
        raise ValueError("consistency prediction endpoint order mismatch")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_inference(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load only frozen inference outputs; never inspect a target or split table."""
    root = _output(config)
    task = TaskType.parse(config.get("task", "prevalent"))
    horizon = _horizon(config, task)
    ids_path = root / "participant_ids.npy"
    names_path = root / "prediction_endpoints.npy"
    mask_path = root / "final_valid_mask.npy"
    scores_path = root / "consistency_scores.npz"
    manifest_path = root / "run_manifest.json"
    eids = np.load(ids_path, allow_pickle=False)
    endpoint_values = np.load(names_path, allow_pickle=False)
    if eids.ndim != 1 or endpoint_values.ndim != 1:
        raise ValueError("prediction participant IDs and endpoints must be one-dimensional")
    endpoints = tuple(str(value) for value in endpoint_values.tolist())
    if len(set(endpoints)) != len(endpoints) or len(set(normalize_eids(eids))) != eids.size:
        raise ValueError("prediction participant IDs and endpoints must be unique")
    expected = (eids.size, len(endpoints))
    valid = np.load(mask_path, allow_pickle=False)
    if valid.shape != expected or valid.dtype.kind != "b":
        raise ValueError("final_valid_mask must be a boolean participant-by-endpoint array")
    prediction_path = root / ("final_scores.npy" if horizon is None else "horizon_risks.npz")
    if horizon is None:
        probability = np.load(prediction_path, allow_pickle=False)
    else:
        with np.load(prediction_path, allow_pickle=False) as archive:
            _axes(archive["participant_ids"], archive["endpoints"], eids, endpoints)
            if horizon not in archive.files:
                raise ValueError(f"inference does not contain consistency horizon {horizon}")
            horizon_valid = archive["valid_mask"]
            if horizon_valid.shape != expected or horizon_valid.dtype.kind != "b":
                raise ValueError("horizon valid_mask must be a boolean participant-by-endpoint array")
            valid = valid & horizon_valid
            probability = archive[horizon]
    probability = np.asarray(probability, dtype=np.float32)
    if probability.shape != expected:
        raise ValueError("consistency probabilities must align with the complete inference cohort")
    if np.any(valid & (~np.isfinite(probability) | (probability < 0) | (probability > 1))):
        raise ValueError("valid consistency predictions must be finite probabilities in [0, 1]")
    with np.load(scores_path, allow_pickle=False) as archive:
        required = {"participant_ids", "endpoints", "kg", "feature", "evidence"}
        if not required.issubset(archive.files):
            raise ValueError(
                "consistency_scores requires participant IDs, endpoints, kg, feature and evidence"
            )
        _axes(archive["participant_ids"], archive["endpoints"], eids, endpoints)
        scores = {key: np.asarray(archive[key], dtype=np.float32) for key in ("kg", "feature", "evidence")}
        if any(value.shape != expected for value in scores.values()):
            raise ValueError("consistency score arrays do not match inference shape")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") != task.value or manifest.get("endpoints") != list(endpoints):
        raise ValueError("inference manifest task/endpoint identity mismatch")
    provenance = manifest.get("artifact_provenance", {})
    # Contributions are participant-specific; model/knowledge/scaling identities
    # must match when frozen rules are applied to another cohort.
    identity = {
        name: entry["sha256"]
        for name, entry in provenance.items()
        if name not in {"contributions", "feature_values_path"}
        and isinstance(entry, Mapping)
        and entry.get("sha256")
    }
    if not identity.get("fusion_model"):
        raise ValueError("inference manifest requires fusion model SHA-256 provenance")
    paths = (ids_path, names_path, mask_path, scores_path, prediction_path, manifest_path)
    source_files = {path.name: file_provenance(path)["sha256"] for path in paths}
    return {
        "task": task.value,
        "horizon": horizon,
        "participant_ids": eids,
        "endpoints": endpoints,
        "predictions": probability,
        "valid_mask": valid,
        **scores,
        "model_identity": identity,
        "source_files": source_files,
        "input_sha256": _fingerprint(source_files),
    }


def _dataset_alignment(inference: Mapping[str, Any], dataset: TrainingDataset) -> None:
    if dataset.scores.task.value != inference["task"]:
        raise ValueError("frozen consistency task does not match evaluation dataset")
    _axes(
        inference["participant_ids"],
        np.asarray(inference["endpoints"]),
        dataset.scores.participant_ids,
        dataset.scores.endpoints,
    )


def _targets(dataset: TrainingDataset, role: SplitRole, horizon: str | None) -> tuple[np.ndarray, np.ndarray]:
    targets = dataset.view(role).targets
    if horizon is None:
        return targets.labels, targets.label_mask
    if targets.event_time is None:
        raise ValueError("longitudinal consistency requires event times and censoring information")
    labels, known = build_horizon_labels(
        event=targets.labels,
        time=targets.event_time,
        label_mask=targets.label_mask,
        horizons=np.asarray([horizon_years(horizon)]),
    )
    return labels.reshape(targets.labels.shape), known.reshape(targets.label_mask.shape)


def _check_rules(artifact: Mapping[str, Any], inference: Mapping[str, Any]) -> None:
    if artifact["task"] != inference["task"] or artifact["horizon"] != inference["horizon"]:
        raise ValueError("frozen consistency task/horizon does not match inference")
    if artifact["endpoints"] != list(inference["endpoints"]):
        raise ValueError("frozen consistency endpoint order does not match inference")
    if artifact.get("model_identity") != inference["model_identity"]:
        raise ValueError("frozen consistency model/knowledge identity does not match inference")


def _is_survival_search(artifact: Mapping[str, Any]) -> bool:
    return artifact.get("schema") == "trace.survival_consistency_policies.v1"


def _load_rules(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return load_survival_policies(path) if _is_survival_search(payload) else load_frozen_rules(path)


def calibrate_consistency(config: Mapping[str, Any]) -> Path:
    inference = _read_inference(config)
    dataset = dataset_from_config(config, require_training_scores=False)
    _dataset_alignment(inference, dataset)
    rows = dataset.splits.indices(SplitRole.VALIDATION)
    labels, known = _targets(dataset, SplitRole.VALIDATION, inference["horizon"])
    settings = _settings(config)
    search = settings.get("longitudinal_strategy_search", "posthoc")
    if search not in {"posthoc", "product_filters"}:
        raise ValueError("consistency.longitudinal_strategy_search must be posthoc or product_filters")
    if inference["task"] == TaskType.LONGITUDINAL.value and search == "posthoc":
        artifact = select_survival_policies(
            predictions=inference["predictions"][rows],
            labels=labels,
            label_mask=known & inference["valid_mask"][rows],
            kg_scores=inference["kg"][rows],
            evidence_scores=inference["evidence"][rows],
            endpoints=inference["endpoints"],
            split_role=SplitRole.VALIDATION,
            horizon=inference["horizon"],
            threshold_points=int(settings.get("threshold_points", 41)),
            joint_grid_points=int(settings.get("joint_grid_points", 9)),
            max_finalists=int(settings.get("max_finalists", 120)),
        )
    else:
        artifact = select_frozen_rules(
            predictions=inference["predictions"][rows],
            labels=labels,
            label_mask=known & inference["valid_mask"][rows],
            kg_scores=inference["kg"][rows],
            feature_scores=inference["feature"][rows],
            evidence_scores=inference["evidence"][rows],
            endpoints=inference["endpoints"],
            split_role=SplitRole.VALIDATION,
            task=inference["task"],
            horizon=inference["horizon"],
            threshold_grid=settings.get("threshold_grid"),
            min_f1_delta=float(settings.get("min_f1_delta", 0.0)),
            min_validation_positives=int(settings.get("min_validation_positives", 1)),
            min_validation_base_positives=int(settings.get("min_validation_base_positives", 1)),
        )
    artifact["model_identity"] = inference["model_identity"]
    artifact["source"] = {
        "inference_directory": str(_output(config)),
        "input_sha256": inference["input_sha256"],
        "files": inference["source_files"],
        "split_sha256": dataset.splits.canonical_sha256,
        "validation_rows_sha256": array_sha256(rows),
        "validation_eids_sha256": array_sha256(normalize_eids(dataset.scores.participant_ids[rows])),
        "validation_labels_sha256": array_sha256(labels),
        "validation_known_sha256": array_sha256(known),
        "validation_participants": int(rows.size),
        "test_rows_used": 0,
    }
    writer = write_survival_policies if _is_survival_search(artifact) else write_frozen_rules
    return writer(_path(config, "rules_path", "consistency_rules.json"), artifact)


def apply_consistency(config: Mapping[str, Any]) -> Path:
    """Apply validation rules to inference arrays without loading targets or splits."""
    inference = _read_inference(config)
    rules_path = _path(config, "rules_path", "consistency_rules.json")
    artifact = _load_rules(rules_path)
    _check_rules(artifact, inference)
    if _is_survival_search(artifact):
        baseline = np.zeros_like(inference["valid_mask"])
        filtered = np.zeros_like(baseline)
        for start in range(0, baseline.shape[0], 4096):
            block = slice(start, start + 4096)
            baseline[block], filtered[block], _ = apply_survival_policies(
                artifact,
                predictions=inference["predictions"][block],
                valid_mask=inference["valid_mask"][block],
                kg_scores=inference["kg"][block],
                evidence_scores=inference["evidence"][block],
                include_strategies=False,
            )
    else:
        baseline, filtered = apply_frozen_rules(
            artifact,
            predictions=inference["predictions"],
            valid_mask=inference["valid_mask"],
            kg_scores=inference["kg"],
            feature_scores=inference["feature"],
            evidence_scores=inference["evidence"],
        )
    path = _path(config, "predictions_path", "consistency_predictions.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        participant_ids=inference["participant_ids"],
        endpoints=np.asarray(inference["endpoints"]),
        valid_mask=inference["valid_mask"],
        baseline=baseline,
        filtered=filtered,
        task=np.asarray(inference["task"]),
        horizon=np.asarray(inference["horizon"] or ""),
        rules_sha256=np.asarray(file_provenance(rules_path)["sha256"]),
        input_sha256=np.asarray(inference["input_sha256"]),
    )
    for name, values in (("baseline", baseline), ("filtered", filtered)):
        csv_path = path.with_name(f"{path.stem}_{name}.csv")
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["participant_id", *inference["endpoints"]])
            writer.writerows(
                [str(eid), *[int(value) if valid else "" for value, valid in zip(row, mask)]]
                for eid, row, mask in zip(inference["participant_ids"], values, inference["valid_mask"])
            )
    return path


def _report(
    prediction: np.ndarray, labels: np.ndarray, valid: np.ndarray, names: tuple[str, ...]
) -> dict[str, Any]:
    entries = [
        {
            "endpoint": name,
            "endpoint_index": i,
            "n": int(valid[:, i].sum()),
            **classification_counts(prediction[:, i], labels[:, i], valid[:, i]),
        }
        for i, name in enumerate(names)
    ]
    macro = {}
    for metric in ("precision", "recall", "f1"):
        values = [entry[metric] for entry in entries if entry[metric] is not None]
        macro[metric] = float(np.mean(values)) if values else None
    return {
        "n_valid": int(valid.sum()),
        "micro": classification_counts(prediction, labels, valid),
        "macro": macro,
        "endpoints": entries,
    }


def evaluate_frozen_consistency(config: Mapping[str, Any], dataset: TrainingDataset) -> dict[str, Any] | None:
    path = _path(config, "predictions_path", "consistency_predictions.npz")
    if not path.is_file() and not _settings(config).get("predictions_path"):
        return None
    inference = _read_inference(config)
    _dataset_alignment(inference, dataset)
    rules_path = _path(config, "rules_path", "consistency_rules.json")
    artifact = _load_rules(rules_path)
    _check_rules(artifact, inference)
    with np.load(path, allow_pickle=False) as archive:
        _axes(
            archive["participant_ids"],
            archive["endpoints"],
            inference["participant_ids"],
            inference["endpoints"],
        )
        if str(archive["task"].item()) != inference["task"] or str(archive["horizon"].item()) != (
            inference["horizon"] or ""
        ):
            raise ValueError("frozen consistency prediction task/horizon mismatch")
        if str(archive["rules_sha256"].item()) != file_provenance(rules_path)["sha256"]:
            raise ValueError(
                "stale consistency predictions: frozen rules have changed; rerun apply-consistency"
            )
        if str(archive["input_sha256"].item()) != inference["input_sha256"]:
            raise ValueError(
                "stale consistency predictions: inference inputs changed; rerun apply-consistency"
            )
        masks = {key: archive[key] for key in ("valid_mask", "baseline", "filtered")}
        if any(
            value.shape != inference["valid_mask"].shape or value.dtype.kind != "b"
            for value in masks.values()
        ):
            raise ValueError("frozen consistency masks must align with inference shape")
        if not np.array_equal(masks["valid_mask"], inference["valid_mask"]):
            raise ValueError("frozen consistency validity mask differs from inference")
        if (
            np.any(masks["filtered"] & ~masks["valid_mask"])
            or np.any(masks["baseline"] & ~masks["valid_mask"])
            or (not _is_survival_search(artifact) and np.any(masks["filtered"] & ~masks["baseline"]))
        ):
            raise ValueError("frozen consistency masks violate baseline/validity containment")
    rows = dataset.splits.indices(SplitRole.TEST)
    labels, known = _targets(dataset, SplitRole.TEST, inference["horizon"])
    valid = known & masks["valid_mask"][rows] & np.isfinite(labels)
    report = {
        "task": inference["task"],
        "horizon": inference["horizon"],
        "split": "test",
        "participants": int(rows.size),
        "row_indices_sha256": array_sha256(rows),
        "eids_sha256": array_sha256(normalize_eids(dataset.scores.participant_ids[rows])),
        "rules_path": str(rules_path),
        "predictions_path": str(path),
        "baseline": _report(masks["baseline"][rows], labels, valid, inference["endpoints"]),
        "filtered": _report(masks["filtered"][rows], labels, valid, inference["endpoints"]),
        "metric_method": (
            "known_outcomes_at_horizon (complete-case, not IPCW)"
            if inference["horizon"]
            else "observed_binary_labels"
        ),
    }
    if _is_survival_search(artifact):
        report["strategy_search"] = evaluate_survival_policies(
            artifact,
            predictions=inference["predictions"][rows],
            labels=labels,
            label_mask=valid,
            kg_scores=inference["kg"][rows],
            evidence_scores=inference["evidence"][rows],
        )
    return report
