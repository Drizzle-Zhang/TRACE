from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from trace_agent.cli.longitudinal_evaluation import evaluate_horizon_predictions, horizon_years
from trace_agent.contracts.scores import ModuleScoreBatch
from trace_agent.contracts.task import SplitRole, TaskType
from trace_agent.data.arrays import (
    load_module_score_bundle,
    select_participant,
)
from trace_agent.data.split_table import canonical_eids_equal
from trace_agent.identity import framework_metadata
from trace_agent.runtime.artifacts import build_trace_agent, load_evidence_configuration
from trace_agent.runtime.serialization import agent_result_to_dict
from trace_agent.training.manifests import array_sha256, split_provenance
from trace_agent.training.service import dataset_from_config


@dataclass(frozen=True)
class InferenceSummary:
    manifest_path: Path
    participants: int
    consistency_participants: int


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"config.{key} must be a mapping")
    return value


def _output_directory(config: Mapping[str, Any]) -> Path:
    output = _mapping(config, "output")
    configured = output.get("directory")
    if not configured:
        raise ValueError("missing required config: output.directory")
    path = Path(str(configured)).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_batch(config: Mapping[str, Any]) -> ModuleScoreBatch:
    data = _mapping(config, "data")
    input_dir = data.get("input_dir")
    prefix = data.get("prefix")
    if not input_dir or not prefix:
        raise ValueError("data.input_dir and data.prefix are required")
    batch = load_module_score_bundle(
        input_dir,
        prefix=str(prefix),
        task=TaskType.parse(config.get("task", "prevalent")),
        allow_legacy_pickle_names=bool(data.get("allow_legacy_pickle_names", False)),
    )
    if batch.n_participants == 0:
        raise ValueError("the input score bundle contains no participants")
    return batch


def _initial_masks(config: Mapping[str, Any], batch: ModuleScoreBatch) -> np.ndarray:
    planner = _mapping(config, "planner")
    configured_mask = planner.get("initial_mask_path")
    if configured_mask:
        masks = np.asarray(np.load(configured_mask, allow_pickle=False), dtype=bool)
        if masks.shape != batch.availability.shape:
            raise ValueError(
                "planner.initial_mask_path shape must be " f"{batch.availability.shape}, got {masks.shape}"
            )
    else:
        configured_modules = tuple(str(value) for value in planner.get("initial_modules", []))
        unknown = sorted(set(configured_modules).difference(batch.modules))
        if unknown:
            raise ValueError(f"planner.initial_modules contains unknown modules: {unknown}")
        selected = np.asarray(
            [module in configured_modules for module in batch.modules],
            dtype=bool,
        )
        masks = np.broadcast_to(selected, batch.availability.shape).copy()
    if np.any(masks & ~batch.availability):
        raise ValueError("an initial module is unavailable for at least one participant")
    return masks


def prepare_data(config: Mapping[str, Any]) -> Path:
    mode = _mapping(config, "module_data").get("input_mode", "scores")
    if mode == "modules":
        from trace_agent.training.modules.service import prepare_module_data

        return prepare_module_data(config)
    if mode != "scores":
        raise ValueError("module_data.input_mode must be modules or scores")
    data = _mapping(config, "data")
    split = data.get("split_table")
    dataset = None
    if split is not None:
        if not isinstance(split, Mapping):
            raise ValueError("config.data.split_table must be a mapping")
        if split.get("path"):
            # Preparation also precedes frozen-model evaluation. Only fitting
            # commands require OOF scores; identity/target checks still apply.
            dataset = dataset_from_config(config, require_training_scores=False)
    batch = dataset.scores if dataset is not None else _load_batch(config)
    output = _output_directory(config)
    manifest = {
        "framework": framework_metadata(),
        "task": batch.task.value,
        "participants": batch.n_participants,
        "modules": list(batch.modules),
        "endpoints": list(batch.endpoints),
        "score_shape": list(batch.scores.shape),
        "valid_score_fraction": float(np.mean(batch.pred_mask)),
        "available_module_fraction": float(np.mean(batch.availability)),
        "source": dict(batch.metadata),
        "config_sha256": config.get("_meta", {}).get("sha256"),
    }
    if dataset is not None:
        manifest["split_provenance"] = split_provenance(dataset)
    path = output / "prepared_data_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def run_inference(config: Mapping[str, Any]) -> InferenceSummary:
    batch = _load_batch(config)
    initial_masks = _initial_masks(config, batch)
    agent = build_trace_agent(config, batch=batch)
    output = _output_directory(config)
    output_values = _mapping(config, "output")
    top_endpoints = int(output_values.get("top_endpoints", 0))
    result_path = output / "agent_results.jsonl"
    final_scores = np.lib.format.open_memmap(
        output / "final_scores.npy",
        mode="w+",
        dtype=np.float32,
        shape=(batch.n_participants, len(batch.endpoints)),
    )
    final_valid = np.lib.format.open_memmap(
        output / "final_valid_mask.npy", mode="w+", dtype=bool, shape=final_scores.shape
    )
    final_used = np.lib.format.open_memmap(
        output / "final_used_modules.npy",
        mode="w+",
        dtype=bool,
        shape=batch.availability.shape,
    )
    accumulated_burden = np.lib.format.open_memmap(
        output / "accumulated_burden.npy",
        mode="w+",
        dtype=np.float32,
        shape=(batch.n_participants,),
    )
    consistency_enabled = bool(_mapping(config, "consistency").get("enabled", False))
    consistency_combined = (
        np.lib.format.open_memmap(
            output / "consistency_combined.npy",
            mode="w+",
            dtype=np.float32,
            shape=(batch.n_participants, len(batch.endpoints)),
        )
        if consistency_enabled
        else None
    )
    consistency_participants = 0
    artifact_provenance: Mapping[str, object] = {}
    horizon_buffers: dict[str, np.ndarray] = {}
    consistency_buffers: dict[str, np.ndarray] = {}
    prediction_csv_paths: dict[str, Path] = {}
    prediction_csv_writers: dict[str, Any] = {}
    horizon_valid: np.ndarray | None = None
    horizon_path = output / "horizon_risks.npz"
    # Never leave an earlier run's probability artifact available as a fallback.
    horizon_path.unlink(missing_ok=True)
    (output / "predictions.csv").unlink(missing_ok=True)
    for previous_csv in output.glob("predictions_*y.csv"):
        try:
            horizon_years(previous_csv.stem.removeprefix("predictions_"))
        except ValueError:
            continue
        previous_csv.unlink()
    (output / "consistency_scores.npz").unlink(missing_ok=True)
    classification_paths = {output / "consistency_predictions.npz"}
    configured_predictions = _mapping(config, "consistency").get("predictions_path")
    if configured_predictions:
        classification_paths.add(Path(configured_predictions))
    for previous_classification in classification_paths:
        previous_classification.unlink(missing_ok=True)
        for kind in ("baseline", "filtered"):
            previous_classification.with_name(f"{previous_classification.stem}_{kind}.csv").unlink(
                missing_ok=True
            )
    with (
        TemporaryDirectory(prefix=".horizon-", dir=output) as temporary,
        result_path.open("w", encoding="utf-8") as handle,
        ExitStack() as csv_handles,
    ):
        for index in range(batch.n_participants):
            participant_batch = select_participant(batch, index)
            result = agent.assess(
                participant_batch,
                initial_used_modules=initial_masks[index],
            )
            if not all(result.validation_checks.values()):
                failures = [name for name, passed in result.validation_checks.items() if not passed]
                raise RuntimeError(
                    f"participant {result.participant_id} failed output " f"validation: {failures}"
                )
            handle.write(
                json.dumps(
                    agent_result_to_dict(result, top_endpoints=top_endpoints),
                    ensure_ascii=False,
                )
                + "\n"
            )
            profile = result.planning.final_profile
            final_valid[index] = profile.valid_mask[0]
            final_scores[index] = np.where(profile.valid_mask[0], profile.decision_scores[0], np.nan)
            if index == 0:
                csv_names = ("",) if batch.task is TaskType.PREVALENT else tuple(profile.horizon_risks)
                for name in csv_names:
                    if batch.task is TaskType.LONGITUDINAL:
                        horizon_years(name)
                    path = output / (f"predictions_{name}.csv" if name else "predictions.csv")
                    prediction_writer = csv.writer(
                        csv_handles.enter_context(path.open("w", encoding="utf-8", newline=""))
                    )
                    prediction_writer.writerow(["participant_id", *batch.endpoints])
                    prediction_csv_paths[name] = path
                    prediction_csv_writers[name] = prediction_writer
            if batch.task is TaskType.PREVALENT:
                prediction_csv_writers[""].writerow(
                    [str(batch.participant_ids[index])]
                    + [
                        float(value) if valid else ""
                        for value, valid in zip(final_scores[index], final_valid[index])
                    ]
                )
            if batch.task is TaskType.LONGITUDINAL:
                if index == 0:
                    for position, name in enumerate(profile.horizon_risks):
                        horizon_buffers[name] = np.lib.format.open_memmap(
                            Path(temporary) / f"risk_{position}.npy",
                            mode="w+",
                            dtype=np.float32,
                            shape=final_scores.shape,
                        )
                    horizon_valid = np.lib.format.open_memmap(
                        Path(temporary) / "valid.npy",
                        mode="w+",
                        dtype=bool,
                        shape=final_scores.shape,
                    )
                if set(profile.horizon_risks) != set(horizon_buffers):
                    raise ValueError("absolute-risk horizons changed between participants")
                assert horizon_valid is not None
                horizon_valid[index] = profile.valid_mask[0]
                for name, buffer in horizon_buffers.items():
                    values = profile.horizon_risks[name][0]
                    if np.any(profile.valid_mask[0] & (~np.isfinite(values) | (values < 0) | (values > 1))):
                        raise ValueError(f"horizon {name} valid probabilities must be finite and in [0, 1]")
                    buffer[index] = values
                    prediction_csv_writers[name].writerow(
                        [str(batch.participant_ids[index])]
                        + [
                            float(value) if valid else ""
                            for value, valid in zip(buffer[index], final_valid[index])
                        ]
                    )
            final_used[index] = result.planning.final_used_modules
            accumulated_burden[index] = result.planning.accumulated_burden
            if index == 0:
                provenance_value = result.metadata.get("artifact_provenance", {})
                if isinstance(provenance_value, Mapping):
                    artifact_provenance = provenance_value
            if result.consistency is not None:
                consistency_participants += 1
                for name, values in (
                    ("kg", result.consistency.kg_scores),
                    ("feature", result.consistency.feature_scores),
                    ("evidence", result.consistency.evidence_scores),
                ):
                    if name not in consistency_buffers:
                        consistency_buffers[name] = np.lib.format.open_memmap(
                            Path(temporary) / f"{name}.npy",
                            mode="w+",
                            dtype=np.float32,
                            shape=final_scores.shape,
                        )
                    consistency_buffers[name][index] = values[0]
                if consistency_combined is not None:
                    consistency_combined[index] = result.consistency.combined_scores[0]
            elif consistency_enabled:
                raise RuntimeError("consistency was enabled but no result was produced")
        if horizon_buffers:
            assert horizon_valid is not None
            horizon_payload: dict[str, Any] = {
                "participant_ids": np.asarray(batch.participant_ids, dtype=np.str_),
                "endpoints": np.asarray(batch.endpoints, dtype=np.str_),
                "valid_mask": horizon_valid,
                **horizon_buffers,
            }
            np.savez_compressed(horizon_path, **horizon_payload)
        if consistency_buffers:
            np.savez_compressed(
                output / "consistency_scores.npz",
                participant_ids=np.asarray(batch.participant_ids, dtype=np.str_),
                endpoints=np.asarray(batch.endpoints, dtype=np.str_),
                **consistency_buffers,
            )
    final_scores.flush()
    final_valid.flush()
    final_used.flush()
    accumulated_burden.flush()
    if consistency_combined is not None:
        consistency_combined.flush()
    np.save(
        output / "participant_ids.npy",
        np.asarray(batch.participant_ids),
    )
    np.save(output / "prediction_endpoints.npy", np.asarray(batch.endpoints, dtype=np.str_))
    manifest = {
        "framework": framework_metadata(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config.get("_meta", {}).get("sha256"),
        "config_path": config.get("_meta", {}).get("config_path"),
        "task": batch.task.value,
        "participants": batch.n_participants,
        "modules": list(batch.modules),
        "endpoints": list(batch.endpoints),
        "artifact_provenance": artifact_provenance,
        "outputs": {
            "results": str(result_path),
            "final_scores": str(output / "final_scores.npy"),
            "final_valid_mask": str(output / "final_valid_mask.npy"),
            "prediction_endpoints": str(output / "prediction_endpoints.npy"),
            "final_used_modules": str(output / "final_used_modules.npy"),
            "accumulated_burden": str(output / "accumulated_burden.npy"),
        },
    }
    if batch.task is TaskType.PREVALENT:
        manifest["outputs"]["predictions_csv"] = str(prediction_csv_paths[""])
    elif prediction_csv_paths:
        manifest["outputs"]["horizon_predictions_csv"] = {
            name: str(path) for name, path in prediction_csv_paths.items()
        }
    if consistency_combined is not None:
        manifest["outputs"]["consistency_combined"] = str(output / "consistency_combined.npy")
        manifest["outputs"]["consistency_scores"] = str(output / "consistency_scores.npz")
    if horizon_buffers:
        manifest["outputs"]["horizon_risks"] = str(horizon_path)
    consistency_settings = _mapping(config, "consistency")
    threshold_checks_path = output / "threshold_checks.csv"
    threshold_summary_path = output / "threshold_summary.json"
    threshold_checks_path.unlink(missing_ok=True)
    threshold_summary_path.unlink(missing_ok=True)
    if (
        consistency_enabled
        and consistency_settings.get("evidence_bundle")
        and bool(consistency_settings.get("enable_threshold_checks", batch.task is TaskType.PREVALENT))
    ):
        store, features, code_to_name = load_evidence_configuration(
            consistency_settings,
            allow_pickle_names=bool(_mapping(config, "data").get("allow_legacy_pickle_names", False)),
        )
        feature_values_path = consistency_settings.get("feature_values_path")
        values = (
            None
            if not feature_values_path
            else np.load(feature_values_path, allow_pickle=False, mmap_mode="r")
        )
        if values is not None and values.shape != (batch.n_participants, len(features)):
            raise ValueError("threshold feature_values must align with participant and feature axes")
        threshold_summary: dict[str, Any] = {
            "n_rows": 0,
            "n_samples": batch.n_participants,
            "status_counts": {},
        }
        with threshold_checks_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_id",
                    "sample_index",
                    "disease",
                    "feature_code",
                    "feature_name",
                    "factor",
                    "factor_token",
                    "threshold",
                    "threshold_status",
                    "observed_value",
                    "source",
                    "snippet",
                ],
            )
            writer.writeheader()
            for start in range(0, batch.n_participants, 1024):
                stop = min(start + 1024, batch.n_participants)
                checks, chunk_summary = store.check_thresholds(
                    endpoints=batch.endpoints,
                    features=features,
                    participant_ids=batch.participant_ids[start:stop],
                    feature_values=None if values is None else values[start:stop],
                    feature_code_to_name=code_to_name,
                    min_certainty_rank=float(consistency_settings.get("min_certainty_rank", 0.0)),
                    min_strength_rank=float(consistency_settings.get("min_strength_rank", 0.0)),
                )
                for row in checks:
                    row["sample_index"] = int(row["sample_index"]) + start
                writer.writerows(checks)
                threshold_summary["n_rows"] += int(chunk_summary["n_rows"])
                for status, count in chunk_summary["status_counts"].items():
                    counts = threshold_summary["status_counts"]
                    counts[status] = counts.get(status, 0) + count
        threshold_summary_path.write_text(
            json.dumps(threshold_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest["outputs"]["threshold_checks"] = str(threshold_checks_path)
        manifest["outputs"]["threshold_summary"] = str(threshold_summary_path)
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return InferenceSummary(
        manifest_path=manifest_path,
        participants=batch.n_participants,
        consistency_participants=consistency_participants,
    )


def run_consistency(config: Mapping[str, Any]) -> Path:
    consistency = _mapping(config, "consistency")
    if not bool(consistency.get("enabled", False)):
        raise ValueError("consistency.enabled must be true for this command")
    summary = run_inference(config)
    if summary.consistency_participants != summary.participants:
        raise RuntimeError("consistency outputs were not produced")
    return summary.manifest_path


def evaluate(config: Mapping[str, Any]) -> Path:
    from trace_agent.cli.consistency_workflow import evaluate_frozen_consistency

    dataset = dataset_from_config(config, require_training_scores=False)
    test = dataset.view(SplitRole.TEST)
    evaluation = _mapping(config, "evaluation")
    if dataset.scores.task is TaskType.LONGITUDINAL:
        horizon_path = evaluation.get("horizon_predictions_path") or (
            _output_directory(config) / "horizon_risks.npz"
        )
        report = evaluate_horizon_predictions(dataset, horizon_path)
        frozen_report = evaluate_frozen_consistency(config, dataset)
        if frozen_report is not None:
            report["consistency"] = frozen_report
        path = _output_directory(config) / "evaluation.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path
    output = _output_directory(config)
    predictions_path = Path(str(evaluation.get("predictions") or (output / "final_scores.npy")))
    prediction_directory = predictions_path.parent
    participant_ids_path = evaluation.get("participant_ids") or (prediction_directory / "participant_ids.npy")
    endpoint_path = evaluation.get("endpoints_file") or (prediction_directory / "prediction_endpoints.npy")
    valid_path = evaluation.get("valid_mask_path") or (prediction_directory / "final_valid_mask.npy")
    predictions = np.asarray(
        np.load(predictions_path, allow_pickle=False),
        dtype=np.float64,
    )
    prediction_eids = np.load(participant_ids_path, allow_pickle=False)
    expected_shape = (
        dataset.scores.n_participants,
        len(dataset.scores.endpoints),
    )
    if predictions.shape != expected_shape:
        raise ValueError(
            "evaluation predictions must align with the complete score bundle; "
            f"expected {expected_shape}, got {predictions.shape}"
        )
    if prediction_eids.shape != (expected_shape[0],) or not canonical_eids_equal(
        prediction_eids, dataset.scores.participant_ids
    ):
        raise ValueError("evaluation participant IDs do not match the score bundle order")
    endpoints = np.load(endpoint_path, allow_pickle=False)
    if endpoints.shape != (expected_shape[1],) or tuple(endpoints.tolist()) != dataset.scores.endpoints:
        raise ValueError("evaluation prediction endpoint order differs from the score bundle")
    prediction_valid = np.load(valid_path, allow_pickle=False)
    if prediction_valid.shape != expected_shape or prediction_valid.dtype.kind != "b":
        raise ValueError(
            "evaluation prediction valid_mask must be a boolean participant-by-endpoint array "
            f"with shape {expected_shape}"
        )
    if np.any(prediction_valid & (~np.isfinite(predictions) | (predictions < 0) | (predictions > 1))):
        raise ValueError("evaluation valid probabilities must be finite and in [0, 1]")
    predictions = np.where(prediction_valid[test.row_indices], predictions[test.row_indices], np.nan)
    labels = np.asarray(test.targets.labels, dtype=np.float64)
    mask = np.asarray(test.targets.label_mask, dtype=bool)
    valid = mask & np.isfinite(predictions) & np.isfinite(labels)
    clipped = np.clip(predictions, 1e-7, 1.0 - 1e-7)
    loss = -(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    endpoint_metrics: list[dict[str, Any]] = []
    aucs: list[float] = []
    for endpoint in range(predictions.shape[1]):
        endpoint_valid = valid[:, endpoint]
        auc = None
        if np.any(endpoint_valid) and np.unique(labels[endpoint_valid, endpoint]).size == 2:
            auc = float(
                roc_auc_score(
                    labels[endpoint_valid, endpoint],
                    predictions[endpoint_valid, endpoint],
                )
            )
            aucs.append(auc)
        endpoint_metrics.append(
            {
                "endpoint_index": endpoint,
                "n": int(np.sum(endpoint_valid)),
                "auroc": auc,
                "log_loss": (
                    float(np.mean(loss[endpoint_valid, endpoint])) if np.any(endpoint_valid) else None
                ),
            }
        )
    report = {
        "split": SplitRole.TEST.value,
        "participants": test.scores.n_participants,
        "row_indices_sha256": array_sha256(test.row_indices),
        "eids_sha256": array_sha256(dataset.splits.eids(SplitRole.TEST)),
        "n_valid": int(np.sum(valid)),
        "masked_log_loss": (float(np.mean(loss[valid])) if np.any(valid) else None),
        "macro_auroc": (float(np.mean(aucs)) if aucs else None),
        "endpoints": endpoint_metrics,
    }
    frozen_report = evaluate_frozen_consistency(config, dataset)
    if frozen_report is not None:
        report["consistency"] = frozen_report
    path = _output_directory(config) / "evaluation.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path
