from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from trace_agent.contracts.task import SplitRole, TaskType
from trace_agent.data.split_table import SplitColumns
from trace_agent.data.training_dataset import TrainingDataset, load_training_dataset
from trace_agent.planning.candidate_ranker import KnowledgeCandidateRanker
from trace_agent.runtime.artifacts import load_burden_profile, load_knowledge_matrix
from trace_agent.runtime.model_artifact import (
    TraceModelArtifact,
    dump_trace_artifact,
    load_trace_artifact,
    validate_upstream_artifacts,
)
from trace_agent.runtime.provenance import file_provenance
from trace_agent.training.longitudinal.integration import (
    LongitudinalIntegrationConfig,
    train_longitudinal_integration,
)
from trace_agent.training.longitudinal.planner import (
    LongitudinalReward,
    train_longitudinal_planner,
)
from trace_agent.training.manifests import (
    split_provenance,
    supervised_input_provenance,
    write_training_manifest,
)
from trace_agent.training.planner import PlannerTrainingConfig
from trace_agent.training.prevalent.calibration import (
    PrevalentCalibrationModel,
    fit_prevalent_calibration,
)
from trace_agent.training.prevalent.integration import (
    PrevalentIntegrationConfig,
    train_prevalent_integration,
)
from trace_agent.training.prevalent.planner import train_prevalent_planner


@dataclass(frozen=True)
class TrainingSummary:
    artifact_path: Path
    manifest_path: Path
    train_rows: int
    validation_rows: int
    test_rows_used: int = 0


def require_mapping(
    mapping: Mapping[str, Any],
    key: str,
    *,
    section: str = "config",
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{section}.{key} must be a mapping")
    return value


def required(mapping: Mapping[str, Any], key: str, *, section: str) -> Any:
    if key not in mapping or mapping[key] in (None, ""):
        raise ValueError(f"missing required config: {section}.{key}")
    return mapping[key]


def dataset_from_config(
    config: Mapping[str, Any], *, require_training_scores: bool = True
) -> TrainingDataset:
    data = require_mapping(config, "data")
    split = require_mapping(data, "split_table", section="data")
    columns = SplitColumns(
        row_index=str(split.get("row_index_column", "row_index")),
        eid=str(split.get("eid_column", "eid")),
        split=str(split.get("split_column", "split")),
    )
    return load_training_dataset(
        str(required(data, "input_dir", section="data")),
        prefix=str(required(data, "prefix", section="data")),
        task=TaskType.parse(config.get("task", "prevalent")),
        split_table=str(required(split, "path", section="data.split_table")),
        split_columns=columns,
        allow_legacy_pickle_names=False,
        require_training_scores=require_training_scores,
    )


def _training_mapping(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return require_mapping(config, "training")


def _artifact_path(config: Mapping[str, Any], component: str) -> Path:
    artifact_keys = {
        "integration": "fusion_model",
        "calibration": "calibrators",
        "planner": "q_model",
    }
    artifacts = require_mapping(config, "artifacts")
    key = artifact_keys[component]
    destination = Path(str(required(artifacts, key, section="artifacts"))).expanduser().resolve()
    if destination.is_dir():
        raise ValueError(f"artifacts.{key} must be a file path, got a directory")
    ancestor = destination.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise ValueError(f"artifacts.{key} parent must be a directory")
    return destination


def _source_files(dataset: TrainingDataset) -> dict[str, Any]:
    root = Path(str(dataset.scores.metadata["input_dir"]))
    prefix = str(dataset.scores.metadata["prefix"])
    score_suffix = "oof_probs" if dataset.scores.task is TaskType.PREVALENT else "scores"
    names = [
        "eids",
        "modules",
        "diseases",
        score_suffix,
        "pred_mask",
        "avail",
    ]
    names.extend(
        ["y", "y_mask"]
        if dataset.scores.task is TaskType.PREVALENT
        else ["event", "time", "signed_time", "label_mask"]
    )
    return {name: file_provenance(root / f"{prefix}_{name}.npy") for name in names}


def _component_settings(
    config: Mapping[str, Any],
    component: str,
    task: TaskType,
) -> Mapping[str, Any]:
    training = _training_mapping(config)
    if component == "planner":
        settings = asdict(
            PlannerTrainingConfig.from_mapping(require_mapping(training, "planner", section="training"))
        )
        settings["candidate_ranker"] = _candidate_ranker_settings(config, task)
        if task is TaskType.LONGITUDINAL:
            settings["longitudinal_reward"] = asdict(_longitudinal_reward(config))
        return settings
    values = require_mapping(training, component, section="training")
    if component == "calibration":
        unknown = sorted(set(values).difference({"min_class_count"}))
        if unknown:
            raise ValueError(f"unknown calibration settings: {unknown}")
        minimum = int(values.get("min_class_count", 10))
        if minimum < 1:
            raise ValueError("min_class_count must be positive")
        return {"min_class_count": minimum}
    factory = PrevalentIntegrationConfig if task is TaskType.PREVALENT else LongitudinalIntegrationConfig
    return asdict(factory.from_mapping(values))


def _publish_training_pair(
    artifact: TraceModelArtifact,
    artifact_path: Path,
    manifest_path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Stage a complete pair; restore the previous publication on an I/O error."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".trace-training-", dir=artifact_path.parent) as temporary:
        staging = Path(temporary)
        staged_artifact = dump_trace_artifact(artifact, staging / artifact_path.name)
        record = file_provenance(staged_artifact)
        record["path"] = str(artifact_path)
        staged_manifest = write_training_manifest(
            staging / manifest_path.name, {**dict(payload), "artifact": record}
        )
        destinations = (artifact_path, manifest_path)
        backups: dict[Path, Path] = {}
        for index, destination in enumerate(destinations):
            if destination.exists():
                backup = staging / f"previous-{index}"
                shutil.copy2(destination, backup)
                backups[destination] = backup
        published: list[Path] = []
        try:
            for source, destination in zip((staged_artifact, staged_manifest), destinations):
                os.replace(source, destination)
                published.append(destination)
        except OSError:
            for destination in reversed(published):
                if destination in backups:
                    os.replace(backups[destination], destination)
                else:
                    destination.unlink(missing_ok=True)
            raise


def _persist_training_result(
    config: Mapping[str, Any],
    dataset: TrainingDataset,
    result: Any,
    *,
    component: str,
    upstream: Mapping[str, Any] | None = None,
) -> TrainingSummary:
    from trace_agent import __version__

    artifact_path = _artifact_path(config, component)
    provenance = split_provenance(dataset)
    provenance["inputs"] = _source_files(dataset)
    provenance["training_inputs"] = supervised_input_provenance(dataset)
    if upstream:
        provenance["upstream_artifacts"] = dict(upstream)
    train_rows = 0 if component == "calibration" else dataset.splits.indices(SplitRole.TRAIN).size
    validation_rows = dataset.splits.indices(SplitRole.VALIDATION).size
    settings = dict(_component_settings(config, component, dataset.scores.task))
    command = {"integration": "train-integration", "calibration": "calibrate", "planner": "train-planner"}[
        component
    ]
    training = {
        "settings": settings,
        "command": command,
        "package_version": __version__,
        "random_seed": settings.get("seed"),
        "manifest_required": True,
        "config_sha256": config.get("_meta", {}).get("sha256"),
        "train_rows_used": int(train_rows),
        "validation_rows_used": int(validation_rows),
        "test_rows_used": 0,
    }
    metrics = dict(getattr(result, "metrics", {}))
    artifact = TraceModelArtifact(
        component=component,
        task=dataset.scores.task,
        model=result.model,
        modules=dataset.scores.modules,
        endpoints=dataset.scores.endpoints,
        training=training,
        provenance=provenance,
        metrics=metrics,
    )
    manifest_path = artifact_path.with_suffix(artifact_path.suffix + ".manifest.json")
    _publish_training_pair(
        artifact,
        artifact_path,
        manifest_path,
        {
            "framework": artifact.framework,
            "command": command,
            "package_version": __version__,
            "random_seed": training["random_seed"],
            "component": component,
            "task": dataset.scores.task.value,
            "created_at": artifact.created_at,
            "training": training,
            "provenance": provenance,
            "metrics": metrics,
            "metrics_path": str(manifest_path),
            "metrics_key": "metrics",
        },
    )
    return TrainingSummary(
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        train_rows=int(train_rows),
        validation_rows=int(validation_rows),
        test_rows_used=0,
    )


def _validate_training_artifact(
    artifact: TraceModelArtifact,
    dataset: TrainingDataset,
) -> None:
    if artifact.modules != dataset.scores.modules:
        raise ValueError("artifact module order does not match the training dataset")
    if artifact.endpoints != dataset.scores.endpoints:
        raise ValueError("artifact endpoint order does not match the training dataset")
    # These checks belong to supervised continuation, not external-cohort inference.
    # Older envelopes may lack split provenance; validate every fingerprint present.
    expected = split_provenance(dataset)
    canonical_key = "split_table_canonical_sha256"
    if canonical_key in artifact.provenance and artifact.provenance[canonical_key] != expected[canonical_key]:
        raise ValueError(f"{artifact.component} artifact training split does not match the current dataset")
    recorded = artifact.provenance.get("splits", {})
    if not isinstance(recorded, Mapping):
        raise TypeError(f"{artifact.component} artifact split provenance must be a mapping")
    for role, values in recorded.items():
        if role not in expected["splits"] or not isinstance(values, Mapping):
            raise ValueError(f"{artifact.component} artifact split provenance is invalid")
        for key in ("count", "row_indices_sha256", "eids_sha256"):
            if key in values and values[key] != expected["splits"][role][key]:
                raise ValueError(
                    f"{artifact.component} artifact {role} split {key} does not match the current dataset"
                )
    recorded_inputs = artifact.provenance.get("training_inputs")
    if recorded_inputs is not None:
        if not isinstance(recorded_inputs, Mapping):
            raise TypeError("artifact training input provenance must be a mapping")
        expected_inputs = supervised_input_provenance(dataset)
        for role, values in recorded_inputs.items():
            if role not in expected_inputs or not isinstance(values, Mapping):
                raise ValueError("artifact training input provenance is invalid")
            for name, fingerprint in values.items():
                if expected_inputs[role].get(name) != fingerprint:
                    raise ValueError(
                        f"{artifact.component} training input {role}.{name} changed; refit the model"
                    )


def train_integration(config: Mapping[str, Any]) -> TrainingSummary:
    _artifact_path(config, "integration")
    dataset = dataset_from_config(config)
    values = require_mapping(_training_mapping(config), "integration", section="training")
    result = (
        train_prevalent_integration(
            dataset.view(SplitRole.TRAIN),
            dataset.view(SplitRole.VALIDATION),
            config=PrevalentIntegrationConfig.from_mapping(values),
        )
        if dataset.scores.task is TaskType.PREVALENT
        else train_longitudinal_integration(
            dataset.view(SplitRole.TRAIN),
            dataset.view(SplitRole.VALIDATION),
            config=LongitudinalIntegrationConfig.from_mapping(values),
        )
    )
    return _persist_training_result(
        config,
        dataset,
        result,
        component="integration",
    )


def calibrate(config: Mapping[str, Any]) -> TrainingSummary:
    _artifact_path(config, "calibration")
    dataset = dataset_from_config(config)
    if dataset.scores.task is not TaskType.PREVALENT:
        raise ValueError("calibration is only defined for prevalent training")
    fusion_path = _artifact_path(config, "integration")
    fusion_artifact = load_trace_artifact(
        fusion_path,
        component="integration",
        task=TaskType.PREVALENT,
    )
    _validate_training_artifact(fusion_artifact, dataset)
    values = _component_settings(config, "calibration", dataset.scores.task)
    result = fit_prevalent_calibration(
        fusion_artifact.model,
        dataset.view(SplitRole.VALIDATION),
        min_class_count=int(values.get("min_class_count", 10)),
    )
    return _persist_training_result(
        config,
        dataset,
        result,
        component="calibration",
        upstream={"integration": file_provenance(fusion_path)},
    )


def _candidate_ranker(
    config: Mapping[str, Any],
    dataset: TrainingDataset,
    *,
    burden: Any,
    knowledge: Any,
) -> KnowledgeCandidateRanker:
    return KnowledgeCandidateRanker(
        kg_matrix=knowledge,
        burden=burden.composite,
        **_candidate_ranker_settings(config, dataset.scores.task),
    )


def _candidate_ranker_settings(config: Mapping[str, Any], task: TaskType) -> dict[str, float]:
    runtime = config.get("planner", {})
    if not isinstance(runtime, Mapping):
        raise TypeError("config.planner must be a mapping")
    training = _training_mapping(config)
    overrides = training.get("candidate_ranker", {})
    if not isinstance(overrides, Mapping):
        raise TypeError("training.candidate_ranker must be a mapping")
    values = {**dict(runtime), **dict(overrides)}
    return {
        "cost_gamma": float(values.get("cost_gamma", 1.0)),
        "cost_epsilon": float(values.get("cost_epsilon", 1e-6)),
        "urgency_threshold": float(
            values.get("urgency_threshold", 0.5 if task is TaskType.PREVALENT else 0.8)
        ),
        "urgency_sigma": float(values.get("urgency_sigma", 0.15)),
    }


def _longitudinal_reward(config: Mapping[str, Any]) -> LongitudinalReward:
    values = _training_mapping(config).get("longitudinal_reward", {})
    if not isinstance(values, Mapping):
        raise TypeError("training.longitudinal_reward must be a mapping")
    unknown = sorted(set(values).difference({"horizons", "horizon_weights"}))
    if unknown:
        raise ValueError(f"unknown longitudinal reward settings: {unknown}")
    return LongitudinalReward(
        horizons=tuple(float(value) for value in values.get("horizons", (5.0, 10.0))),
        horizon_weights=tuple(float(value) for value in values.get("horizon_weights", (0.5, 0.5))),
    )


def train_planner(config: Mapping[str, Any]) -> TrainingSummary:
    _artifact_path(config, "planner")
    dataset = dataset_from_config(config)
    artifacts = require_mapping(config, "artifacts")
    fusion_path = _artifact_path(config, "integration")
    fusion_artifact = load_trace_artifact(
        fusion_path,
        component="integration",
        task=dataset.scores.task,
    )
    _validate_training_artifact(fusion_artifact, dataset)
    burden_path = required(artifacts, "burden_profile", section="artifacts")
    knowledge_path = required(artifacts, "knowledge_matrix", section="artifacts")
    burden = load_burden_profile(burden_path, modules=dataset.scores.modules)
    knowledge = load_knowledge_matrix(
        knowledge_path,
        modules=dataset.scores.modules,
        endpoints=dataset.scores.endpoints,
    )
    ranker = _candidate_ranker(
        config,
        dataset,
        burden=burden,
        knowledge=knowledge,
    )
    planner_values = require_mapping(
        _training_mapping(config),
        "planner",
        section="training",
    )
    planner_config = PlannerTrainingConfig.from_mapping(planner_values)
    upstream: dict[str, Any] = {
        "integration": file_provenance(fusion_path),
        "knowledge_matrix": file_provenance(knowledge_path),
        "burden_profile": file_provenance(burden_path),
    }
    if dataset.scores.task is TaskType.PREVALENT:
        calibration: PrevalentCalibrationModel | None = None
        calibration_path = artifacts.get("calibrators")
        if calibration_path not in (None, ""):
            calibration_artifact = load_trace_artifact(
                str(calibration_path),
                component="calibration",
                task=TaskType.PREVALENT,
            )
            _validate_training_artifact(calibration_artifact, dataset)
            validate_upstream_artifacts(calibration_artifact, {"integration": fusion_path})
            if not isinstance(calibration_artifact.model, PrevalentCalibrationModel):
                raise TypeError("calibration artifact contains an incompatible model")
            calibration = calibration_artifact.model
            upstream["calibration"] = file_provenance(str(calibration_path))
        result = train_prevalent_planner(
            dataset,
            fusion=fusion_artifact.model,
            calibration=calibration,
            burden=burden,
            ranker=ranker,
            config=planner_config,
        )
    else:
        result = train_longitudinal_planner(
            dataset,
            fusion=fusion_artifact.model,
            burden=burden,
            ranker=ranker,
            reward=_longitudinal_reward(config),
            config=planner_config,
        )
    return _persist_training_result(
        config,
        dataset,
        result,
        component="planner",
        upstream=upstream,
    )
