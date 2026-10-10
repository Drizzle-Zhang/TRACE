from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]

from trace_agent.agents.longitudinal import LongitudinalAgent
from trace_agent.agents.prevalent import PrevalentAgent
from trace_agent.agents.trace_agent import TRACEAgent
from trace_agent.consistency.longitudinal import load_longitudinal_scale
from trace_agent.consistency.validator import ConsistencyValidator
from trace_agent.contracts.consistency import ConsistencyThresholds
from trace_agent.contracts.scores import ModuleScoreBatch
from trace_agent.contracts.task import TaskType
from trace_agent.data.burden import BurdenProfile
from trace_agent.integration.absolute_risk import CoxAbsoluteRiskMapper
from trace_agent.integration.base import RiskIntegrator
from trace_agent.integration.longitudinal_cox import (
    LongitudinalRiskIntegrator,
)
from trace_agent.integration.prevalent_lgbm import (
    PrevalentRiskIntegrator,
)
from trace_agent.knowledge.evidence_store import MedicalEvidenceStore
from trace_agent.planning.candidate_ranker import (
    KnowledgeCandidateRanker,
)
from trace_agent.planning.engine import PlannerConfig, PlanningEngine
from trace_agent.planning.q_model import load_q_artifact
from trace_agent.runtime.model_artifact import load_trace_artifact, validate_upstream_artifacts
from trace_agent.runtime.provenance import file_provenance


def _required(mapping: Mapping[str, Any], key: str, *, section: str) -> Any:
    if key not in mapping or mapping[key] in (None, ""):
        raise ValueError(f"missing required config: {section}.{key}")
    return mapping[key]


def _load_array(path: str | Path, *, preferred_key: str) -> np.ndarray:
    resolved = Path(path)
    if resolved.suffix.lower() == ".npy":
        return np.load(resolved, allow_pickle=False)
    if resolved.suffix.lower() == ".npz":
        with np.load(resolved, allow_pickle=False) as payload:
            if preferred_key in payload:
                return np.asarray(payload[preferred_key])
            keys = list(payload.files)
            if len(keys) != 1:
                raise ValueError(f"{resolved} must contain {preferred_key!r} " "or exactly one array")
            return np.asarray(payload[keys[0]])
    raise ValueError(f"unsupported array artifact: {resolved}")


def _load_string_array(path: str | Path, *, allow_pickle: bool) -> tuple[str, ...]:
    try:
        values = np.load(path, allow_pickle=False)
    except ValueError as exc:
        if "Object arrays cannot be loaded" not in str(exc) or not allow_pickle:
            raise
        values = np.load(path, allow_pickle=True)
    flattened = values.reshape(-1).tolist()
    if not all(isinstance(value, str) for value in flattened):
        raise ValueError(f"string array contains non-string entries: {path}")
    return tuple(str(value) for value in flattened)


def _short_module_name(value: str) -> str:
    name = str(value)
    return name.split("__", 1)[1] if "__" in name else name


def _schema_feature_code_name(value: Any) -> tuple[str, str]:
    """Resolve the field aliases used by the original matrix schema loader."""
    if isinstance(value, Mapping):
        code = (
            value.get("feature_code")
            or value.get("code")
            or value.get("field")
            or value.get("id")
            or value.get("name")
        )
        name = (
            value.get("feature_name")
            or value.get("name")
            or value.get("label")
            or value.get("display_name")
            or code
        )
        return str(code), str(name)
    return str(value), str(value)


def load_evidence_configuration(
    values: Mapping[str, Any], *, allow_pickle_names: bool = False
) -> tuple[MedicalEvidenceStore, tuple[str, ...], dict[str, str]]:
    """Resolve the source scorer's feature schema and disease alias tables."""
    features = _load_string_array(
        _required(values, "feature_names", section="consistency"), allow_pickle=allow_pickle_names
    )
    store = MedicalEvidenceStore(
        _required(values, "evidence_bundle", section="consistency"),
        disease_metadata_dir=values.get("disease_metadata_dir"),
    )
    names: dict[str, str] = {}
    schema_path = values.get("feature_schema_path")
    if schema_path:
        schema = json.loads(Path(str(schema_path)).read_text(encoding="utf-8"))
        if not isinstance(schema, Mapping):
            raise TypeError("consistency feature schema must be a JSON mapping")
        global_names = schema.get("feature_code_to_name", {})
        if not isinstance(global_names, Mapping):
            raise TypeError("schema.feature_code_to_name must be a mapping")
        names.update({str(code): str(name) for code, name in global_names.items()})
        # The source schema merges global fields in this order, then uses
        # module names only for codes missing from the global mapping.
        for key in ("feature_codes", "final_feature_names", "features", "feature_names"):
            global_features = schema.get(key)
            if isinstance(global_features, Mapping):
                for code, name in global_features.items():
                    names.setdefault(str(code), str(name))
            elif isinstance(global_features, list):
                for value in global_features:
                    code, name = _schema_feature_code_name(value)
                    names.setdefault(code, name)
        module_maps = schema.get("module_feature_code_to_name", {})
        if not isinstance(module_maps, Mapping):
            raise TypeError("schema.module_feature_code_to_name must be a mapping")
        for mapping in module_maps.values():
            if not isinstance(mapping, Mapping):
                raise TypeError("schema module feature names must be mappings")
            for code, name in mapping.items():
                names.setdefault(str(code), str(name))
        modules = schema.get("modules", [])
        if isinstance(modules, list):
            for module in modules:
                if not isinstance(module, Mapping) or not (
                    module.get("name") or module.get("module") or module.get("id") or module.get("code")
                ):
                    continue
                module_features = (
                    module.get("feature_codes")
                    or module.get("final_feature_names")
                    or module.get("features")
                    or []
                )
                pairs = [_schema_feature_code_name(value) for value in module_features]
                feature_values = module.get("features")
                if (
                    module.get("feature_codes") is not None
                    and isinstance(feature_values, list)
                    and len(feature_values) == len(pairs)
                ):
                    pairs = [
                        (code, _schema_feature_code_name(value)[1])
                        for (code, _), value in zip(pairs, feature_values)
                    ]
                for code, name in pairs:
                    if code:
                        names.setdefault(code, name)
    inline = values.get("feature_code_to_name", {})
    if not isinstance(inline, Mapping):
        raise TypeError("consistency.feature_code_to_name must be a mapping")
    names.update({str(code): str(name) for code, name in inline.items()})
    return store, features, names


def load_knowledge_matrix(
    path: str | Path,
    *,
    modules: tuple[str, ...],
    endpoints: tuple[str, ...],
) -> np.ndarray:
    resolved = Path(path)
    if resolved.suffix.lower() in {".npy", ".npz"}:
        values = _load_array(resolved, preferred_key="knowledge_matrix")
    elif resolved.suffix.lower() == ".csv":
        frame = pd.read_csv(resolved, index_col=0)
        frame.index = frame.index.astype(str)
        frame.columns = frame.columns.astype(str)
        expected_module_keys = {_short_module_name(module) for module in modules}
        row_keys = {_short_module_name(value) for value in frame.index}
        column_keys = {_short_module_name(value) for value in frame.columns}
        if not expected_module_keys.intersection(row_keys) and expected_module_keys.intersection(column_keys):
            frame = frame.T
        row_lookup: dict[str, str] = {}
        for row_name in frame.index:
            key = _short_module_name(row_name)
            if key in row_lookup:
                raise ValueError(f"knowledge matrix has duplicate module key {key!r}")
            row_lookup[key] = row_name
        aligned_rows = [row_lookup.get(_short_module_name(module)) for module in modules]
        missing_modules = [module for module, row_name in zip(modules, aligned_rows) if row_name is None]
        missing_endpoints = [endpoint for endpoint in endpoints if endpoint not in frame.columns]
        if missing_modules or missing_endpoints:
            raise ValueError(
                "knowledge matrix name mismatch; "
                f"missing modules={missing_modules}, "
                f"missing endpoints={missing_endpoints}"
            )
        values = (
            frame.loc[aligned_rows, list(endpoints)]
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(dtype=np.float32)
        )
    else:
        raise ValueError(f"unsupported knowledge matrix format: {resolved}")
    values = np.asarray(values, dtype=np.float32)
    expected = (len(modules), len(endpoints))
    if values.shape != expected and values.shape == expected[::-1]:
        values = values.T
    if values.shape != expected:
        raise ValueError(f"knowledge matrix shape must be {expected}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("knowledge matrix contains non-finite values")
    return values


def load_burden_profile(path: str | Path, *, modules: tuple[str, ...]) -> BurdenProfile:
    resolved = Path(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("burden profile must be a YAML mapping")
    module_payload = payload.get("modules", payload)
    if not isinstance(module_payload, dict):
        raise TypeError("burden profile modules must be a mapping")
    monetary: list[float] = []
    time: list[float] = []
    missing: list[str] = []
    for module in modules:
        lookup_name = module if module in module_payload else _short_module_name(module)
        if lookup_name not in module_payload:
            missing.append(module)
            continue
        value = module_payload[lookup_name]
        if isinstance(value, Mapping):
            monetary.append(float(value.get("monetary", value.get("cost", 0.0))))
            time.append(float(value.get("time", 0.0)))
        else:
            monetary.append(float(value))
            time.append(0.0)
    if missing:
        raise ValueError(f"burden profile is missing modules: {missing}")
    weights = payload.get("weights", {})
    if not isinstance(weights, Mapping):
        raise TypeError("burden weights must be a mapping")
    return BurdenProfile(
        modules=modules,
        monetary=np.asarray(monetary, dtype=np.float32),
        time=np.asarray(time, dtype=np.float32),
        monetary_weight=float(weights.get("monetary", 1.0)),
        time_weight=float(weights.get("time", 0.0)),
    )


def _load_calibrators(
    configured: str | Path | None,
    *,
    modules: tuple[str, ...],
    endpoints: tuple[str, ...],
    integration_path: str | Path | None = None,
) -> dict[str, Any]:
    if configured in (None, ""):
        return {}
    path = Path(str(configured))
    artifact = load_trace_artifact(
        path,
        component="calibration",
        task=TaskType.PREVALENT,
    )
    if artifact.modules != modules:
        raise ValueError("calibration artifact module order does not match the input bundle")
    if artifact.endpoints != endpoints:
        raise ValueError("calibration artifact endpoint order does not match the input bundle")
    if integration_path is not None:
        validate_upstream_artifacts(artifact, {"integration": integration_path})
    payload = getattr(artifact.model, "calibrators", None)
    if not isinstance(payload, Mapping):
        raise TypeError("TRACE calibration model must expose an endpoint mapping")
    calibrators = {str(key): value for key, value in payload.items()}
    if set(calibrators) != set(endpoints):
        raise ValueError("calibration artifact mapping must cover every input endpoint")
    return calibrators


def _format_horizon(value: float) -> str:
    return f"{int(value)}y" if float(value).is_integer() else f"{float(value):g}y"


def load_absolute_risk_mapper(
    configured: str | Path | Mapping[str, Any] | None,
    *,
    endpoints: tuple[str, ...],
) -> CoxAbsoluteRiskMapper | None:
    if configured in (None, ""):
        return None
    if isinstance(configured, Mapping):
        baseline_path = _required(
            configured,
            "baseline_cumhaz",
            section="artifacts.absolute_risk",
        )
        time_path = _required(
            configured,
            "time_grid",
            section="artifacts.absolute_risk",
        )
        baseline_cumhaz = np.asarray(
            np.load(baseline_path, allow_pickle=False),
            dtype=np.float64,
        )
        time_grid = np.asarray(
            np.load(time_path, allow_pickle=False),
            dtype=np.float64,
        ).reshape(-1)
        expected = (len(endpoints), time_grid.size)
        if baseline_cumhaz.shape != expected and baseline_cumhaz.shape == expected[::-1]:
            baseline_cumhaz = baseline_cumhaz.T
        if baseline_cumhaz.shape != expected:
            raise ValueError(
                "baseline cumulative hazard must have shape " f"{expected}, got {baseline_cumhaz.shape}"
            )
        if (
            np.any(~np.isfinite(time_grid))
            or np.any(np.diff(time_grid) <= 0)
            or np.any(~np.isfinite(baseline_cumhaz))
            or np.any(baseline_cumhaz < 0)
        ):
            raise ValueError("baseline cumulative hazard/time grid is invalid")
        endpoint_path = configured.get("endpoints_file")
        if endpoint_path:
            try:
                endpoint_values = np.load(endpoint_path, allow_pickle=False)
            except ValueError as exc:
                if "Object arrays cannot be loaded" not in str(exc):
                    raise
                endpoint_values = np.load(endpoint_path, allow_pickle=True)
            artifact_endpoints = tuple(str(value) for value in endpoint_values.reshape(-1).tolist())
            if artifact_endpoints != endpoints:
                raise ValueError("absolute-risk endpoint order does not match input")
        horizons = np.asarray(configured.get("horizons", time_grid), dtype=np.float64).reshape(-1)
        if horizons.size == 0 or np.any(~np.isfinite(horizons)) or np.any(horizons <= 0):
            raise ValueError("absolute-risk horizons must be finite and positive")
        baseline_survival: dict[str, np.ndarray] = {}
        for horizon in horizons:
            cumulative = np.asarray(
                [
                    np.interp(
                        float(horizon),
                        time_grid,
                        baseline_cumhaz[index],
                    )
                    for index in range(len(endpoints))
                ],
                dtype=np.float64,
            )
            baseline_survival[_format_horizon(float(horizon))] = np.clip(
                np.exp(-cumulative), np.finfo(np.float64).tiny, 1.0
            )
        return CoxAbsoluteRiskMapper(
            endpoints=endpoints,
            baseline_survival=baseline_survival,
        )
    path = Path(str(configured))
    with np.load(path, allow_pickle=False) as payload:
        if "endpoints" in payload:
            artifact_endpoints = tuple(str(value) for value in payload["endpoints"].reshape(-1).tolist())
            if artifact_endpoints != endpoints:
                raise ValueError("absolute-risk endpoint order does not match input")
        baseline = {
            key.removeprefix("baseline_survival_"): np.asarray(payload[key], dtype=np.float64)
            for key in payload.files
            if key != "endpoints"
        }
    if not baseline:
        raise ValueError("absolute-risk artifact contains no baseline survival arrays")
    return CoxAbsoluteRiskMapper(endpoints=endpoints, baseline_survival=baseline)


def _artifact_dimensions(
    artifact: Any,
    *,
    modules: tuple[str, ...],
    endpoints: tuple[str, ...],
    name: str,
) -> None:
    for attribute, expected in (
        ("M", len(modules)),
        ("D", len(endpoints)),
    ):
        value = getattr(artifact, attribute, None)
        if value is not None and int(value) != expected:
            raise ValueError(f"{name}.{attribute}={value} does not match {expected}")
    artifact_modules = getattr(artifact, "modules", None)
    if artifact_modules is not None:
        normalized_artifact = tuple(_short_module_name(str(value)) for value in artifact_modules)
        normalized_input = tuple(_short_module_name(value) for value in modules)
        if normalized_artifact != normalized_input:
            raise ValueError(f"{name} module order does not match the input bundle")
    artifact_endpoints = getattr(artifact, "diseases", None)
    if artifact_endpoints is not None and tuple(str(value) for value in artifact_endpoints) != endpoints:
        raise ValueError(f"{name} endpoint order does not match the input bundle")


def build_trace_agent(
    config: Mapping[str, Any],
    *,
    batch: ModuleScoreBatch,
) -> TRACEAgent:
    task = TaskType.parse(config.get("task", batch.task))
    if task is not batch.task:
        raise ValueError("configured task does not match the input score bundle")
    artifacts = config.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TypeError("config.artifacts must be a mapping")
    fusion_path = Path(str(_required(artifacts, "fusion_model", section="artifacts")))
    q_path = Path(str(_required(artifacts, "q_model", section="artifacts")))
    kg_path = Path(str(_required(artifacts, "knowledge_matrix", section="artifacts")))
    burden_path = Path(str(_required(artifacts, "burden_profile", section="artifacts")))
    fusion_artifact = load_trace_artifact(
        fusion_path,
        component="integration",
        task=task,
    )
    if fusion_artifact.modules != batch.modules:
        raise ValueError("fusion artifact module order does not match the input bundle")
    if fusion_artifact.endpoints != batch.endpoints:
        raise ValueError("fusion artifact endpoint order does not match the input bundle")
    fusion = fusion_artifact.model
    q_artifact = load_q_artifact(
        q_path,
        task=task,
        modules=batch.modules,
        endpoints=batch.endpoints,
        upstream_paths={
            "integration": fusion_path,
            "knowledge_matrix": kg_path,
            "burden_profile": burden_path,
            "calibration": artifacts.get("calibrators") if task is TaskType.PREVALENT else None,
        },
    )
    q_model = q_artifact.model
    _artifact_dimensions(
        fusion,
        modules=batch.modules,
        endpoints=batch.endpoints,
        name="fusion",
    )
    knowledge = load_knowledge_matrix(kg_path, modules=batch.modules, endpoints=batch.endpoints)
    burden = load_burden_profile(burden_path, modules=batch.modules)
    if task is TaskType.PREVALENT:
        integrator: RiskIntegrator = PrevalentRiskIntegrator(
            fusion=fusion,
            calibrators=_load_calibrators(
                artifacts.get("calibrators"),
                modules=batch.modules,
                endpoints=batch.endpoints,
                integration_path=fusion_path,
            ),
        )
    else:
        integrator = LongitudinalRiskIntegrator(
            fusion=fusion,
            absolute_risk_mapper=load_absolute_risk_mapper(
                artifacts.get("absolute_risk"),
                endpoints=batch.endpoints,
            ),
        )
    planner_values = config.get("planner", {})
    if not isinstance(planner_values, Mapping):
        raise TypeError("config.planner must be a mapping")
    training_settings = q_artifact.training.get("settings", {})
    if not isinstance(training_settings, Mapping):
        raise TypeError("Q-model training settings must be a mapping")
    trained_stop_mode = training_settings.get("stop_value_mode")
    default_stop_mode = "fixed_zero" if trained_stop_mode is None else str(trained_stop_mode)
    stop_value_mode = str(planner_values.get("stop_value_mode", default_stop_mode))
    if trained_stop_mode is not None and stop_value_mode != str(trained_stop_mode):
        raise ValueError(
            f"planner.stop_value_mode={stop_value_mode!r} conflicts with "
            f"Q-model training stop_value_mode={trained_stop_mode!r}"
        )
    planner = PlanningEngine(
        integrator=integrator,
        q_model=q_model,
        candidate_ranker=KnowledgeCandidateRanker(
            kg_matrix=knowledge,
            burden=burden.composite,
            cost_gamma=float(planner_values.get("cost_gamma", 1.0)),
            cost_epsilon=float(planner_values.get("cost_epsilon", 1e-6)),
            urgency_threshold=float(
                planner_values.get(
                    "urgency_threshold",
                    0.5 if task is TaskType.PREVALENT else 0.8,
                )
            ),
            urgency_sigma=float(planner_values.get("urgency_sigma", 0.15)),
        ),
        burden_profile=burden,
        config=PlannerConfig(
            max_steps=int(planner_values.get("max_steps", 0)),
            candidate_actions=int(planner_values.get("candidate_actions", 16)),
            stop_threshold=float(planner_values.get("stop_threshold", 0.0)),
            stop_value_mode=stop_value_mode,
        ),
    )
    consistency_values = config.get("consistency", {})
    if not isinstance(consistency_values, Mapping):
        raise TypeError("config.consistency must be a mapping")
    validator = None
    contribution_provider = None
    consistency_scale_path = None
    evidence_store = None
    if bool(consistency_values.get("enabled", False)):
        module_score_transform = None
        if task is TaskType.LONGITUDINAL:
            consistency_scale_path = Path(
                consistency_values.get("longitudinal_scale_path")
                or (Path(config["output"]["directory"]) / "longitudinal_consistency_scale.json")
            )
            module_score_transform = load_longitudinal_scale(
                consistency_scale_path, modules=batch.modules, endpoints=batch.endpoints
            ).transform
            if not consistency_values.get("horizon"):
                raise ValueError("longitudinal consistency requires consistency.horizon")
        threshold_values = consistency_values.get("thresholds", {})
        if not isinstance(threshold_values, Mapping):
            raise ValueError("consistency.thresholds must be a mapping")
        medical_weights = (
            None
            if not consistency_values.get("feature_weights")
            else _load_array(
                consistency_values["feature_weights"],
                preferred_key="feature_weights",
            )
        )
        evidence_weights = (
            None
            if not consistency_values.get("evidence_weights")
            else _load_array(
                consistency_values["evidence_weights"],
                preferred_key="evidence_weights",
            )
        )
        evidence_reliability = (
            None
            if not consistency_values.get("evidence_reliability")
            else _load_array(
                consistency_values["evidence_reliability"],
                preferred_key="evidence_reliability",
            )
        )
        evidence_bundle = consistency_values.get("evidence_bundle")
        evidence_rules = None
        if evidence_bundle:
            if evidence_weights is not None:
                raise ValueError(
                    "configure either consistency.evidence_weights or "
                    "consistency.evidence_bundle, not both"
                )
            data_values = config.get("data", {})
            allow_pickle_names = bool(
                data_values.get("allow_legacy_pickle_names", False)
                if isinstance(data_values, Mapping)
                else False
            )
            evidence_store, feature_names, code_to_name = load_evidence_configuration(
                consistency_values, allow_pickle_names=allow_pickle_names
            )
            compiled_evidence = evidence_store.compile(
                endpoints=batch.endpoints,
                features=feature_names,
                feature_code_to_name=code_to_name,
                min_certainty_rank=float(consistency_values.get("min_certainty_rank", 0.0)),
                min_strength_rank=float(consistency_values.get("min_strength_rank", 0.0)),
            )
            evidence_weights = compiled_evidence.weights
            evidence_reliability = compiled_evidence.reliability
            evidence_rules = compiled_evidence.rules
        validator = ConsistencyValidator(
            kg_matrix=knowledge,
            medical_weights=medical_weights,
            evidence_weights=evidence_weights,
            evidence_reliability=evidence_reliability,
            evidence_rules=evidence_rules,
            module_score_transform=module_score_transform,
            prediction_horizon=consistency_values.get("horizon"),
            thresholds=ConsistencyThresholds(
                prediction=float(threshold_values.get("prediction", 0.5)),
                kg=float(threshold_values.get("kg", 0.5)),
                feature=float(threshold_values.get("feature", 0.5)),
                evidence=float(threshold_values.get("evidence", 0.5)),
                combined=float(threshold_values.get("combined", 0.5)),
            ),
        )
        contributions_path = consistency_values.get("contributions")
        if contributions_path:
            contributions = _load_array(contributions_path, preferred_key="contributions").astype(
                np.float32, copy=False
            )
            expected_prefix = (
                batch.n_participants,
                len(batch.endpoints),
            )
            if contributions.ndim != 3 or contributions.shape[:2] != expected_prefix:
                raise ValueError(
                    "consistency contributions must have shape "
                    f"{expected_prefix} + (features,), got "
                    f"{contributions.shape}"
                )

            def contribution_provider(participant_batch: ModuleScoreBatch, _planning: Any) -> np.ndarray:
                index = int(participant_batch.metadata.get("source_participant_index", 0))
                return contributions[index : index + 1]

    provenance = {
        "fusion_model": file_provenance(fusion_path),
        "q_model": file_provenance(q_path),
        "knowledge_matrix": file_provenance(kg_path),
        "burden_profile": file_provenance(burden_path),
    }
    if consistency_scale_path is not None:
        provenance["consistency_scale"] = file_provenance(consistency_scale_path)
    for key in (
        "feature_weights",
        "feature_names",
        "evidence_weights",
        "evidence_reliability",
        "contributions",
        "feature_schema_path",
        "feature_values_path",
    ):
        configured = consistency_values.get(key)
        if configured and Path(str(configured)).is_file():
            provenance[key] = file_provenance(configured)
    if evidence_store is not None:
        for source_path in evidence_store.source_paths:
            provenance[f"evidence_bundle.{source_path.name}"] = file_provenance(source_path)
        evidence_settings = {
            "feature_code_to_name": code_to_name,
            "min_certainty_rank": float(consistency_values.get("min_certainty_rank", 0.0)),
            "min_strength_rank": float(consistency_values.get("min_strength_rank", 0.0)),
        }
        provenance["evidence_configuration"] = {
            "sha256": hashlib.sha256(
                json.dumps(evidence_settings, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
        }
    absolute_risk_config = artifacts.get("absolute_risk")
    if isinstance(absolute_risk_config, Mapping):
        for key in (
            "baseline_cumhaz",
            "time_grid",
            "endpoints_file",
        ):
            configured = absolute_risk_config.get(key)
            if configured:
                provenance[f"absolute_risk.{key}"] = file_provenance(configured)
    elif absolute_risk_config:
        provenance["absolute_risk"] = file_provenance(absolute_risk_config)
    calibrators_path = artifacts.get("calibrators")
    if calibrators_path and Path(str(calibrators_path)).is_file():
        provenance["calibrators"] = file_provenance(calibrators_path)
    agent_class = PrevalentAgent if task is TaskType.PREVALENT else LongitudinalAgent
    task_agent = agent_class(
        planner=planner,
        consistency_validator=validator,
        contribution_provider=contribution_provider,
        decision_strategy=str(consistency_values.get("strategy", "combined")),
        artifact_provenance=provenance,
    )
    return TRACEAgent(task_agents=[task_agent])
