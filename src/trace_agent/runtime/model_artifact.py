from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from trace_agent.contracts.task import TaskType
from trace_agent.identity import framework_metadata
from trace_agent.runtime.provenance import file_provenance

ARTIFACT_SCHEMA_VERSION = 1
_COMPONENTS = {"integration", "calibration", "planner", "module"}


def _validated_names(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    if any(not value for value in names):
        raise ValueError(f"{label} names must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} names must be unique")
    return names


@dataclass(frozen=True)
class TraceModelArtifact:
    component: str
    task: TaskType
    model: Any
    modules: tuple[str, ...]
    endpoints: tuple[str, ...]
    training: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = ARTIFACT_SCHEMA_VERSION
    framework: Mapping[str, Any] = field(default_factory=framework_metadata)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        component = str(self.component).strip().lower()
        if component not in _COMPONENTS:
            raise ValueError(f"component must be one of {sorted(_COMPONENTS)}, got {component!r}")
        if int(self.schema_version) != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported TRACE artifact schema {self.schema_version!r}; "
                f"expected {ARTIFACT_SCHEMA_VERSION}"
            )
        if self.model is None:
            raise ValueError("TRACE model artifact requires a model")
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "task", TaskType.parse(self.task))
        object.__setattr__(
            self,
            "modules",
            _validated_names(tuple(self.modules), label="module"),
        )
        object.__setattr__(
            self,
            "endpoints",
            _validated_names(tuple(self.endpoints), label="endpoint"),
        )
        object.__setattr__(self, "training", dict(self.training))
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "framework", dict(self.framework))
        if component == "module":
            if len(self.modules) != 1 or len(self.endpoints) != 1:
                raise ValueError("module artifacts require one module and one endpoint")
            from trace_agent.risk_skills.module_model import ModuleModel

            if not isinstance(self.model, ModuleModel):
                raise TypeError("module artifact requires a ModuleModel")
            if self.model.task is not self.task:
                raise ValueError("module artifact task mismatch")
            if self.modules != (self.model.module,) or self.endpoints != (self.model.endpoint,):
                raise ValueError("module artifact identity mismatch")


def validate_upstream_artifacts(artifact: TraceModelArtifact, paths: Mapping[str, str | Path | None]) -> None:
    """Check recorded model dependencies; legacy envelopes may lack this record."""
    recorded = artifact.provenance.get("upstream_artifacts")
    if recorded is None:
        return
    if not isinstance(recorded, Mapping):
        raise TypeError("artifact upstream_artifacts must be a mapping")
    for name, entry in recorded.items():
        configured = paths.get(name)
        if configured is None or not isinstance(entry, Mapping) or not entry.get("sha256"):
            raise ValueError(f"{artifact.component} upstream {name} identity is missing")
        if file_provenance(configured)["sha256"] != entry["sha256"]:
            raise ValueError(f"{artifact.component} upstream {name} identity differs from the fitted model")
    if artifact.component == "planner" and paths.get("calibration") and "calibration" not in recorded:
        raise ValueError(
            "planner upstream calibration identity differs: model was trained without calibration"
        )


def dump_trace_artifact(
    artifact: TraceModelArtifact,
    path: str | Path,
) -> Path:
    if not isinstance(artifact, TraceModelArtifact):
        raise TypeError("artifact must be a TraceModelArtifact")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        joblib.dump(artifact, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_trace_artifact(
    path: str | Path,
    *,
    component: str | None = None,
    task: TaskType | str | None = None,
) -> TraceModelArtifact:
    resolved = Path(path).expanduser().resolve()
    loaded = joblib.load(resolved)
    if not isinstance(loaded, TraceModelArtifact):
        raise TypeError(
            f"{resolved} is not a TRACE model artifact; retrain or convert it "
            "to the versioned TRACE artifact format"
        )
    if loaded.schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported TRACE artifact schema {loaded.schema_version}; "
            f"expected {ARTIFACT_SCHEMA_VERSION}"
        )
    if component is not None:
        expected_component = str(component).strip().lower()
        if loaded.component != expected_component:
            raise ValueError(f"expected component {expected_component}, got {loaded.component}")
    if task is not None:
        expected_task = TaskType.parse(task)
        if loaded.task is not expected_task:
            raise ValueError(f"expected task {expected_task.value}, got {loaded.task.value}")
    if loaded.training.get("manifest_required", False):
        manifest_path = resolved.with_suffix(resolved.suffix + ".manifest.json")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"training artifact manifest is missing or invalid: {manifest_path}") from error
        if not isinstance(manifest, Mapping) or not isinstance(manifest.get("artifact"), Mapping):
            raise ValueError("training artifact manifest must contain an artifact record")
        if manifest["artifact"].get("sha256") != file_provenance(resolved)["sha256"]:
            raise ValueError("training artifact manifest hash mismatch")
        if manifest.get("component") != loaded.component or manifest.get("task") != loaded.task.value:
            raise ValueError("training artifact manifest component/task mismatch")
    return loaded
