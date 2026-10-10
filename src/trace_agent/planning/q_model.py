from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import numpy as np

from trace_agent.contracts.task import TaskType
from trace_agent.runtime.model_artifact import (
    TraceModelArtifact,
    load_trace_artifact,
    validate_upstream_artifacts,
)


class QModel(Protocol):
    def predict(self, rows: np.ndarray) -> np.ndarray: ...


def load_q_artifact(
    path: str | Path,
    *,
    task: TaskType | str | None = None,
    modules: tuple[str, ...] | None = None,
    endpoints: tuple[str, ...] | None = None,
    upstream_paths: Mapping[str, str | Path | None] | None = None,
) -> TraceModelArtifact:
    artifact = load_trace_artifact(
        Path(path),
        component="planner",
        task=task,
    )
    if modules is not None and artifact.modules != modules:
        raise ValueError("Q-model module order does not match the input bundle")
    if endpoints is not None and artifact.endpoints != endpoints:
        raise ValueError("Q-model endpoint order does not match the input bundle")
    if upstream_paths is not None:
        validate_upstream_artifacts(artifact, upstream_paths)
    model = artifact.model
    if not callable(getattr(model, "predict", None)):
        raise TypeError("Q-model artifact must provide predict(rows)")
    return artifact


def load_q_model(
    path: str | Path,
    *,
    task: TaskType | str | None = None,
    modules: tuple[str, ...] | None = None,
    endpoints: tuple[str, ...] | None = None,
    upstream_paths: Mapping[str, str | Path | None] | None = None,
) -> QModel:
    return load_q_artifact(
        path,
        task=task,
        modules=modules,
        endpoints=endpoints,
        upstream_paths=upstream_paths,
    ).model
