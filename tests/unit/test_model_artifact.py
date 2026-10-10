from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from trace_agent.contracts.task import TaskType
from trace_agent.planning.q_model import load_q_model
from trace_agent.runtime.model_artifact import (
    ARTIFACT_SCHEMA_VERSION,
    TraceModelArtifact,
    dump_trace_artifact,
    load_trace_artifact,
)


class DummyModel:
    def predict(self, rows: np.ndarray) -> np.ndarray:
        return np.zeros(len(rows), dtype=np.float32)


def _artifact(*, component: str = "integration") -> TraceModelArtifact:
    return TraceModelArtifact(
        component=component,
        task=TaskType.PREVALENT,
        model=DummyModel(),
        modules=("M1",),
        endpoints=("D1",),
        training={"seed": 7},
        provenance={"split_table_sha256": "a" * 64},
        metrics={"macro_auroc": 0.75},
    )


def test_trace_model_artifact_round_trip(tmp_path: Path) -> None:
    path = dump_trace_artifact(_artifact(), tmp_path / "fusion.joblib")

    loaded = load_trace_artifact(
        path,
        component="integration",
        task="prevalent",
    )

    assert loaded.schema_version == ARTIFACT_SCHEMA_VERSION
    assert isinstance(loaded.model, DummyModel)
    assert loaded.modules == ("M1",)
    assert not (tmp_path / "fusion.joblib.tmp").exists()


def test_artifact_loader_rejects_wrong_component(tmp_path: Path) -> None:
    path = dump_trace_artifact(_artifact(component="planner"), tmp_path / "q.joblib")

    with pytest.raises(ValueError, match="expected component integration"):
        load_trace_artifact(path, component="integration")


def test_artifact_loader_rejects_raw_joblib_objects(tmp_path: Path) -> None:
    path = tmp_path / "raw.joblib"
    joblib.dump(DummyModel(), path)

    with pytest.raises(TypeError, match="TRACE model artifact"):
        load_trace_artifact(path)


def test_artifact_rejects_duplicate_named_axes() -> None:
    with pytest.raises(ValueError, match="module names must be unique"):
        TraceModelArtifact(
            component="integration",
            task=TaskType.PREVALENT,
            model=DummyModel(),
            modules=("M1", "M1"),
            endpoints=("D1",),
        )


def test_q_model_loader_unwraps_trace_planner_artifact(tmp_path: Path) -> None:
    path = dump_trace_artifact(_artifact(component="planner"), tmp_path / "q.joblib")

    model = load_q_model(path)

    np.testing.assert_array_equal(model.predict(np.ones((2, 3))), [0.0, 0.0])
