from __future__ import annotations

import numpy as np
import pytest

from trace_agent.runtime.model_artifact import TraceModelArtifact, dump_trace_artifact, load_trace_artifact


class MarginEstimator:
    def predict(self, matrix, *, output_margin=False):
        assert output_margin is True
        return np.full(matrix.shape[0], -0.7)


class ProbabilityEstimator:
    classes_ = np.array([1, 0])

    def predict_proba(self, matrix):
        return np.tile([0.7, 0.3], (matrix.shape[0], 1))


def test_cox_wrapper_requests_log_risk():
    from trace_agent.risk_skills.module_model import ModuleModel

    model = ModuleModel("longitudinal", "M1", "D1", ("b", "a"), MarginEstimator())
    assert model.score_scale == "log_risk"
    np.testing.assert_allclose(model.predict(np.ones((3, 2))), -0.7)
    with pytest.raises(ValueError, match="feature width"):
        model.predict(np.ones((3, 1)))


def test_probability_wrapper_uses_positive_class_position():
    from trace_agent.risk_skills.module_model import ModuleModel

    model = ModuleModel("prevalent", "M1", "D1", ("a",), ProbabilityEstimator())
    assert model.score_scale == "probability"
    np.testing.assert_allclose(model.predict(np.ones((2, 1))), 0.7)


@pytest.mark.parametrize("names", [(), ("a", "a"), ("",)])
def test_module_model_rejects_invalid_feature_identity(names):
    from trace_agent.risk_skills.module_model import ModuleModel

    with pytest.raises(ValueError, match="feature names"):
        ModuleModel("prevalent", "M1", "D1", names, ProbabilityEstimator())


def test_module_artifact_round_trip(tmp_path):
    from trace_agent.risk_skills.module_model import ModuleModel

    model = ModuleModel("incident", "M1", "D1", ("a",), MarginEstimator())
    artifact = TraceModelArtifact("module", "incident", model, ("M1",), ("D1",))
    path = dump_trace_artifact(artifact, tmp_path / "module.joblib")
    loaded = load_trace_artifact(path, component="module", task="longitudinal")
    assert loaded.schema_version == 1
    assert loaded.model.feature_names == ("a",)
    np.testing.assert_allclose(loaded.model.predict(np.ones((2, 1))), -0.7)


@pytest.mark.parametrize(
    ("task", "modules", "endpoints", "message"),
    [
        ("prevalent", ("M1", "M2"), ("D1",), "one module"),
        ("prevalent", ("OTHER",), ("D1",), "identity"),
        ("longitudinal", ("M1",), ("D1",), "task"),
    ],
)
def test_module_artifact_rejects_mismatch(task, modules, endpoints, message):
    from trace_agent.risk_skills.module_model import ModuleModel

    model = ModuleModel("prevalent", "M1", "D1", ("a",), ProbabilityEstimator())
    with pytest.raises(ValueError, match=message):
        TraceModelArtifact("module", task, model, modules, endpoints)
