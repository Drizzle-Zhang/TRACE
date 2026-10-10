from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from trace_agent.contracts.task import TaskType
from trace_agent.risk_skills.longitudinal import LongitudinalModelAdapter
from trace_agent.risk_skills.model_bank import JoblibModelBank
from trace_agent.risk_skills.prevalent import PrevalentModelAdapter
from trace_agent.risk_skills.registry import RiskSkillRegistry
from trace_agent.risk_skills.scorer import score_feature_modules


class ProbabilityModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        positive = np.asarray(values[:, 0], dtype=np.float64)
        return np.column_stack([1.0 - positive, positive])


class RiskModel:
    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values[:, 0], dtype=np.float64) * 2.0


def test_model_adapters_preserve_task_specific_score_semantics() -> None:
    prevalent = PrevalentModelAdapter(ProbabilityModel())
    longitudinal = LongitudinalModelAdapter(RiskModel())
    values = np.array([[0.2], [0.7]])

    np.testing.assert_allclose(prevalent.predict(values), [0.2, 0.7])
    np.testing.assert_allclose(longitudinal.predict(values), [0.4, 1.4])


def test_joblib_model_bank_supports_calibrate_and_calibration_suffixes(tmp_path: Path) -> None:
    module_dir = tmp_path / "M1"
    module_dir.mkdir()
    model_path = module_dir / "D1_sublevel_RS_model_calibration.joblib"
    joblib.dump(ProbabilityModel(), model_path)

    bank = JoblibModelBank(tmp_path)

    assert bank.resolve_path("M1", "D1") == model_path
    assert isinstance(bank.load("M1", "D1"), ProbabilityModel)


def test_registry_scorer_builds_masked_module_score_batch() -> None:
    registry = RiskSkillRegistry()
    registry.register(
        task=TaskType.PREVALENT,
        module="M1",
        endpoint="D1",
        skill=PrevalentModelAdapter(ProbabilityModel()),
    )
    batch = score_feature_modules(
        participant_ids=np.array([1, 2]),
        modules=("M1", "M2"),
        endpoints=("D1",),
        features_by_module={
            "M1": np.array([[0.2], [0.7]]),
            "M2": np.array([[9.0], [9.0]]),
        },
        availability=np.array([[1, 1], [1, 0]], dtype=bool),
        registry=registry,
        task=TaskType.PREVALENT,
    )

    np.testing.assert_allclose(batch.scores[:, 0, 0], [0.2, 0.7])
    assert np.isnan(batch.scores[:, 1, 0]).all()
    assert batch.pred_mask[:, 0, 0].all()
    assert not batch.pred_mask[:, 1, 0].any()
