from __future__ import annotations

import numpy as np

from trace_agent.contracts.scores import ModuleScoreBatch
from trace_agent.contracts.task import TaskType
from trace_agent.integration.absolute_risk import CoxAbsoluteRiskMapper
from trace_agent.integration.longitudinal_cox import LongitudinalRiskIntegrator
from trace_agent.integration.prevalent_lgbm import PrevalentRiskIntegrator


class FakePrevalentFusion:
    def fuse(self, probs_md, mask_m, pred_mask_md=None):
        active = np.asarray(mask_m, dtype=bool)[:, None] & np.asarray(pred_mask_md, dtype=bool)
        values = np.where(active, probs_md, np.nan)
        return np.nanmean(values, axis=0)


class ShiftCalibrator:
    def predict(self, values):
        return np.clip(np.asarray(values) + 0.1, 0.0, 1.0)


class FakeLongitudinalFusion:
    def predict(self, scores_nmd, *, pred_mask_nmd=None, used_mask_nm=None, standardized=False):
        active = np.asarray(pred_mask_nmd, dtype=bool) & np.asarray(used_mask_nm, dtype=bool)[:, :, None]
        values = np.where(active, scores_nmd, np.nan)
        result = np.nanmean(values, axis=1)
        return result / 2.0 if standardized else result

    def standardize(self, risk_nd):
        return np.asarray(risk_nd) / 2.0


def _batch(task: TaskType) -> ModuleScoreBatch:
    return ModuleScoreBatch(
        participant_ids=np.array([1, 2]),
        modules=("M1", "M2"),
        endpoints=("D1",),
        scores=np.array([[[0.2], [0.8]], [[0.4], [0.6]]], dtype=np.float32),
        pred_mask=np.ones((2, 2, 1), dtype=bool),
        availability=np.ones((2, 2), dtype=bool),
        task=task,
    )


def test_prevalent_integrator_applies_observed_mask_and_calibration() -> None:
    integrator = PrevalentRiskIntegrator(
        fusion=FakePrevalentFusion(),
        calibrators={"D1": ShiftCalibrator()},
    )
    profile = integrator.integrate(
        _batch(TaskType.PREVALENT),
        observed_modules=np.array([[1, 0], [1, 1]], dtype=bool),
    )

    np.testing.assert_allclose(profile.raw_scores[:, 0], [0.2, 0.5])
    np.testing.assert_allclose(profile.calibrated_scores[:, 0], [0.3, 0.6])


def test_cox_absolute_risk_mapper_and_longitudinal_integrator() -> None:
    mapper = CoxAbsoluteRiskMapper(
        endpoints=("D1",),
        baseline_survival={"10y": np.array([0.8], dtype=np.float32)},
    )
    integrator = LongitudinalRiskIntegrator(
        fusion=FakeLongitudinalFusion(),
        absolute_risk_mapper=mapper,
    )
    profile = integrator.integrate(
        _batch(TaskType.LONGITUDINAL),
        observed_modules=np.ones((2, 2), dtype=bool),
    )

    np.testing.assert_allclose(profile.raw_scores[:, 0], [0.5, 0.5])
    expected = 1.0 - np.power(0.8, np.exp(0.5))
    np.testing.assert_allclose(profile.horizon_risks["10y"][:, 0], expected)
    np.testing.assert_allclose(profile.metadata["standardized_scores"][:, 0], [0.25, 0.25])
