"""Finite priors for unobserved endpoints are not assessment observations."""

import numpy as np

from trace_agent.contracts.scores import ModuleScoreBatch
from trace_agent.contracts.task import SplitRole, TaskType
from trace_agent.data.training_dataset import TrainingTargets, TrainingView
from trace_agent.training.longitudinal.integration import (
    ConstantRiskEstimator,
    LongitudinalFusionModel,
    _validation_metrics,
)
from trace_agent.training.prevalent.calibration import fit_prevalent_calibration
from trace_agent.training.prevalent.integration import (
    ConstantProbabilityEstimator,
    PrevalentFusionModel,
    PrevalentIntegrationConfig,
    train_prevalent_integration,
)


def _view(task, role, *, unobserved=False):
    labels = np.array([[1, 0], [1, 1], [0, 0], [1, 1]], dtype=np.float32)
    scores = np.tile((0.2 + 0.6 * labels)[:, None, :], (1, 2, 1))
    pred_mask = np.ones(scores.shape, dtype=bool)
    if unobserved:
        pred_mask[:2, :, 0] = False
        scores[~pred_mask] = np.nan
    batch = ModuleScoreBatch(
        participant_ids=np.arange(4),
        modules=("M1", "M2"),
        endpoints=("D1", "D2"),
        scores=scores,
        pred_mask=pred_mask,
        availability=np.ones((4, 2), dtype=bool),
        task=task,
    )
    time = np.full(labels.shape, 12.0, dtype=np.float32)
    time[labels == 1] = 2.0
    return TrainingView(
        role=role,
        row_indices=np.arange(4),
        scores=batch,
        targets=TrainingTargets(
            labels=labels,
            label_mask=np.ones(labels.shape, dtype=bool),
            event_time=time if task is TaskType.LONGITUDINAL else None,
            signed_time=np.where(labels == 1, time, -time) if task is TaskType.LONGITUDINAL else None,
        ),
    )


def test_isotonic_fit_excludes_unobserved_endpoints_with_finite_prior():
    validation = _view(TaskType.PREVALENT, SplitRole.VALIDATION, unobserved=True)
    fusion = PrevalentFusionModel(("M1", "M2"), ("D1", "D2"), (ConstantProbabilityEstimator(0.4),) * 2)
    before = validation.targets.label_mask.copy()
    result = fit_prevalent_calibration(fusion, validation, min_class_count=1)
    np.testing.assert_allclose(result.model.calibrators["D1"].predict([0.4]), [0.5])
    np.testing.assert_array_equal(validation.targets.label_mask, before)


def test_calibration_metrics_exclude_unobserved_finite_prior():
    validation = _view(TaskType.PREVALENT, SplitRole.VALIDATION, unobserved=True)
    fusion = PrevalentFusionModel(("M1", "M2"), ("D1", "D2"), (ConstantProbabilityEstimator(0.4),) * 2)
    result = fit_prevalent_calibration(fusion, validation, min_class_count=1)
    for name in ("uncalibrated", "calibrated"):
        assert result.metrics[name]["n_valid"] == 6
        assert result.metrics[name]["endpoints"][0]["n"] == 2


def test_prevalent_validation_metrics_exclude_unobserved_finite_prior():
    train = _view(TaskType.PREVALENT, SplitRole.TRAIN)
    validation = _view(TaskType.PREVALENT, SplitRole.VALIDATION, unobserved=True)
    result = train_prevalent_integration(
        train,
        validation,
        config=PrevalentIntegrationConfig(subsets_per_participant=0, min_class_count=20),
    )
    assert result.metrics["n_valid"] == 6
    assert result.metrics["endpoints"][0]["n"] == 2
    # The no-observation prior and its fitting policy remain available internally.
    prediction = result.model.predict(
        validation.scores.scores,
        pred_mask_nmd=validation.scores.pred_mask,
        used_mask_nm=validation.scores.availability,
    )
    assert np.isfinite(prediction[:2, 0]).all()


def test_longitudinal_validation_horizons_exclude_unobserved_finite_prior():
    validation = _view(TaskType.LONGITUDINAL, SplitRole.VALIDATION, unobserved=True)
    fusion = LongitudinalFusionModel(
        modules=("M1", "M2"),
        diseases=("D1", "D2"),
        estimators=(ConstantRiskEstimator(),) * 2,
        risk_loc=np.zeros(2),
        risk_scale=np.ones(2),
        baseline_survival={"5y": np.array([0.8, 0.7]), "10y": np.array([0.6, 0.5])},
    )
    result = _validation_metrics(fusion, validation)
    for metrics in result["horizons"].values():
        assert metrics["n_valid"] == 6
        assert metrics["endpoints"][0]["n"] == 2
