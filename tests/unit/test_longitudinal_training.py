from __future__ import annotations

import numpy as np
import pytest

from trace_agent.contracts.scores import ModuleScoreBatch
from trace_agent.contracts.task import SplitRole, TaskType
from trace_agent.data.training_dataset import TrainingTargets, TrainingView
from trace_agent.integration.longitudinal_cox import LongitudinalRiskIntegrator
from trace_agent.training.longitudinal.integration import (
    LongitudinalFusionModel,
    LongitudinalIntegrationConfig,
    breslow_cumulative_hazard,
    train_longitudinal_integration,
)


def _view(role: SplitRole, *, offset: int, n: int) -> TrainingView:
    event = np.asarray(
        [[index % 2, (index + 1) % 2] for index in range(n)],
        dtype=np.float32,
    )
    time = np.asarray(
        [[2.0 + index, 3.0 + index] for index in range(n)],
        dtype=np.float32,
    )
    scores = np.empty((n, 2, 2), dtype=np.float32)
    scores[:, 0, 0] = 0.2 * event[:, 0] + 0.01 * time[:, 0]
    scores[:, 1, 0] = 0.3 * event[:, 0] + 0.02 * time[:, 0]
    scores[:, 0, 1] = 0.2 * event[:, 1] + 0.01 * time[:, 1]
    scores[:, 1, 1] = 0.3 * event[:, 1] + 0.02 * time[:, 1]
    batch = ModuleScoreBatch(
        participant_ids=np.arange(offset, offset + n),
        modules=("M1", "M2"),
        endpoints=("D1", "D2"),
        scores=scores,
        pred_mask=np.ones_like(scores, dtype=bool),
        availability=np.ones((n, 2), dtype=bool),
        task=TaskType.LONGITUDINAL,
    )
    return TrainingView(
        role=role,
        row_indices=np.arange(offset, offset + n),
        scores=batch,
        targets=TrainingTargets(
            labels=event,
            label_mask=np.ones_like(event, dtype=bool),
            event_time=time,
            signed_time=np.where(event > 0, time, -time),
        ),
    )


def test_breslow_cumulative_hazard_uses_event_risk_sets() -> None:
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1, 0, 1])
    log_risk = np.zeros(3)

    cumulative = breslow_cumulative_hazard(
        time,
        event,
        log_risk,
        horizon=3.0,
    )

    assert cumulative == pytest.approx(1.0 / 3.0 + 1.0)


def test_longitudinal_fusion_trains_and_predicts_partial_masks() -> None:
    train = _view(SplitRole.TRAIN, offset=0, n=12)
    validation = _view(SplitRole.VALIDATION, offset=12, n=6)

    result = train_longitudinal_integration(
        train,
        validation,
        config=LongitudinalIntegrationConfig(
            seed=4,
            subsets_per_participant=1,
            min_events=2,
            n_estimators=8,
            max_depth=2,
            n_jobs=1,
            horizons=(5.0, 10.0),
        ),
    )
    used = validation.scores.availability.copy()
    used[:, 1:] = False
    raw = result.model.predict(
        validation.scores.scores,
        pred_mask_nmd=validation.scores.pred_mask,
        used_mask_nm=used,
        standardized=False,
    )

    assert isinstance(result.model, LongitudinalFusionModel)
    assert raw.shape == (validation.scores.n_participants, 2)
    assert result.model.standardize(raw).shape == raw.shape
    assert set(result.model.baseline_survival) == {"5y", "10y"}
    assert result.train_row_indices.tolist() == train.row_indices.tolist()
    assert result.metrics["participants"] == validation.scores.n_participants


def test_longitudinal_integrator_uses_embedded_baseline_survival() -> None:
    train = _view(SplitRole.TRAIN, offset=0, n=12)
    validation = _view(SplitRole.VALIDATION, offset=12, n=4)
    model = train_longitudinal_integration(
        train,
        validation,
        config=LongitudinalIntegrationConfig(
            min_events=2,
            n_estimators=4,
            max_depth=2,
            horizons=(5.0,),
        ),
    ).model

    profile = LongitudinalRiskIntegrator(fusion=model).integrate(
        validation.scores,
        validation.scores.availability,
    )

    assert set(profile.horizon_risks) == {"5y"}
    assert profile.horizon_risks["5y"].shape == (4, 2)


def test_longitudinal_integration_rejects_test_view() -> None:
    with pytest.raises(ValueError, match="train view"):
        train_longitudinal_integration(
            _view(SplitRole.TEST, offset=0, n=6),
            _view(SplitRole.VALIDATION, offset=6, n=4),
            config=LongitudinalIntegrationConfig(
                min_events=1,
                n_estimators=4,
            ),
        )
