from __future__ import annotations

import numpy as np
import pytest

from trace_agent.contracts.scores import ModuleScoreBatch
from trace_agent.contracts.task import SplitRole, TaskType
from trace_agent.data.training_dataset import TrainingTargets, TrainingView
from trace_agent.integration.calibration import apply_endpoint_calibrators
from trace_agent.training.prevalent.calibration import (
    IdentityCalibrator,
    PrevalentCalibrationModel,
    fit_prevalent_calibration,
)


class MeanFusion:
    def predict(self, scores_nmd, *, pred_mask_nmd, used_mask_nm):
        active = np.asarray(pred_mask_nmd, dtype=bool) & np.asarray(used_mask_nm, dtype=bool)[:, :, None]
        return np.nanmean(np.where(active, scores_nmd, np.nan), axis=1)


def _view(role: SplitRole) -> TrainingView:
    labels = np.array(
        [[0, 1], [0, 1], [1, 1], [1, 1], [0, 1], [1, 1]],
        dtype=np.float32,
    )
    first = 0.1 + labels[:, 0] * 0.7
    second = np.full(labels.shape[0], 0.6, dtype=np.float32)
    probabilities = np.column_stack([first, second])
    scores = np.stack([probabilities, probabilities], axis=1)
    batch = ModuleScoreBatch(
        participant_ids=np.arange(labels.shape[0]),
        modules=("M1", "M2"),
        endpoints=("D1", "D2"),
        scores=scores,
        pred_mask=np.ones_like(scores, dtype=bool),
        availability=np.ones((labels.shape[0], 2), dtype=bool),
        task=TaskType.PREVALENT,
    )
    return TrainingView(
        role=role,
        row_indices=np.arange(labels.shape[0]),
        scores=batch,
        targets=TrainingTargets(
            labels=labels,
            label_mask=np.ones_like(labels, dtype=bool),
        ),
    )


def test_calibration_fits_only_from_validation_view() -> None:
    validation = _view(SplitRole.VALIDATION)

    result = fit_prevalent_calibration(
        MeanFusion(),
        validation,
        min_class_count=1,
    )

    assert isinstance(result.model, PrevalentCalibrationModel)
    assert result.source_rows.tolist() == validation.row_indices.tolist()
    assert set(result.model.calibrators) == {"D1", "D2"}
    assert isinstance(result.model.calibrators["D2"], IdentityCalibrator)
    assert set(result.metrics) == {"uncalibrated", "calibrated"}


@pytest.mark.parametrize("role", [SplitRole.TRAIN, SplitRole.TEST])
def test_calibration_rejects_nonvalidation_views(role: SplitRole) -> None:
    with pytest.raises(ValueError, match="validation view"):
        fit_prevalent_calibration(
            MeanFusion(),
            _view(role),
            min_class_count=1,
        )


def test_calibration_model_applies_endpoint_mapping() -> None:
    validation = _view(SplitRole.VALIDATION)
    result = fit_prevalent_calibration(
        MeanFusion(),
        validation,
        min_class_count=1,
    )
    raw = MeanFusion().predict(
        validation.scores.scores,
        pred_mask_nmd=validation.scores.pred_mask,
        used_mask_nm=validation.scores.availability,
    )

    calibrated = apply_endpoint_calibrators(
        raw,
        validation.scores.endpoints,
        result.model.calibrators,
    )

    assert calibrated is not None
    np.testing.assert_allclose(calibrated[:, 1], raw[:, 1])
    assert np.all((calibrated[:, 0] >= 0.0) & (calibrated[:, 0] <= 1.0))
