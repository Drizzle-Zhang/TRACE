from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sklearn.isotonic import IsotonicRegression

from trace_agent.contracts.task import SplitRole, TaskType
from trace_agent.data.training_dataset import TrainingView
from trace_agent.integration.calibration import apply_endpoint_calibrators
from trace_agent.training.common import (
    masked_classification_metrics,
    require_view_role,
)


class PrevalentFusion(Protocol):
    def predict(
        self,
        scores_nmd: np.ndarray,
        *,
        pred_mask_nmd: np.ndarray,
        used_mask_nm: np.ndarray,
    ) -> np.ndarray: ...


@dataclass
class IdentityCalibrator:
    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32).reshape(-1)


@dataclass
class PrevalentCalibrationModel:
    endpoints: tuple[str, ...]
    calibrators: Mapping[str, Any]

    def __post_init__(self) -> None:
        self.endpoints = tuple(str(value) for value in self.endpoints)
        self.calibrators = dict(self.calibrators)
        if set(self.calibrators) != set(self.endpoints):
            raise ValueError("calibrators must contain every endpoint exactly once")


@dataclass(frozen=True)
class CalibrationResult:
    model: PrevalentCalibrationModel
    source_rows: np.ndarray
    metrics: Mapping[str, Any]


def fit_prevalent_calibration(
    fusion: PrevalentFusion,
    validation: TrainingView,
    *,
    min_class_count: int = 10,
) -> CalibrationResult:
    require_view_role(validation, SplitRole.VALIDATION)
    if validation.scores.task is not TaskType.PREVALENT:
        raise ValueError("prevalent calibration requires prevalent score data")
    if min_class_count < 1:
        raise ValueError("min_class_count must be positive")

    probability = np.asarray(
        fusion.predict(
            validation.scores.scores,
            pred_mask_nmd=validation.scores.pred_mask,
            used_mask_nm=validation.scores.availability,
        ),
        dtype=np.float32,
    )
    expected = (
        validation.scores.n_participants,
        len(validation.scores.endpoints),
    )
    if probability.shape != expected:
        raise ValueError(f"prevalent fusion returned {probability.shape}, expected {expected}")

    active = np.any(validation.scores.pred_mask & validation.scores.availability[:, :, None], axis=1)
    observed_labels = validation.targets.label_mask & active
    calibrators: dict[str, Any] = {}
    for endpoint, name in enumerate(validation.scores.endpoints):
        selected = (
            observed_labels[:, endpoint]
            & np.isfinite(validation.targets.labels[:, endpoint])
            & np.isfinite(probability[:, endpoint])
        )
        labels = validation.targets.labels[selected, endpoint]
        negative = int(np.sum(labels == 0))
        positive = int(np.sum(labels == 1))
        if min(negative, positive) < min_class_count:
            calibrators[name] = IdentityCalibrator()
            continue
        calibrators[name] = IsotonicRegression(out_of_bounds="clip").fit(
            probability[selected, endpoint],
            labels,
        )

    model = PrevalentCalibrationModel(
        endpoints=validation.scores.endpoints,
        calibrators=calibrators,
    )
    calibrated = apply_endpoint_calibrators(
        probability,
        validation.scores.endpoints,
        model.calibrators,
    )
    if calibrated is None:
        raise RuntimeError("calibration mapping unexpectedly produced no output")
    return CalibrationResult(
        model=model,
        source_rows=validation.row_indices.copy(),
        metrics={
            "uncalibrated": masked_classification_metrics(
                probability,
                validation.targets.labels,
                observed_labels,
            ),
            "calibrated": masked_classification_metrics(
                calibrated,
                validation.targets.labels,
                observed_labels,
            ),
        },
    )
