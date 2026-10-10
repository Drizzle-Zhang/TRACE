"""Native model identity, selected feature order, and prediction scale."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from trace_agent.contracts.task import TaskType


@dataclass(frozen=True)
class ModuleModel:
    task: TaskType | str
    module: str
    endpoint: str
    feature_names: tuple[str, ...]
    estimator: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", TaskType.parse(self.task))
        names = tuple(str(name) for name in self.feature_names)
        if not names or any(not name.strip() for name in names) or len(set(names)) != len(names):
            raise ValueError("module feature names must be non-empty and unique")
        if not self.module.strip() or not self.endpoint.strip() or self.estimator is None:
            raise ValueError("module model requires identity and estimator")
        object.__setattr__(self, "feature_names", names)

    @property
    def score_scale(self) -> str:
        return "probability" if self.task is TaskType.PREVALENT else "log_risk"

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        """Predict selected columns; the scorer separately masks missing rows."""
        values = np.asarray(matrix, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("module feature width mismatch")
        if np.isinf(values).any():
            raise ValueError("module features contain infinity")
        if self.task is TaskType.PREVALENT:
            probabilities = np.asarray(self.estimator.predict_proba(values), dtype=float)
            classes = np.asarray(self.estimator.classes_)
            positions = np.flatnonzero(classes == 1)
            if positions.size != 1 or probabilities.shape != (values.shape[0], classes.size):
                raise ValueError("module probability output mismatch")
            predicted = probabilities[:, int(positions[0])]
            if np.any(np.isfinite(predicted) & ((predicted < 0) | (predicted > 1))):
                raise ValueError("module probability outside [0,1]")
        else:
            # survival:cox's default is exp(margin), not the log-risk contract.
            predicted = np.asarray(self.estimator.predict(values, output_margin=True), dtype=float)
        if predicted.shape != (values.shape[0],):
            raise ValueError("module prediction row count mismatch")
        return predicted.astype(np.float32)
