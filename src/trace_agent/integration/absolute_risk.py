from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoxAbsoluteRiskMapper:
    endpoints: tuple[str, ...] | Sequence[str]
    baseline_survival: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        endpoints = tuple(str(endpoint) for endpoint in self.endpoints)
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("absolute-risk endpoints must be unique")
        baseline: dict[str, np.ndarray] = {}
        for horizon, values in self.baseline_survival.items():
            array = np.asarray(values, dtype=np.float64).reshape(-1)
            if array.shape != (len(endpoints),):
                raise ValueError(f"baseline survival {horizon!r} must have shape {(len(endpoints),)}")
            if np.any(~np.isfinite(array)) or np.any((array <= 0.0) | (array > 1.0)):
                raise ValueError("baseline survival values must be finite in (0, 1]")
            baseline[str(horizon)] = array
        object.__setattr__(self, "endpoints", endpoints)
        object.__setattr__(self, "baseline_survival", baseline)

    def transform(
        self,
        log_risk: np.ndarray,
        endpoints: Sequence[str],
    ) -> dict[str, np.ndarray]:
        if tuple(str(endpoint) for endpoint in endpoints) != self.endpoints:
            raise ValueError("risk endpoints do not match absolute-risk mapper endpoints")
        risk = np.asarray(log_risk, dtype=np.float64)
        if risk.ndim != 2 or risk.shape[1] != len(self.endpoints):
            raise ValueError("log_risk must be participant-by-endpoint")
        relative_risk = np.exp(np.clip(risk, -30.0, 30.0))
        return {
            horizon: (-np.expm1(np.log(survival)[None, :] * relative_risk)).astype(np.float32)
            for horizon, survival in self.baseline_survival.items()
        }
