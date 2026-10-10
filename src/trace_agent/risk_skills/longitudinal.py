from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LongitudinalModelAdapter:
    model: Any

    def predict(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values)
        result = np.asarray(self.model.predict(matrix), dtype=np.float64).reshape(-1)
        if result.shape != (matrix.shape[0],):
            raise ValueError("longitudinal model returned an invalid number of scores")
        if np.any(~np.isfinite(result)):
            raise ValueError("longitudinal model returned non-finite risk scores")
        return result.astype(np.float32, copy=False)
