from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PrevalentModelAdapter:
    model: Any

    def predict(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values)
        if hasattr(self.model, "predict_proba"):
            probabilities = np.asarray(self.model.predict_proba(matrix), dtype=np.float64)
            if probabilities.ndim != 2 or probabilities.shape[0] != matrix.shape[0]:
                raise ValueError("prevalent model predict_proba returned an invalid shape")
            classes = np.asarray(getattr(self.model, "classes_", np.arange(probabilities.shape[1])))
            positive_columns = np.flatnonzero(classes == 1)
            column = int(positive_columns[0]) if positive_columns.size else probabilities.shape[1] - 1
            result = probabilities[:, column]
        elif hasattr(self.model, "decision_function"):
            margin = np.asarray(self.model.decision_function(matrix), dtype=np.float64).reshape(-1)
            result = np.empty_like(margin)
            positive_margin = margin >= 0
            result[positive_margin] = 1.0 / (1.0 + np.exp(-margin[positive_margin]))
            exp_margin = np.exp(margin[~positive_margin])
            result[~positive_margin] = exp_margin / (1.0 + exp_margin)
        else:
            result = np.asarray(self.model.predict(matrix), dtype=np.float64).reshape(-1)
        if result.shape != (matrix.shape[0],):
            raise ValueError("prevalent model returned an invalid number of scores")
        if np.any(~np.isfinite(result)):
            raise ValueError("prevalent model returned non-finite probabilities")
        if np.any((result < 0.0) | (result > 1.0)):
            raise ValueError("prevalent model probabilities must be in [0, 1]")
        return result.astype(np.float32, copy=False)
