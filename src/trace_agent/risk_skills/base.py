from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RiskSkill(Protocol):
    def predict(self, values: np.ndarray) -> np.ndarray:
        """Return one score per participant."""
