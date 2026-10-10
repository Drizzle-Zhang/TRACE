from __future__ import annotations

from typing import Protocol

import numpy as np

from trace_agent.contracts.scores import ModuleScoreBatch, RiskProfile


class RiskIntegrator(Protocol):
    def integrate(
        self,
        batch: ModuleScoreBatch,
        observed_modules: np.ndarray,
    ) -> RiskProfile:
        """Integrate currently observed module scores into a risk profile."""
