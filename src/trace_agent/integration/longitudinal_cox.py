from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from trace_agent.contracts.scores import ModuleScoreBatch, RiskProfile
from trace_agent.contracts.task import TaskType
from trace_agent.integration.absolute_risk import CoxAbsoluteRiskMapper


@dataclass
class LongitudinalRiskIntegrator:
    fusion: Any
    absolute_risk_mapper: CoxAbsoluteRiskMapper | None = None

    def integrate(
        self,
        batch: ModuleScoreBatch,
        observed_modules: np.ndarray,
    ) -> RiskProfile:
        if batch.task is not TaskType.LONGITUDINAL:
            raise ValueError("longitudinal integrator requires a longitudinal ModuleScoreBatch")
        observed = np.asarray(observed_modules, dtype=bool)
        expected = (batch.n_participants, len(batch.modules))
        if observed.shape != expected:
            raise ValueError(f"observed_modules shape must be {expected}, got {observed.shape}")
        if np.any(observed & ~batch.availability):
            raise ValueError("observed modules must be available")

        raw = np.asarray(
            self.fusion.predict(
                batch.scores,
                pred_mask_nmd=batch.pred_mask,
                used_mask_nm=observed,
                standardized=False,
            ),
            dtype=np.float32,
        )
        expected_risk = (batch.n_participants, len(batch.endpoints))
        if raw.shape != expected_risk:
            raise ValueError(f"longitudinal fusion returned {raw.shape}, expected {expected_risk}")
        standardized = np.asarray(self.fusion.standardize(raw), dtype=np.float32)
        valid_mask = np.any(batch.pred_mask & observed[:, :, None], axis=1) & np.isfinite(raw)
        mapper = self.absolute_risk_mapper
        if mapper is None and getattr(self.fusion, "baseline_survival", None):
            mapper = CoxAbsoluteRiskMapper(
                endpoints=batch.endpoints,
                baseline_survival=self.fusion.baseline_survival,
            )
        horizon_risks = {} if mapper is None else mapper.transform(raw, batch.endpoints)
        return RiskProfile(
            participant_ids=batch.participant_ids,
            endpoints=batch.endpoints,
            raw_scores=raw,
            valid_mask=valid_mask,
            observed_modules=observed,
            task=TaskType.LONGITUDINAL,
            horizon_risks=horizon_risks,
            metadata={"standardized_scores": standardized},
        )
