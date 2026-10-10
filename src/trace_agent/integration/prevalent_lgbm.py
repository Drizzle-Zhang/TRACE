from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from trace_agent.contracts.scores import ModuleScoreBatch, RiskProfile
from trace_agent.contracts.task import TaskType
from trace_agent.integration.calibration import apply_endpoint_calibrators


@dataclass
class PrevalentRiskIntegrator:
    fusion: Any
    calibrators: Mapping[str, Any] = field(default_factory=dict)

    def integrate(
        self,
        batch: ModuleScoreBatch,
        observed_modules: np.ndarray,
    ) -> RiskProfile:
        if batch.task is not TaskType.PREVALENT:
            raise ValueError("prevalent integrator requires a prevalent ModuleScoreBatch")
        observed = np.asarray(observed_modules, dtype=bool)
        expected = (batch.n_participants, len(batch.modules))
        if observed.shape != expected:
            raise ValueError(f"observed_modules shape must be {expected}, got {observed.shape}")
        if np.any(observed & ~batch.availability):
            raise ValueError("observed modules must be available")

        probabilities = np.full(
            (batch.n_participants, len(batch.endpoints)),
            np.nan,
            dtype=np.float32,
        )
        valid_mask = np.any(batch.pred_mask & observed[:, :, None], axis=1)
        logits = np.full_like(probabilities, np.nan)
        for row in range(batch.n_participants):
            if hasattr(self.fusion, "fuse_logits"):
                logits[row] = np.asarray(
                    self.fusion.fuse_logits(
                        batch.scores[row],
                        observed[row],
                        pred_mask_md=batch.pred_mask[row],
                    ),
                    dtype=np.float32,
                )
            probabilities[row] = np.asarray(
                self.fusion.fuse(
                    batch.scores[row],
                    observed[row],
                    pred_mask_md=batch.pred_mask[row],
                ),
                dtype=np.float32,
            )
        calibrated = apply_endpoint_calibrators(
            probabilities,
            batch.endpoints,
            self.calibrators,
        )
        return RiskProfile(
            participant_ids=batch.participant_ids,
            endpoints=batch.endpoints,
            raw_scores=probabilities,
            valid_mask=valid_mask,
            observed_modules=observed,
            task=TaskType.PREVALENT,
            calibrated_scores=calibrated,
            metadata={"logits": logits},
        )
