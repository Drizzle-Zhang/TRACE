from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trace_agent.contracts.scores import RiskProfile
from trace_agent.contracts.task import TaskType


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


@dataclass(frozen=True)
class KnowledgeCandidateRanker:
    kg_matrix: np.ndarray
    burden: np.ndarray
    cost_gamma: float = 1.0
    cost_epsilon: float = 1e-6
    urgency_threshold: float = 0.5
    urgency_sigma: float = 0.15

    def __post_init__(self) -> None:
        kg = np.asarray(self.kg_matrix, dtype=np.float32)
        burden = np.asarray(self.burden, dtype=np.float32).reshape(-1)
        if kg.ndim != 2:
            raise ValueError("kg_matrix must be module-by-endpoint")
        if burden.shape != (kg.shape[0],):
            raise ValueError("burden must contain one value per module")
        if not np.isfinite(kg).all():
            raise ValueError("kg_matrix must contain only finite values")
        if np.any(~np.isfinite(burden)) or np.any(burden < 0):
            raise ValueError("burden must be finite and non-negative")
        if not np.isfinite(self.cost_gamma):
            raise ValueError("cost_gamma must be finite")
        if self.cost_epsilon <= 0 or not np.isfinite(self.cost_epsilon):
            raise ValueError("cost_epsilon must be finite and positive")
        if self.urgency_sigma <= 0 or not np.isfinite(self.urgency_sigma):
            raise ValueError("urgency_sigma must be finite and positive")
        object.__setattr__(self, "kg_matrix", kg)
        object.__setattr__(self, "burden", burden)

    def rank(
        self,
        *,
        profile: RiskProfile,
        pred_mask_md: np.ndarray,
        availability: np.ndarray,
        used_modules: np.ndarray,
        top_k: int,
    ) -> np.ndarray:
        if profile.n_participants != 1:
            raise ValueError("candidate ranking expects one participant")
        modules, endpoints = self.kg_matrix.shape
        if len(profile.endpoints) != endpoints:
            raise ValueError("knowledge matrix endpoints do not match the risk profile")
        pred_mask = np.asarray(pred_mask_md, dtype=bool)
        if pred_mask.shape != (modules, endpoints):
            raise ValueError(f"pred_mask_md shape must be {(modules, endpoints)}, " f"got {pred_mask.shape}")
        available = np.asarray(availability, dtype=bool).reshape(-1)
        used = np.asarray(used_modules, dtype=bool).reshape(-1)
        if available.shape != (modules,) or used.shape != (modules,):
            raise ValueError("availability and used_modules must align with modules")
        candidates = np.flatnonzero(available & ~used & np.any(pred_mask, axis=1))
        if candidates.size == 0 or top_k <= 0:
            return np.empty(0, dtype=np.int64)

        signal = np.asarray(profile.raw_scores[0], dtype=np.float64)
        if profile.task is TaskType.PREVALENT:
            logits = profile.metadata.get("logits")
            if logits is not None:
                signal = np.asarray(logits, dtype=np.float64).reshape(profile.raw_scores.shape)[0]
        else:
            standardized = profile.metadata.get("standardized_scores")
            if standardized is not None:
                signal = np.asarray(standardized, dtype=np.float64).reshape(profile.raw_scores.shape)[0]
        probabilities = _sigmoid(np.clip(signal, -20.0, 20.0))
        urgency = np.exp(-np.abs(probabilities - float(self.urgency_threshold)) / float(self.urgency_sigma))
        uncertainty = 4.0 * probabilities * (1.0 - probabilities)
        coverage = (
            np.mean(pred_mask[used].astype(np.float64), axis=0)
            if np.any(used)
            else np.zeros(endpoints, dtype=np.float64)
        )
        weights = urgency * uncertainty * (1.0 - np.clip(coverage, 0.0, 1.0))
        # Current prediction validity is not acquisition eligibility: an unused
        # module may provide the first evidence for an endpoint.
        weights = np.where(np.isfinite(signal) & np.isfinite(weights), weights, 0.0)
        if float(np.sum(weights)) <= 0:
            if profile.task is TaskType.LONGITUDINAL:
                order = np.argsort(self.burden[candidates], kind="stable")
                return candidates[order[: int(top_k)]]
            # Preserve native prevalent fallback; the frozen survival cost
            # fallback does not define the prevalent candidate algorithm.
            weights = np.asarray(profile.valid_mask[0], dtype=np.float64)
            if float(np.sum(weights)) <= 0:
                weights = np.ones(endpoints, dtype=np.float64)
        weights /= float(np.sum(weights))

        gain = np.sum(
            self.kg_matrix[candidates] * pred_mask[candidates].astype(np.float32) * weights[None, :],
            axis=1,
        )
        if profile.task is TaskType.LONGITUDINAL and float(np.max(np.abs(gain))) <= 1e-12:
            order = np.argsort(self.burden[candidates], kind="stable")
            return candidates[order[: int(top_k)]]
        denominator = np.power(
            np.maximum(self.burden[candidates], 0.0) + float(self.cost_epsilon),
            float(self.cost_gamma),
        )
        score = np.asarray(gain, dtype=np.float64) / denominator
        score = np.where(np.isfinite(score), score, -np.inf)
        order = np.lexsort(
            (
                candidates,
                self.burden[candidates].astype(np.float64),
                -score,
            )
        )
        return candidates[order[: min(int(top_k), candidates.size)]].astype(np.int64, copy=False)
