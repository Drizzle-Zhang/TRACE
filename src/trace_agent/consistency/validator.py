from __future__ import annotations

from collections.abc import Callable

import numpy as np

from trace_agent.consistency.evidence import compute_evidence_consistency, compute_rule_evidence_consistency
from trace_agent.consistency.feature import compute_feature_consistency
from trace_agent.consistency.kg import compute_kg_consistency
from trace_agent.consistency.strategy import apply_consistency_filter
from trace_agent.contracts.consistency import (
    ConsistencyResult,
    ConsistencyThresholds,
)
from trace_agent.contracts.scores import ModuleScoreBatch, RiskProfile
from trace_agent.contracts.task import TaskType
from trace_agent.knowledge.evidence_store import CompiledEvidenceRules


class ConsistencyValidator:
    """Compose KG, model-attribution and medical-evidence checks."""

    def __init__(
        self,
        *,
        kg_matrix: np.ndarray,
        thresholds: ConsistencyThresholds,
        medical_weights: np.ndarray | None = None,
        evidence_weights: np.ndarray | None = None,
        evidence_reliability: np.ndarray | None = None,
        evidence_rules: CompiledEvidenceRules | None = None,
        module_score_transform: Callable[[np.ndarray], np.ndarray] | None = None,
        prediction_horizon: str | None = None,
    ) -> None:
        kg = np.asarray(kg_matrix, dtype=np.float32)
        if kg.ndim != 2:
            raise ValueError("kg_matrix must be module-by-endpoint")
        self.kg_matrix = kg
        self.thresholds = thresholds
        self.medical_weights = (
            None if medical_weights is None else np.asarray(medical_weights, dtype=np.float32)
        )
        self.evidence_weights = (
            None if evidence_weights is None else np.asarray(evidence_weights, dtype=np.float32)
        )
        self.evidence_reliability = (
            None if evidence_reliability is None else np.asarray(evidence_reliability, dtype=np.float32)
        )
        self.module_score_transform = module_score_transform
        self.evidence_rules = evidence_rules
        self.prediction_horizon = prediction_horizon
        for name, values in (
            ("medical_weights", self.medical_weights),
            ("evidence_weights", self.evidence_weights),
            ("evidence_reliability", self.evidence_reliability),
        ):
            if values is not None and values.ndim != 2:
                raise ValueError(f"{name} must be endpoint-by-feature")

    def validate(
        self,
        *,
        batch: ModuleScoreBatch,
        profile: RiskProfile,
        contributions: np.ndarray | None = None,
    ) -> ConsistencyResult:
        if tuple(batch.endpoints) != tuple(profile.endpoints):
            raise ValueError("score batch and risk profile endpoint order differs")
        if batch.participant_ids.shape != profile.participant_ids.shape or not np.array_equal(
            batch.participant_ids, profile.participant_ids
        ):
            raise ValueError("score batch and risk profile participants differ")
        if self.kg_matrix.shape != (
            len(batch.modules),
            len(batch.endpoints),
        ):
            raise ValueError("kg_matrix does not match batch modules/endpoints")
        observed = np.asarray(profile.observed_modules, dtype=bool)
        if observed.shape != batch.availability.shape:
            raise ValueError("profile observed_modules does not match batch modules")
        module_scores = np.asarray(batch.scores, dtype=np.float32)
        predictions = profile.decision_scores
        if batch.task is TaskType.LONGITUDINAL:
            if self.module_score_transform is None:
                raise ValueError("longitudinal consistency requires a frozen module score scale")
            if not self.prediction_horizon or self.prediction_horizon not in profile.horizon_risks:
                raise ValueError("longitudinal consistency requires a named available probability horizon")
            predictions = profile.horizon_risks[self.prediction_horizon]
            if np.any(
                profile.valid_mask & (~np.isfinite(predictions) | (predictions < 0) | (predictions > 1))
            ):
                raise ValueError("longitudinal consistency horizon probabilities must be in [0, 1]")
        if self.module_score_transform is not None:
            module_scores = np.asarray(
                self.module_score_transform(module_scores),
                dtype=np.float32,
            )
            if module_scores.shape != batch.scores.shape:
                raise ValueError("module_score_transform must preserve score shape")
        active_mask = batch.pred_mask & observed[:, :, None]
        kg_scores = compute_kg_consistency(module_scores, active_mask, self.kg_matrix)

        neutral = np.full(profile.raw_scores.shape, 0.5, dtype=np.float32)
        feature_scores = neutral
        evidence_scores = neutral
        details: dict[str, object] = {}
        if contributions is not None:
            contribution_values = np.asarray(contributions, dtype=np.float32)
            expected_prefix = profile.raw_scores.shape
            if contribution_values.ndim != 3 or contribution_values.shape[:2] != expected_prefix:
                raise ValueError("contributions must be participant-by-endpoint-by-feature")
            if self.medical_weights is not None:
                feature_result = compute_feature_consistency(
                    contributions=contribution_values,
                    medical_weights=self.medical_weights,
                )
                feature_scores = feature_result.scores
                details["feature_matched_counts"] = feature_result.matched_counts
            if self.evidence_rules is not None or self.evidence_weights is not None:
                if self.evidence_rules is not None:
                    if self.evidence_rules.endpoints != batch.endpoints:
                        raise ValueError("evidence rules endpoint order does not match score batch")
                    evidence_result = compute_rule_evidence_consistency(
                        contributions=contribution_values, rules=self.evidence_rules
                    )
                else:
                    assert self.evidence_weights is not None
                    evidence_result = compute_evidence_consistency(
                        contributions=contribution_values,
                        evidence_weights=self.evidence_weights,
                        reliability=self.evidence_reliability,
                    )
                evidence_scores = evidence_result.scores
                details["evidence_support_scores"] = evidence_result.support_scores
                details["evidence_no_conflict_scores"] = evidence_result.no_conflict_scores
                details["evidence_matched_counts"] = evidence_result.matched_counts
        result = apply_consistency_filter(
            predictions=np.where(profile.valid_mask, predictions, np.nan),
            kg_scores=kg_scores,
            feature_scores=feature_scores,
            evidence_scores=evidence_scores,
            thresholds=self.thresholds,
        )
        return ConsistencyResult(
            kg_scores=result.kg_scores,
            feature_scores=result.feature_scores,
            evidence_scores=result.evidence_scores,
            combined_scores=result.combined_scores,
            strategy_masks=result.strategy_masks,
            thresholds=result.thresholds,
            details=details,
        )
