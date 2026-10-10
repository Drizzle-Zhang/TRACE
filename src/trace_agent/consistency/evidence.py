from __future__ import annotations

import numpy as np

from trace_agent.contracts.consistency import EvidenceConsistencyResult
from trace_agent.knowledge.evidence_store import CompiledEvidenceRules


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def compute_evidence_consistency(
    *,
    contributions: np.ndarray,
    evidence_weights: np.ndarray,
    match_mask: np.ndarray | None = None,
    reliability: np.ndarray | None = None,
    scale: float = 1.5,
) -> EvidenceConsistencyResult:
    """Score guideline/evidence support and conflict for each endpoint."""

    contrib = np.asarray(contributions, dtype=np.float32)
    weights = np.asarray(evidence_weights, dtype=np.float32)
    if contrib.ndim != 3 or weights.shape != contrib.shape[1:]:
        raise ValueError("contributions and evidence_weights must align by endpoint/feature")
    matches = weights != 0 if match_mask is None else np.asarray(match_mask, dtype=bool)
    if matches.shape != weights.shape:
        raise ValueError("match_mask must match evidence_weights")
    reliability_values = (
        np.ones_like(weights, dtype=np.float32)
        if reliability is None
        else np.asarray(reliability, dtype=np.float32)
    )
    if reliability_values.shape != weights.shape:
        raise ValueError("reliability must match evidence_weights")
    if (
        np.any(~np.isfinite(reliability_values))
        or np.any(reliability_values < 0)
        or np.any(reliability_values > 1)
    ):
        raise ValueError("reliability must be finite and in [0, 1]")
    finite_contrib = np.isfinite(contrib)
    valid = matches[None, :, :] & finite_contrib & np.isfinite(weights)[None, :, :]
    contribution_scale = np.max(np.where(finite_contrib, np.abs(contrib), 0.0), axis=2, keepdims=True)
    weight_scale = np.max(np.where(matches, np.abs(weights), 0.0), axis=1, keepdims=True)
    normalized_contrib = np.divide(
        contrib,
        contribution_scale,
        out=np.zeros_like(contrib),
        where=finite_contrib & (contribution_scale > 0),
    )
    normalized_weights = np.divide(
        weights,
        weight_scale,
        out=np.zeros_like(weights),
        where=weight_scale > 0,
    )
    weighted = np.where(
        valid,
        normalized_contrib * normalized_weights[None, :, :] * reliability_values[None, :, :],
        0.0,
    )
    raw = np.sum(weighted, axis=2)
    support = np.sum(np.maximum(weighted, 0.0), axis=2)
    conflict = np.sum(np.maximum(-weighted, 0.0), axis=2)
    matched_counts = np.sum(valid, axis=2).astype(np.int32)
    has_matches = matched_counts > 0
    scores = np.where(has_matches, _sigmoid(float(scale) * raw), 0.5)
    support_scores = np.where(
        has_matches & (support > 0),
        _sigmoid(float(scale) * support),
        0.5,
    )
    no_conflict_scores = np.where(
        has_matches & (conflict > 0),
        _sigmoid(-float(scale) * conflict),
        0.5,
    )
    return EvidenceConsistencyResult(
        scores=scores.astype(np.float32),
        support_scores=support_scores.astype(np.float32),
        no_conflict_scores=no_conflict_scores.astype(np.float32),
        matched_counts=matched_counts,
    )


def compute_rule_evidence_consistency(
    *,
    contributions: np.ndarray,
    rules: CompiledEvidenceRules,
    scale: float = 1.5,
) -> EvidenceConsistencyResult:
    """Score rule matches before aggregating support and conflict per endpoint."""
    contrib = np.asarray(contributions, dtype=np.float64)
    if contrib.ndim != 3 or contrib.shape[1:] != (len(rules.endpoints), len(rules.features)):
        raise ValueError("contributions must align with rule endpoints and features")
    finite = np.isfinite(contrib)
    contribution_scale = np.max(np.where(finite, np.abs(contrib), 0.0), axis=2, keepdims=True)
    normalized = np.divide(
        contrib,
        contribution_scale,
        out=np.zeros_like(contrib),
        where=finite & (contribution_scale > 0),
    )
    raw = np.zeros(contrib.shape[:2], dtype=np.float64)
    support = np.zeros_like(raw)
    conflict = np.zeros_like(raw)
    matched_counts = np.zeros(raw.shape, dtype=np.int32)
    for endpoint in range(len(rules.endpoints)):
        selected = rules.endpoint_indices == endpoint
        if not np.any(selected):
            continue
        # All endpoint rules set the weight scale, even if their feature is
        # missing for a participant or absent from the supplied feature axis.
        weight_scale = float(np.max(np.abs(rules.weights[selected]))) or 1.0
        selected &= rules.feature_indices >= 0
        features = rules.feature_indices[selected]
        valid = finite[:, endpoint, features]
        weighted = (
            normalized[:, endpoint, features]
            * (rules.weights[selected] / weight_scale)[None, :]
            * rules.reliability[selected][None, :]
        )
        raw[:, endpoint] = np.sum(weighted, axis=1)
        support[:, endpoint] = np.sum(np.maximum(weighted, 0.0), axis=1)
        conflict[:, endpoint] = np.sum(np.maximum(-weighted, 0.0), axis=1)
        matched_counts[:, endpoint] = np.sum(valid, axis=1)
    return EvidenceConsistencyResult(
        scores=_sigmoid(float(scale) * raw).astype(np.float32),
        support_scores=_sigmoid(float(scale) * support).astype(np.float32),
        no_conflict_scores=_sigmoid(-float(scale) * conflict).astype(np.float32),
        matched_counts=matched_counts,
    )
