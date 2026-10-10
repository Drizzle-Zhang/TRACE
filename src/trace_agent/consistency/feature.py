from __future__ import annotations

import numpy as np

from trace_agent.contracts.consistency import FeatureConsistencyResult


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def compute_feature_consistency(
    *,
    contributions: np.ndarray,
    medical_weights: np.ndarray,
    match_mask: np.ndarray | None = None,
) -> FeatureConsistencyResult:
    """Compare signed model contributions with signed medical weights."""

    contrib = np.asarray(contributions, dtype=np.float32)
    weights = np.asarray(medical_weights, dtype=np.float32)
    if contrib.ndim != 3:
        raise ValueError("contributions must be participant-by-endpoint-by-feature")
    if weights.shape != contrib.shape[1:]:
        raise ValueError("medical_weights must be endpoint-by-feature")
    matches = weights != 0 if match_mask is None else np.asarray(match_mask, dtype=bool)
    if matches.shape != weights.shape:
        raise ValueError("match_mask must match medical_weights")
    finite_contrib = np.isfinite(contrib)
    valid = matches[None, :, :] & finite_contrib & np.isfinite(weights)[None, :, :]
    # Attribution magnitude includes unmatched features; medical weights are
    # normalized over the contributors actually present for each participant.
    contribution_scale = np.max(np.where(finite_contrib, np.abs(contrib), 0.0), axis=2, keepdims=True)
    weight_scale = np.max(np.where(valid, np.abs(weights)[None, :, :], 0.0), axis=2, keepdims=True)
    normalized_contrib = np.divide(
        contrib,
        contribution_scale,
        out=np.zeros_like(contrib),
        where=finite_contrib & (contribution_scale > 0),
    )
    normalized_weights = np.divide(
        weights[None, :, :],
        weight_scale,
        out=np.zeros_like(contrib),
        where=valid & (weight_scale > 0),
    )
    raw = np.sum(
        np.where(
            valid,
            normalized_contrib * normalized_weights,
            0.0,
        ),
        axis=2,
    )
    matched_counts = np.sum(valid, axis=2).astype(np.int32)
    scores = np.where(matched_counts > 0, _sigmoid(raw), 0.5)
    return FeatureConsistencyResult(
        scores=scores.astype(np.float32),
        raw_scores=raw.astype(np.float32),
        matched_counts=matched_counts,
    )
