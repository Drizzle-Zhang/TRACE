from __future__ import annotations

import numpy as np


def compute_kg_consistency(
    module_scores: np.ndarray,
    pred_mask: np.ndarray,
    kg_values: np.ndarray,
) -> np.ndarray:
    """Compute a module-mask-aware knowledge consistency score."""

    scores = np.asarray(module_scores, dtype=np.float32)
    mask = np.asarray(pred_mask, dtype=bool)
    kg = np.asarray(kg_values, dtype=np.float32)
    if scores.ndim != 3:
        raise ValueError("module_scores must have shape participant-by-module-by-endpoint")
    if mask.shape != scores.shape:
        raise ValueError("pred_mask must match module_scores")
    if kg.shape != scores.shape[1:]:
        raise ValueError("kg_values must have shape module-by-endpoint")
    if np.any(mask & ~np.isfinite(scores)):
        raise ValueError("valid module scores must be finite")
    weights = kg[None, :, :] * mask.astype(np.float32)
    numerator = np.sum(np.where(mask, scores, 0.0) * weights, axis=1)
    denominator = np.sum(weights, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float32),
        where=denominator > 0,
    ).astype(np.float32, copy=False)
