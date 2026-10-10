from __future__ import annotations

import numpy as np


def build_horizon_labels(
    *,
    event: np.ndarray,
    time: np.ndarray,
    label_mask: np.ndarray,
    horizons: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    event_array = np.asarray(event, dtype=bool).reshape(-1)
    time_array = np.asarray(time, dtype=np.float64).reshape(-1)
    mask = np.asarray(label_mask, dtype=bool).reshape(-1)
    horizon_array = np.asarray(horizons, dtype=np.float64).reshape(-1)
    if event_array.shape != time_array.shape or mask.shape != time_array.shape:
        raise ValueError("event, time and label_mask must share one shape")
    if np.any(~np.isfinite(horizon_array)) or np.any(horizon_array <= 0):
        raise ValueError("horizons must be finite and positive")

    labels = event_array[:, None] & (time_array[:, None] <= horizon_array[None, :])
    # A participant censored exactly at h is observed event-free through h.
    known = labels | (time_array[:, None] >= horizon_array[None, :])
    known &= mask[:, None] & np.isfinite(time_array)[:, None]
    return labels.astype(np.float32), known


def _softplus(values: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, np.asarray(values, dtype=np.float64))


def multi_horizon_bce_gain(
    *,
    current_logits: np.ndarray,
    updated_logits: np.ndarray,
    labels: np.ndarray,
    known_mask: np.ndarray,
    horizon_weights: np.ndarray,
) -> float:
    current = np.asarray(current_logits, dtype=np.float64)
    updated = np.asarray(updated_logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.float64)
    known = np.asarray(known_mask, dtype=bool)
    weights = np.asarray(horizon_weights, dtype=np.float64).reshape(-1)
    if current.shape != updated.shape or current.shape != truth.shape or known.shape != truth.shape:
        raise ValueError("logits, labels and known_mask must share one shape")
    if current.ndim < 2 or current.shape[-1] != weights.size:
        raise ValueError("horizon_weights must align with the final logits dimension")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("horizon_weights must be finite, non-negative and non-zero")
    valid = known & np.isfinite(current) & np.isfinite(updated) & np.isfinite(truth)
    effective = np.where(valid, np.broadcast_to(weights, truth.shape), 0.0)
    denominator = float(np.sum(effective))
    if denominator <= 0:
        return 0.0
    current_loss = _softplus(current[valid]) - truth[valid] * current[valid]
    updated_loss = _softplus(updated[valid]) - truth[valid] * updated[valid]
    return float(np.sum((current_loss - updated_loss) * effective[valid]) / denominator)
