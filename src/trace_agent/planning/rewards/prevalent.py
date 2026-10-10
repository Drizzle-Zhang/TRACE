from __future__ import annotations

import numpy as np


def _binary_cross_entropy(probability: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    labels = np.asarray(labels, dtype=np.float64)
    return -(labels * np.log(probability) + (1.0 - labels) * np.log(1.0 - probability))


def prevalent_bce_gain(
    *,
    current_probability: np.ndarray,
    updated_probability: np.ndarray,
    labels: np.ndarray,
    label_mask: np.ndarray,
) -> float:
    current = np.asarray(current_probability, dtype=np.float64)
    updated = np.asarray(updated_probability, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.float64)
    mask = np.asarray(label_mask, dtype=bool)
    if current.shape != updated.shape or current.shape != truth.shape or mask.shape != truth.shape:
        raise ValueError("probabilities, labels and label_mask must share one shape")
    valid = mask & np.isfinite(current) & np.isfinite(updated) & np.isfinite(truth)
    if not np.any(valid):
        return 0.0
    return float(
        np.mean(_binary_cross_entropy(current[valid], truth[valid]))
        - np.mean(_binary_cross_entropy(updated[valid], truth[valid]))
    )
