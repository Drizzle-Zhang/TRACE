from __future__ import annotations

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


def _prevalent_summary(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    probabilities = np.clip(_sigmoid(logits), 1e-6, 1.0 - 1e-6)
    entropy = -probabilities * np.log(probabilities) - (1.0 - probabilities) * np.log(1.0 - probabilities)
    p_sorted = np.sort(probabilities)[::-1]
    z_sorted = np.sort(logits)[::-1]
    p_max = float(p_sorted[0]) if p_sorted.size else 0.0
    p_second = float(p_sorted[1]) if p_sorted.size > 1 else 0.0
    z_max = float(z_sorted[0]) if z_sorted.size else 0.0
    z_second = float(z_sorted[1]) if z_sorted.size > 1 else 0.0
    return np.asarray(
        [
            float(np.mean(entropy)) if entropy.size else 0.0,
            p_max,
            p_second,
            p_max - p_second if p_sorted.size > 1 else 0.0,
            float(np.mean(p_sorted[: min(3, p_sorted.size)])) if p_sorted.size else 0.0,
            float(np.mean(logits)) if logits.size else 0.0,
            float(np.std(logits)) if logits.size else 0.0,
            z_max,
            z_second,
            z_max - z_second if z_sorted.size > 1 else 0.0,
        ],
        dtype=np.float32,
    )


def _longitudinal_summary(risk: np.ndarray) -> np.ndarray:
    finite = np.asarray(risk, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.zeros(8, dtype=np.float32)
    values = np.sort(finite)[::-1]
    top = np.zeros(3, dtype=np.float32)
    top[: min(3, values.size)] = values[: min(3, values.size)]
    maximum = float(values[0])
    second = float(values[1]) if values.size > 1 else maximum
    return np.concatenate(
        [
            top,
            np.asarray(
                [
                    float(np.mean(values)),
                    float(np.std(values)),
                    maximum,
                    maximum - second if values.size > 1 else 0.0,
                    maximum - float(values[-1]),
                ],
                dtype=np.float32,
            ),
        ]
    )


def build_state_features(
    profile: RiskProfile,
    *,
    used_modules: np.ndarray,
    accumulated_burden: float,
) -> np.ndarray:
    """Build a Q-state vector compatible with saved planner artifacts."""

    if profile.n_participants != 1:
        raise ValueError("planner state can only be built for one participant")
    used = np.asarray(used_modules, dtype=np.float32).reshape(-1)
    values = np.asarray(profile.raw_scores[0], dtype=np.float32).copy()
    values[~np.asarray(profile.valid_mask[0], dtype=bool)] = 0.0
    if profile.task is TaskType.PREVALENT:
        logits = profile.metadata.get("logits")
        if logits is not None:
            values = np.asarray(logits, dtype=np.float32).reshape(profile.raw_scores.shape)[0].copy()
            values[~np.asarray(profile.valid_mask[0], dtype=bool)] = 0.0
        summary = _prevalent_summary(values)
    else:
        standardized = profile.metadata.get("standardized_scores")
        if standardized is not None:
            values = np.asarray(standardized, dtype=np.float32).reshape(profile.raw_scores.shape)[0].copy()
            values[~np.asarray(profile.valid_mask[0], dtype=bool)] = 0.0
        summary = _longitudinal_summary(values)
    return np.concatenate(
        [
            values,
            summary,
            used,
            np.asarray(
                [float(np.sum(used > 0.5)), float(accumulated_burden)],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32, copy=False)
