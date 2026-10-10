from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from trace_agent.contracts.consistency import (
    ConsistencyResult,
    ConsistencyThresholds,
)
from trace_agent.contracts.task import SplitRole


def _aligned(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    converted = tuple(np.asarray(array) for array in arrays)
    shapes = {array.shape for array in converted}
    if len(shapes) != 1:
        raise ValueError("all consistency arrays must share one shape")
    return converted


def _combined(kg: np.ndarray, feature: np.ndarray, evidence: np.ndarray) -> np.ndarray:
    stack = np.stack([kg, feature, evidence], axis=0).astype(np.float32)
    finite = np.isfinite(stack)
    return np.divide(
        np.sum(np.where(finite, stack, 0.0), axis=0),
        np.sum(finite, axis=0),
        out=np.full(stack.shape[1:], np.nan, dtype=np.float32),
        where=np.sum(finite, axis=0) > 0,
    )


def _f1(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> tuple[float, int]:
    predicted = prediction & valid
    positive = truth & valid
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    denominator = 2 * tp + fp + fn
    return (
        (2.0 * tp / denominator) if denominator else 0.0,
        fp,
    )


def _best_threshold(
    score: np.ndarray,
    *,
    baseline_positive: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    grid: np.ndarray,
) -> float:
    best_key: tuple[float, int, float] | None = None
    best_value = float(grid[0])
    for threshold in grid:
        kept = baseline_positive & np.isfinite(score) & (score >= float(threshold))
        f1, false_positives = _f1(kept, truth, valid)
        key = (-f1, false_positives, float(threshold))
        if best_key is None or key < best_key:
            best_key = key
            best_value = float(threshold)
    return best_value


def learn_consistency_thresholds(
    *,
    predictions: np.ndarray,
    labels: np.ndarray,
    label_mask: np.ndarray,
    kg_scores: np.ndarray,
    feature_scores: np.ndarray,
    evidence_scores: np.ndarray,
    split_role: SplitRole | str,
    prediction_threshold: float = 0.5,
    grid: Iterable[float] | np.ndarray | None = None,
) -> ConsistencyThresholds:
    """Select thresholds once on validation labels."""

    role = SplitRole.parse(split_role)
    if not role.allows_threshold_selection:
        raise ValueError("consistency thresholds may only be learned on validation data")
    pred, truth, mask, kg, feature, evidence = _aligned(
        predictions,
        labels,
        label_mask,
        kg_scores,
        feature_scores,
        evidence_scores,
    )
    pred = pred.astype(np.float32)
    truth_bool = truth.astype(np.float32) >= 0.5
    valid = mask.astype(bool) & np.isfinite(truth) & np.isfinite(pred)
    grid_values = np.asarray(
        list(grid) if grid is not None else np.linspace(0.0, 1.0, 101),
        dtype=np.float64,
    ).reshape(-1)
    if (
        grid_values.size == 0
        or np.any(~np.isfinite(grid_values))
        or np.any((grid_values < 0) | (grid_values > 1))
    ):
        raise ValueError("threshold grid must contain values in [0, 1]")
    baseline = pred >= float(prediction_threshold)
    combined = _combined(kg, feature, evidence)
    return ConsistencyThresholds(
        prediction=float(prediction_threshold),
        kg=_best_threshold(
            kg,
            baseline_positive=baseline,
            truth=truth_bool,
            valid=valid,
            grid=grid_values,
        ),
        feature=_best_threshold(
            feature,
            baseline_positive=baseline,
            truth=truth_bool,
            valid=valid,
            grid=grid_values,
        ),
        evidence=_best_threshold(
            evidence,
            baseline_positive=baseline,
            truth=truth_bool,
            valid=valid,
            grid=grid_values,
        ),
        combined=_best_threshold(
            combined,
            baseline_positive=baseline,
            truth=truth_bool,
            valid=valid,
            grid=grid_values,
        ),
        source_split=role,
    )


def apply_consistency_filter(
    *,
    predictions: np.ndarray,
    kg_scores: np.ndarray,
    feature_scores: np.ndarray,
    evidence_scores: np.ndarray,
    thresholds: ConsistencyThresholds,
) -> ConsistencyResult:
    """Apply frozen thresholds without consulting inference/test labels."""

    pred, kg, feature, evidence = _aligned(predictions, kg_scores, feature_scores, evidence_scores)
    pred = pred.astype(np.float32)
    kg = kg.astype(np.float32)
    feature = feature.astype(np.float32)
    evidence = evidence.astype(np.float32)
    baseline = np.isfinite(pred) & (pred >= thresholds.prediction)
    combined = _combined(kg, feature, evidence)
    masks = {
        "baseline": baseline,
        "kg": baseline & np.isfinite(kg) & (kg >= thresholds.kg),
        "feature": baseline & np.isfinite(feature) & (feature >= thresholds.feature),
        "evidence": baseline & np.isfinite(evidence) & (evidence >= thresholds.evidence),
        "joint": baseline
        & np.isfinite(kg)
        & np.isfinite(feature)
        & np.isfinite(evidence)
        & (kg >= thresholds.kg)
        & (feature >= thresholds.feature)
        & (evidence >= thresholds.evidence),
        "combined": baseline & np.isfinite(combined) & (combined >= thresholds.combined),
    }
    return ConsistencyResult(
        kg_scores=kg,
        feature_scores=feature,
        evidence_scores=evidence,
        combined_scores=combined,
        strategy_masks=masks,
        thresholds=thresholds,
    )
