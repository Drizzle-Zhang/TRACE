"""Per-endpoint validation selection and label-free frozen consistency filters.

The probability and strategy tie rules preserve the frozen original algorithms;
see docs/methods.md and docs/provenance/README.md for source identities and golden cases.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from trace_agent.contracts.task import SplitRole, TaskType

STRATEGIES = (
    "kg_mcs_product",
    "medical_knowledge_consistency",
    "kg_mcs_product_x_medical_knowledge",
)


def classification_counts(predicted: np.ndarray, labels: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(labels) >= 0.5
    selected = np.asarray(valid, dtype=bool)
    if predicted.shape != truth.shape or selected.shape != truth.shape:
        raise ValueError("predictions, labels and valid mask must share one shape")
    tp = int(np.sum(predicted & truth & selected))
    fp = int(np.sum(predicted & ~truth & selected))
    fn = int(np.sum(~predicted & truth & selected))
    tn = int(np.sum(~predicted & ~truth & selected))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_positive": tp + fp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def select_probability_cutoff(
    probability: np.ndarray, labels: np.ndarray, observed: np.ndarray
) -> dict[str, Any]:
    """Maximize validation F1, choosing the highest cutoff on an exact tie."""
    scores = np.asarray(probability, dtype=np.float32)
    truth = np.asarray(labels)
    mask = np.asarray(observed, dtype=bool)
    if scores.ndim != 1 or scores.shape != truth.shape or mask.shape != truth.shape:
        raise ValueError("probability cutoff inputs must be aligned one-dimensional arrays")
    valid = mask & np.isfinite(scores) & np.isfinite(truth) & np.isin(truth, (0, 1))
    p = scores[valid].astype(np.float64)
    y = truth[valid] >= 0.5
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probability cutoff inputs must be probabilities in [0, 1]")
    positive = int(y.sum())
    reason = "no_observed_labels" if y.size == 0 else "no_positive_labels" if positive == 0 else ""
    cutoff = None
    if not reason:
        order = np.argsort(-p, kind="mergesort")
        sorted_p, sorted_y = p[order], y[order]
        tp = np.cumsum(sorted_y)
        fp = np.cumsum(~sorted_y)
        fn = positive - tp
        f1 = 2.0 * tp / (2 * tp + fp + fn)
        candidates = np.flatnonzero(np.r_[sorted_p[1:] != sorted_p[:-1], True])
        best = candidates[int(np.argmax(f1[candidates]))]
        cutoff = float(sorted_p[best])
    predicted = np.zeros(y.shape, bool) if cutoff is None else p >= cutoff
    return {
        "cutoff": cutoff,
        "cutoff_valid": cutoff is not None,
        "invalid_reason": reason,
        "obs_count": int(y.size),
        "pos_count": positive,
        "neg_count": int(y.size - positive),
        "metrics": classification_counts(predicted, y, np.ones(y.shape, bool)),
    }


def strategy_score(strategy: str, kg: np.ndarray, feature: np.ndarray, evidence: np.ndarray) -> np.ndarray:
    if strategy == "kg_mcs_product":
        return kg * feature
    if strategy == "medical_knowledge_consistency":
        return evidence
    if strategy == "kg_mcs_product_x_medical_knowledge":
        return kg * feature * evidence
    raise ValueError(f"unknown frozen consistency strategy: {strategy}")


def threshold_grid_values(value: str | Sequence[float] | None) -> np.ndarray:
    if value is None:
        value = "0:1:0.001"
    if isinstance(value, str) and ":" in value:
        start, stop, step = (float(part) for part in value.split(":"))
        if not np.isfinite([start, stop, step]).all() or step <= 0 or stop < start:
            raise ValueError("threshold grid requires finite start <= stop and positive step")
        count = int(np.floor((stop - start) / step + 1e-9)) + 1
        grid = np.asarray([round(start + i * step, 10) for i in range(count)], dtype=np.float64)
    elif isinstance(value, str):
        grid = np.asarray([float(part) for part in value.split(",") if part.strip()])
    else:
        grid = np.asarray(value, dtype=np.float64)
    if grid.ndim != 1 or not grid.size or np.any(~np.isfinite(grid)) or np.any((grid < 0) | (grid > 1)):
        raise ValueError("threshold grid must be non-empty and in [0, 1]")
    return grid


def _score_arrays(
    predictions: np.ndarray, kg: np.ndarray, feature: np.ndarray, evidence: np.ndarray
) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(value, dtype=np.float32) for value in (predictions, kg, feature, evidence))
    if arrays[0].ndim != 2 or any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("probability and consistency scores must share a participant-by-endpoint shape")
    return arrays


def select_frozen_rules(
    *,
    predictions: np.ndarray,
    labels: np.ndarray,
    label_mask: np.ndarray,
    kg_scores: np.ndarray,
    feature_scores: np.ndarray,
    evidence_scores: np.ndarray,
    endpoints: tuple[str, ...],
    split_role: SplitRole | str,
    task: TaskType | str = TaskType.PREVALENT,
    horizon: str | None = None,
    threshold_grid: str | Sequence[float] | None = None,
    min_f1_delta: float = 0.0,
    min_validation_positives: int = 1,
    min_validation_base_positives: int = 1,
) -> dict[str, Any]:
    """Fit endpoint cutoffs and one of the three original strategies on validation."""
    if SplitRole.parse(split_role) is not SplitRole.VALIDATION:
        raise ValueError("consistency rules may only be selected on validation")
    p, kg, feature, evidence = _score_arrays(predictions, kg_scores, feature_scores, evidence_scores)
    truth = np.asarray(labels)
    mask = np.asarray(label_mask, dtype=bool)
    if truth.shape != p.shape or mask.shape != p.shape or len(endpoints) != p.shape[1]:
        raise ValueError("labels, masks and endpoints must align with predictions")
    if min_validation_positives < 1 or min_validation_base_positives < 1 or not np.isfinite(min_f1_delta):
        raise ValueError("selection minima must be positive and min_f1_delta must be finite")
    grid = threshold_grid_values(threshold_grid)
    rows = []
    for index, endpoint in enumerate(endpoints):
        valid = mask[:, index] & np.isfinite(p[:, index]) & np.isfinite(truth[:, index])
        valid &= np.isin(truth[:, index], (0, 1))
        probability = select_probability_cutoff(p[:, index], truth[:, index], valid)
        cutoff = probability["cutoff"]
        baseline = np.zeros(p.shape[0], bool) if cutoff is None else valid & (p[:, index] >= cutoff)
        baseline_metrics = classification_counts(baseline, truth[:, index], valid)
        reason = ""
        if cutoff is None:
            reason = "invalid_probability_cutoff"
        elif probability["pos_count"] < min_validation_positives:
            reason = "too_few_validation_positives"
        elif baseline_metrics["pred_positive"] < min_validation_base_positives:
            reason = "too_few_validation_base_predictions"
        best_key: tuple[float, float, float, float, int] | None = None
        chosen: tuple[str, float, dict[str, Any], float] | None = None
        if not reason:
            for strategy_index, strategy in enumerate(STRATEGIES):
                score = strategy_score(strategy, kg[:, index], feature[:, index], evidence[:, index])
                for threshold in grid:
                    predicted = baseline & np.isfinite(score) & (score >= threshold)
                    metrics = classification_counts(predicted, truth[:, index], valid)
                    delta = metrics["f1"] - baseline_metrics["f1"]
                    fp_reduction = (
                        (baseline_metrics["fp"] - metrics["fp"]) / baseline_metrics["fp"]
                        if baseline_metrics["fp"]
                        else -1e30
                    )
                    key = (
                        delta,
                        metrics["precision"] if metrics["precision"] is not None else -1e30,
                        fp_reduction,
                        -float(threshold),
                        -strategy_index,
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        chosen = (strategy, float(threshold), metrics, delta)
        enabled = chosen is not None and chosen[3] > min_f1_delta
        if not enabled and not reason:
            reason = "no_validation_f1_improvement"
        rows.append(
            {
                "endpoint": endpoint,
                "endpoint_index": index,
                "cutoff": cutoff,
                "cutoff_valid": probability["cutoff_valid"],
                "probability_invalid_reason": probability["invalid_reason"],
                "selected_strategy": chosen[0] if enabled and chosen is not None else None,
                "selected_threshold": chosen[1] if enabled and chosen is not None else None,
                "use_consistency_cutoff": enabled,
                "fallback_reason": reason,
                "validation_baseline": baseline_metrics,
                "validation_filtered": chosen[2] if enabled and chosen is not None else baseline_metrics,
            }
        )
    artifact = {
        "schema_version": 1,
        "task": TaskType.parse(task).value,
        "horizon": horizon,
        "selection_source": "validation_only",
        "endpoints": list(endpoints),
        "settings": {
            "threshold_grid": grid.tolist(),
            "min_f1_delta": float(min_f1_delta),
            "min_validation_positives": int(min_validation_positives),
            "min_validation_base_positives": int(min_validation_base_positives),
        },
        "rules": rows,
    }
    validate_frozen_rules(artifact)
    return artifact


def validate_frozen_rules(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != 1 or artifact.get("selection_source") != "validation_only":
        raise ValueError("frozen rules require schema version 1 and validation-only selection")
    task = TaskType.parse(artifact.get("task", ""))
    horizon = artifact.get("horizon")
    if (task is TaskType.PREVALENT and horizon is not None) or (
        task is TaskType.LONGITUDINAL and not isinstance(horizon, str)
    ):
        raise ValueError("frozen rule task/horizon mismatch")
    endpoints, rows = artifact.get("endpoints"), artifact.get("rules")
    if not isinstance(endpoints, list) or not endpoints or len(set(endpoints)) != len(endpoints):
        raise ValueError("frozen rules require unique endpoint names")
    if not isinstance(rows, list) or len(rows) != len(endpoints):
        raise ValueError("frozen rule endpoint count mismatch")
    for index, (endpoint, row) in enumerate(zip(endpoints, rows)):
        if row.get("endpoint") != endpoint or row.get("endpoint_index") != index:
            raise ValueError("frozen rule endpoint order mismatch")
        cutoff = row.get("cutoff")
        cutoff_valid = row.get("cutoff_valid")
        if cutoff_valid is True:
            if not isinstance(cutoff, (int, float)) or not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
                raise ValueError("frozen probability cutoff must be finite and in [0, 1]")
        elif cutoff_valid is not False or cutoff is not None:
            raise ValueError("invalid frozen probability cutoff flag")
        if row.get("use_consistency_cutoff") is True:
            threshold = row.get("selected_threshold")
            if not cutoff_valid or row.get("selected_strategy") not in STRATEGIES:
                raise ValueError("invalid frozen consistency strategy")
            if (
                not isinstance(threshold, (int, float))
                or not np.isfinite(threshold)
                or not 0 <= threshold <= 1
            ):
                raise ValueError("invalid frozen consistency threshold")
        elif row.get("use_consistency_cutoff") is not False:
            raise ValueError("invalid frozen consistency flag")


def apply_frozen_rules(
    artifact: Mapping[str, Any],
    *,
    predictions: np.ndarray,
    valid_mask: np.ndarray,
    kg_scores: np.ndarray,
    feature_scores: np.ndarray,
    evidence_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return baseline and filtered positives without accessing any labels."""
    validate_frozen_rules(artifact)
    p, kg, feature, evidence = _score_arrays(predictions, kg_scores, feature_scores, evidence_scores)
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != p.shape or p.shape[1] != len(artifact["endpoints"]):
        raise ValueError("frozen rule endpoint count or prediction mask mismatch")
    if np.any(valid & (~np.isfinite(p) | (p < 0) | (p > 1))):
        raise ValueError("valid frozen-filter predictions must be probabilities in [0, 1]")
    baseline = np.zeros(p.shape, bool)
    filtered = np.zeros(p.shape, bool)
    for index, row in enumerate(artifact["rules"]):
        if not row["cutoff_valid"]:
            continue
        baseline[:, index] = valid[:, index] & (p[:, index] >= row["cutoff"])
        filtered[:, index] = baseline[:, index]
        if row["use_consistency_cutoff"]:
            score = strategy_score(
                row["selected_strategy"], kg[:, index], feature[:, index], evidence[:, index]
            )
            filtered[:, index] &= np.isfinite(score) & (score >= row["selected_threshold"])
    return baseline, filtered


def write_frozen_rules(path: str | Path, artifact: Mapping[str, Any]) -> Path:
    validate_frozen_rules(artifact)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(artifact), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return target


def load_frozen_rules(path: str | Path) -> dict[str, Any]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise TypeError("frozen consistency artifact must be an object")
    validate_frozen_rules(artifact)
    return artifact
