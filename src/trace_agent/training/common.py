from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from trace_agent.contracts.task import SplitRole
from trace_agent.data.training_dataset import TrainingView


def require_view_role(view: TrainingView, role: SplitRole) -> None:
    if view.role is not role:
        raise ValueError(f"expected a {role.value} view, got {view.role.value}")


def sample_observed_masks(
    availability: np.ndarray,
    *,
    subsets_per_participant: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    available = np.asarray(availability, dtype=bool)
    if available.ndim != 2:
        raise ValueError("availability must be a two-dimensional matrix")
    repeats = int(subsets_per_participant)
    if repeats < 0:
        raise ValueError("subsets_per_participant cannot be negative")
    rng = np.random.default_rng(int(seed))
    source_rows: list[int] = []
    masks: list[np.ndarray] = []
    for row in range(available.shape[0]):
        full = available[row].copy()
        source_rows.append(row)
        masks.append(full)
        candidates = np.flatnonzero(full)
        if candidates.size == 0:
            continue
        for _ in range(repeats):
            count = int(rng.integers(1, candidates.size + 1))
            chosen = rng.choice(candidates, size=count, replace=False)
            mask = np.zeros(available.shape[1], dtype=bool)
            mask[chosen] = True
            source_rows.append(row)
            masks.append(mask)
    return (
        np.asarray(source_rows, dtype=np.int64),
        np.asarray(masks, dtype=bool),
    )


def masked_classification_metrics(
    probability: np.ndarray,
    labels: np.ndarray,
    label_mask: np.ndarray,
) -> dict[str, Any]:
    predictions = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.float64)
    mask = np.asarray(label_mask, dtype=bool)
    if predictions.shape != truth.shape or mask.shape != truth.shape:
        raise ValueError("probability, labels, and label_mask must share one shape")
    valid = mask & np.isfinite(predictions) & np.isfinite(truth)
    clipped = np.clip(predictions, 1e-7, 1.0 - 1e-7)
    loss = -(truth * np.log(clipped) + (1.0 - truth) * np.log(1.0 - clipped))
    endpoint_metrics: list[dict[str, Any]] = []
    aucs: list[float] = []
    average_precisions: list[float] = []
    for endpoint in range(predictions.shape[1]):
        selected = valid[:, endpoint]
        endpoint_truth = truth[selected, endpoint]
        auc: float | None = None
        average_precision: float | None = None
        if selected.any() and np.unique(endpoint_truth).size == 2:
            auc = float(roc_auc_score(endpoint_truth, predictions[selected, endpoint]))
            average_precision = float(
                average_precision_score(
                    endpoint_truth,
                    predictions[selected, endpoint],
                )
            )
            aucs.append(auc)
            average_precisions.append(average_precision)
        endpoint_metrics.append(
            {
                "endpoint_index": endpoint,
                "n": int(np.sum(selected)),
                "positive": int(np.sum(endpoint_truth == 1)),
                "negative": int(np.sum(endpoint_truth == 0)),
                "auroc": auc,
                "average_precision": average_precision,
                "log_loss": (float(np.mean(loss[selected, endpoint])) if selected.any() else None),
            }
        )
    return {
        "participants": int(predictions.shape[0]),
        "n_valid": int(np.sum(valid)),
        "masked_log_loss": float(np.mean(loss[valid])) if valid.any() else None,
        "macro_auroc": float(np.mean(aucs)) if aucs else None,
        "macro_average_precision": (float(np.mean(average_precisions)) if average_precisions else None),
        "endpoints": endpoint_metrics,
    }
