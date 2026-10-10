"""Train-only outer cross-fitting for one declared module/endpoint pair."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from trace_agent.contracts.task import TaskType
from trace_agent.risk_skills.module_model import ModuleModel
from trace_agent.training.modules.longitudinal import (
    fit_longitudinal_module,
    survival_inner_rows,
)
from trace_agent.training.modules.prevalent import fit_prevalent_module
from trace_agent.training.modules.settings import (
    InsufficientModuleData,
    ModuleFit,
    ModuleTrainingSettings,
    check_binary_training,
)


@dataclass(frozen=True)
class PairTrainingResult:
    status: str
    model: ModuleModel | None
    oof_scores: np.ndarray
    oof_mask: np.ndarray
    metadata: Mapping[str, Any]


def crossfit_pair(
    *,
    task: TaskType | str,
    module: str,
    endpoint: str,
    feature_names: tuple[str, ...],
    matrix: np.ndarray,
    labels: np.ndarray,
    label_mask: np.ndarray,
    row_indices: np.ndarray,
    folds: np.ndarray,
    settings: ModuleTrainingSettings,
    signed_time: np.ndarray | None = None,
    fit_function: Callable[..., ModuleFit] | None = None,
) -> PairTrainingResult:
    resolved = TaskType.parse(task)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
        raise ValueError("crossfit feature shape mismatch")
    n = matrix.shape[0]
    if any(np.asarray(value).shape != (n,) for value in (labels, label_mask, row_indices, folds)):
        raise ValueError("crossfit train row alignment mismatch")
    if set(np.asarray(folds).tolist()) != {1, 2, 3, 4, 5}:
        raise ValueError("crossfit requires five non-empty folds")
    if np.isinf(matrix).any():
        raise ValueError("crossfit features contain infinity")
    feature_available = np.isfinite(matrix).any(axis=1)
    eligible = feature_available & np.asarray(label_mask, dtype=bool)
    targets = np.asarray(labels, dtype=float)
    if resolved is TaskType.LONGITUDINAL:
        if signed_time is None or np.asarray(signed_time).shape != (n,):
            raise ValueError("crossfit requires aligned signed_time")
        targets = np.asarray(signed_time, dtype=float)
        if np.any(eligible & (~np.isfinite(targets) | (targets == 0))):
            raise ValueError("eligible signed_time must be finite and non-zero")
        if not np.array_equal((targets[eligible] > 0).astype(float), labels[eligible]):
            raise ValueError("event and signed_time disagree")
    subsets = [eligible & (folds != fold) for fold in range(1, 6)] + [eligible]
    try:
        for selected in subsets:
            if resolved is TaskType.PREVALENT:
                check_binary_training(labels[selected], settings, min_class=5)
            else:
                survival_inner_rows(targets[selected], settings)
    except InsufficientModuleData as exc:
        if settings.insufficient_data_policy == "error":
            raise
        return PairTrainingResult(
            "skipped",
            None,
            np.full(n, np.nan, dtype=np.float32),
            np.zeros(n, dtype=bool),
            {"reason": str(exc), "eligible_train_count": int(eligible.sum())},
        )
    fit = fit_function or (
        fit_prevalent_module if resolved is TaskType.PREVALENT else fit_longitudinal_module
    )
    scores = np.full(n, np.nan, dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    fold_metadata = []
    for fold, selected in enumerate(subsets[:5], 1):
        fitted = fit(matrix[selected], targets[selected], row_indices[selected], settings)
        wrapper = ModuleModel(resolved, module, endpoint, feature_names, fitted.estimator)
        held = feature_available & (folds == fold)
        if held.any():
            predicted = wrapper.predict(matrix[held])
            positions = np.flatnonzero(held)
            valid = np.isfinite(predicted)
            scores[positions[valid]] = predicted[valid]
            mask[positions[valid]] = True
        fold_metadata.append(
            {
                "fold": fold,
                **dict(fitted.metadata),
                "held_rows": row_indices[folds == fold].tolist(),
            }
        )
    final = fit(matrix[eligible], targets[eligible], row_indices[eligible], settings)
    model = ModuleModel(resolved, module, endpoint, feature_names, final.estimator)
    return PairTrainingResult(
        "trained",
        model,
        scores,
        mask,
        {
            "folds": fold_metadata,
            "final_fit": dict(final.metadata),
            "eligible_train_count": int(eligible.sum()),
            "oof_valid_count": int(mask.sum()),
        },
    )
