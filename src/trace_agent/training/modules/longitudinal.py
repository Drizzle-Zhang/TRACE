"""Cox module fitting with train-only round selection and full refitting."""

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from trace_agent.training.modules.settings import (
    InsufficientModuleData,
    ModuleFit,
    ModuleTrainingSettings,
    check_binary_training,
)


def survival_inner_rows(
    signed_time: np.ndarray, settings: ModuleTrainingSettings
) -> tuple[np.ndarray, np.ndarray]:
    signed = np.asarray(signed_time, dtype=float).reshape(-1)
    if not np.isfinite(signed).all() or np.any(signed == 0):
        raise ValueError("Cox signed_time must be finite and non-zero")
    event = (signed > 0).astype(float)
    check_binary_training(event, settings, min_class=2)
    try:
        fit, validation = train_test_split(
            np.arange(signed.size),
            test_size=settings.inner_validation_fraction,
            random_state=settings.seed,
            stratify=event,
        )
    except ValueError as exc:
        raise InsufficientModuleData("internal survival split is not feasible") from exc
    if np.unique(event[fit]).size != 2 or np.unique(event[validation]).size != 2:
        raise InsufficientModuleData("internal survival split needs event and censor in both parts")
    return np.asarray(fit), np.asarray(validation)


def fit_longitudinal_module(
    matrix: np.ndarray,
    signed_time: np.ndarray,
    row_indices: np.ndarray,
    settings: ModuleTrainingSettings,
    *,
    estimator_factory: Callable[..., Any] | None = None,
) -> ModuleFit:
    if matrix.shape[0] != signed_time.size or row_indices.size != signed_time.size:
        raise ValueError("longitudinal fit row alignment mismatch")
    inner_fit, inner_validation = survival_inner_rows(signed_time, settings)
    if estimator_factory is None:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("Cox module training requires trace-agent[longitudinal]") from exc
        estimator_factory = XGBRegressor
    parameters = {
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "random_state": settings.seed,
        "n_jobs": settings.n_jobs,
        "learning_rate": settings.learning_rate,
        "max_depth": settings.max_depth,
    }
    selector = estimator_factory(
        **parameters,
        n_estimators=settings.n_estimators,
        early_stopping_rounds=settings.early_stopping_rounds,
    )
    selector.fit(
        matrix[inner_fit],
        signed_time[inner_fit],
        eval_set=[(matrix[inner_validation], signed_time[inner_validation])],
        verbose=False,
    )
    rounds = int(selector.best_iteration) + 1
    if not 1 <= rounds <= settings.n_estimators:
        raise ValueError("Cox selected rounds outside configured range")
    estimator = estimator_factory(**parameters, n_estimators=rounds)
    estimator.fit(matrix, signed_time, verbose=False)
    return ModuleFit(
        estimator,
        {
            "fit_rows": row_indices.tolist(),
            "inner_fit_rows": row_indices[inner_fit].tolist(),
            "inner_validation_rows": row_indices[inner_validation].tolist(),
            "selected_rounds": rounds,
            "validation_rows_used": 0,
            "test_rows_used": 0,
        },
    )
