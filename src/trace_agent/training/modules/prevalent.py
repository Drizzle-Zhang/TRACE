"""LightGBM diagnosis models calibrated within the provided training rows."""

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

from trace_agent.training.modules.settings import (
    ModuleFit,
    ModuleTrainingSettings,
    check_binary_training,
)


def fit_prevalent_module(
    matrix: np.ndarray,
    labels: np.ndarray,
    row_indices: np.ndarray,
    settings: ModuleTrainingSettings,
) -> ModuleFit:
    check_binary_training(labels, settings, min_class=5)
    if matrix.shape[0] != labels.size or row_indices.size != labels.size:
        raise ValueError("prevalent fit row alignment mismatch")
    base = LGBMClassifier(
        objective="binary",
        metric="auc",
        verbosity=-1,
        random_state=settings.seed,
        n_estimators=settings.n_estimators,
        learning_rate=settings.learning_rate,
        num_leaves=settings.num_leaves,
        min_child_samples=settings.min_child_samples,
        n_jobs=settings.n_jobs,
    )
    estimator = CalibratedClassifierCV(estimator=base, method="isotonic", cv=5, ensemble=True, n_jobs=1)
    estimator.fit(matrix, labels)
    return ModuleFit(
        estimator,
        {
            "fit_rows": row_indices.tolist(),
            "calibration_folds": 5,
            "validation_rows_used": 0,
            "test_rows_used": 0,
        },
    )
