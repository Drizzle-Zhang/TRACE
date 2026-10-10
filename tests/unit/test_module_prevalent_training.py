from __future__ import annotations

import numpy as np
import pytest


def test_prevalent_fit_has_five_isotonic_calibrators():
    from trace_agent.training.modules.prevalent import fit_prevalent_module
    from trace_agent.training.modules.settings import ModuleTrainingSettings

    matrix = np.arange(80, dtype=float).reshape(40, 2)
    labels = (np.arange(40) % 2).astype(float)
    settings = ModuleTrainingSettings(n_estimators=3, min_positive=5, min_child_samples=1)
    fit = fit_prevalent_module(matrix, labels, np.arange(40), settings)
    assert len(fit.estimator.calibrated_classifiers_) == 5
    assert fit.estimator.method == "isotonic"
    assert fit.metadata["fit_rows"] == list(range(40))
    probability = fit.estimator.predict_proba(matrix)
    assert probability.shape == (40, 2)
    assert np.isfinite(probability).all()


def test_internal_calibration_base_fits_only_the_provided_rows(monkeypatch):
    from lightgbm import LGBMClassifier

    from trace_agent.training.modules.prevalent import fit_prevalent_module
    from trace_agent.training.modules.settings import ModuleTrainingSettings

    rows = np.arange(100, 140)
    matrix = np.column_stack([rows, np.ones(40)])
    calls = []
    original_fit = LGBMClassifier.fit

    def recording_fit(self, values, labels, *args, **kwargs):
        calls.append(set(np.asarray(values)[:, 0].astype(int).tolist()))
        return original_fit(self, values, labels, *args, **kwargs)

    monkeypatch.setattr(LGBMClassifier, "fit", recording_fit)
    fit_prevalent_module(
        matrix, (rows % 2).astype(float), rows, ModuleTrainingSettings(n_estimators=3, min_positive=5)
    )
    assert len(calls) == 5
    assert all(len(fitted) == 32 and fitted.issubset(rows.tolist()) for fitted in calls)
    assert all(sum(int(row) in fitted for fitted in calls) == 4 for row in rows)


@pytest.mark.parametrize(
    "values",
    [
        {"unknown": True},
        {"seed": -1},
        {"n_estimators": 0},
        {"min_positive": 0},
        {"learning_rate": np.nan},
        {"n_jobs": True},
        {"insufficient_data_policy": "ignore"},
    ],
)
def test_invalid_module_settings_error(values):
    from trace_agent.training.modules.settings import ModuleTrainingSettings

    with pytest.raises(ValueError):
        ModuleTrainingSettings.from_mapping(values, task="prevalent")


def test_cox_default_uses_one_thousand_estimators():
    from trace_agent.training.modules.settings import ModuleTrainingSettings

    assert ModuleTrainingSettings.from_mapping({}, task="longitudinal").n_estimators == 1000


def test_prevalent_inner_fivefold_requires_five_of_each_class():
    from trace_agent.training.modules.prevalent import fit_prevalent_module
    from trace_agent.training.modules.settings import InsufficientModuleData, ModuleTrainingSettings

    with pytest.raises(InsufficientModuleData):
        fit_prevalent_module(
            np.ones((8, 1)),
            np.array([0.0] * 4 + [1.0] * 4),
            np.arange(8),
            ModuleTrainingSettings(min_positive=1),
        )
