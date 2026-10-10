import numpy as np
import pytest

from trace_agent.training.modules.settings import (
    InsufficientModuleData,
    ModuleTrainingSettings,
)


def test_cox_selects_rounds_internally_and_refits_all_rows():
    from trace_agent.training.modules.longitudinal import fit_longitudinal_module

    calls = []

    class SpyEstimator:
        best_iteration = 2

        def __init__(self, **parameters):
            self.parameters = parameters

        def fit(self, matrix, labels, **kwargs):
            calls.append((self.parameters, matrix[:, 0].astype(int).tolist(), labels, kwargs))
            return self

    rows = np.arange(100, 140)
    matrix = np.column_stack([rows, np.ones(40)])
    signed = np.where(np.arange(40) % 2 == 0, 3.0, -12.0)
    settings = ModuleTrainingSettings(n_estimators=10, min_positive=5, early_stopping_rounds=2)
    fit = fit_longitudinal_module(matrix, signed, rows, settings, estimator_factory=SpyEstimator)
    first_parameters, inner_fit, _, first_kwargs = calls[0]
    eval_matrix, eval_labels = first_kwargs["eval_set"][0]
    inner_eval = eval_matrix[:, 0].astype(int).tolist()
    assert set(inner_fit).isdisjoint(inner_eval)
    assert set(inner_fit) | set(inner_eval) == set(rows)
    assert first_parameters["early_stopping_rounds"] == 2
    assert np.array_equal(eval_labels, signed[np.asarray(inner_eval) - 100])
    assert calls[1][0]["n_estimators"] == 3
    assert "early_stopping_rounds" not in calls[1][0]
    assert "eval_set" not in calls[1][3]
    assert calls[1][1] == rows.tolist()
    np.testing.assert_array_equal(calls[1][2], signed)
    assert fit.metadata["selected_rounds"] == 3


def test_real_cox_predicts_finite_log_risk():
    pytest.importorskip("xgboost")
    from xgboost import XGBRegressor

    from trace_agent.training.modules.longitudinal import fit_longitudinal_module

    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(40, 2))
    signed = np.where(np.arange(40) % 2 == 0, 3.0, -12.0)
    fit = fit_longitudinal_module(
        matrix,
        signed,
        np.arange(40),
        ModuleTrainingSettings(
            n_estimators=3,
            min_positive=5,
            early_stopping_rounds=2,
        ),
    )
    assert isinstance(fit.estimator, XGBRegressor)
    assert fit.estimator.get_params()["objective"] == "survival:cox"
    assert np.isfinite(fit.estimator.predict(matrix, output_margin=True)).all()
    assert 1 <= fit.metadata["selected_rounds"] <= 3


@pytest.mark.parametrize("signed", [np.array([0.0, -1.0, 2.0, -2.0]), np.array([np.nan, -1.0, 2.0, -2.0])])
def test_invalid_survival_labels_error(signed):
    from trace_agent.training.modules.longitudinal import survival_inner_rows

    with pytest.raises(ValueError, match="finite and non-zero"):
        survival_inner_rows(signed, ModuleTrainingSettings(min_positive=1))


def test_insufficient_internal_survival_split_errors():
    from trace_agent.training.modules.longitudinal import survival_inner_rows

    with pytest.raises(InsufficientModuleData):
        survival_inner_rows(np.array([1.0, 2.0, -1.0, -2.0]), ModuleTrainingSettings(min_positive=1))


@pytest.mark.parametrize("failure", ["rounds", "fit"])
def test_estimator_failures_are_not_silently_skipped(failure):
    from trace_agent.training.modules.longitudinal import fit_longitudinal_module

    class BrokenEstimator:
        best_iteration = 100

        def __init__(self, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            if failure == "fit":
                raise RuntimeError("estimator failed")
            return self

    signed = np.where(np.arange(40) % 2 == 0, 3.0, -12.0)
    expected = RuntimeError if failure == "fit" else ValueError
    with pytest.raises(expected):
        fit_longitudinal_module(
            np.ones((40, 2)),
            signed,
            np.arange(40),
            ModuleTrainingSettings(n_estimators=3, min_positive=5),
            estimator_factory=BrokenEstimator,
        )


def test_survival_fit_alignment_errors():
    from trace_agent.training.modules.longitudinal import fit_longitudinal_module

    with pytest.raises(ValueError, match="alignment"):
        fit_longitudinal_module(np.ones((5, 2)), np.ones(4), np.arange(4), ModuleTrainingSettings())
