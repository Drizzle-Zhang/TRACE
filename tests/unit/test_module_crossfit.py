import numpy as np
import pytest

from trace_agent.training.modules.settings import (
    InsufficientModuleData,
    ModuleFit,
    ModuleTrainingSettings,
)


class ConstantEstimator:
    classes_ = np.array([0, 1])

    def predict_proba(self, matrix):
        return np.tile([0.6, 0.4], (matrix.shape[0], 1))


def pair_arguments():
    rows = np.arange(100)
    return {
        "task": "prevalent",
        "module": "M1",
        "endpoint": "D1",
        "feature_names": ("row",),
        "matrix": rows[:, None].astype(float),
        "labels": (rows % 2).astype(float),
        "label_mask": np.ones(100, dtype=bool),
        "row_indices": rows,
        "folds": np.tile(np.arange(1, 6), 20),
        "settings": ModuleTrainingSettings(min_positive=5),
    }


def test_outer_fold_never_enters_fit_and_ineligible_rows_still_get_scores():
    from trace_agent.training.modules.crossfit import crossfit_pair

    calls = []

    def fit(matrix, labels, row_indices, settings):
        calls.append(set(row_indices.tolist()))
        return ModuleFit(ConstantEstimator(), {"fit_rows": row_indices.tolist()})

    args = pair_arguments()
    args["label_mask"][0] = False
    result = crossfit_pair(**args, fit_function=fit)
    assert result.status == "trained"
    assert result.oof_mask.all()
    for fold, fitted in enumerate(calls[:5], 1):
        assert fitted.isdisjoint(args["row_indices"][args["folds"] == fold].tolist())
        assert 0 not in fitted
    assert len(calls) == 6
    assert calls[-1] == set(range(1, 100))


@pytest.mark.parametrize("policy", ["skip", "error"])
def test_preflight_insufficiency_performs_zero_fits(policy):
    from trace_agent.training.modules.crossfit import crossfit_pair

    def forbidden_fit(*args):
        pytest.fail("preflight must complete before fitting")

    args = pair_arguments()
    args["labels"][:] = 0
    args["settings"] = ModuleTrainingSettings(min_positive=5, insufficient_data_policy=policy)
    if policy == "error":
        with pytest.raises(InsufficientModuleData):
            crossfit_pair(**args, fit_function=forbidden_fit)
    else:
        result = crossfit_pair(**args, fit_function=forbidden_fit)
        assert result.status == "skipped"
        assert result.model is None
        assert not result.oof_mask.any()
        assert np.isnan(result.oof_scores).all()


def test_all_missing_feature_rows_keep_nan_and_false():
    from trace_agent.training.modules.crossfit import crossfit_pair

    args = pair_arguments()
    args["matrix"][0] = np.nan
    result = crossfit_pair(**args, fit_function=lambda *a: ModuleFit(ConstantEstimator(), {}))
    assert np.isnan(result.oof_scores[0])
    assert not result.oof_mask[0]
    assert result.oof_mask[1:].all()


def test_unexpected_fit_error_is_not_skipped():
    from trace_agent.training.modules.crossfit import crossfit_pair

    def broken(*args):
        raise RuntimeError("broken estimator")

    with pytest.raises(RuntimeError, match="broken estimator"):
        crossfit_pair(**pair_arguments(), fit_function=broken)


def test_cox_inner_splits_stay_inside_outer_complement():
    from trace_agent.training.modules.crossfit import crossfit_pair
    from trace_agent.training.modules.longitudinal import fit_longitudinal_module

    class SpyCox:
        best_iteration = 0

        def __init__(self, **parameters):
            pass

        def fit(self, *args, **kwargs):
            return self

        def predict(self, matrix, *, output_margin):
            assert output_margin is True
            return np.ones(matrix.shape[0]) * -2

    args = pair_arguments()
    args.update(task="longitudinal", signed_time=np.where(args["labels"] == 1, 3.0, -12.0))
    result = crossfit_pair(
        **args, fit_function=lambda *a: fit_longitudinal_module(*a, estimator_factory=SpyCox)
    )
    for record in result.metadata["folds"]:
        held = set(record["held_rows"])
        inner_fit, inner_val = set(record["inner_fit_rows"]), set(record["inner_validation_rows"])
        assert inner_fit.isdisjoint(inner_val)
        assert held.isdisjoint(inner_fit | inner_val)
        assert inner_fit | inner_val == set(record["fit_rows"])
    assert result.metadata["final_fit"]["fit_rows"] == list(range(100))
    assert np.all(result.oof_scores == -2)


@pytest.mark.parametrize("bad_input", ["shape", "folds", "infinity", "signed", "event"])
def test_invalid_crossfit_inputs_error(bad_input):
    from trace_agent.training.modules.crossfit import crossfit_pair

    args = pair_arguments()
    if bad_input == "shape":
        args["label_mask"] = np.ones(99, dtype=bool)
    elif bad_input == "folds":
        args["folds"][:] = 1
    elif bad_input == "infinity":
        args["matrix"][0] = np.inf
    else:
        args["task"] = "longitudinal"
        args["signed_time"] = np.ones(99) if bad_input == "signed" else np.ones(100)
    with pytest.raises(ValueError):
        crossfit_pair(**args)
