"""Calibrated LightGBM module predictions and crossfit reference cases."""

import json
from pathlib import Path

import numpy as np

from trace_agent.data.module_outcomes import encode_outcomes
from trace_agent.risk_skills.module_model import ModuleModel
from trace_agent.training.modules.crossfit import crossfit_pair
from trace_agent.training.modules.prevalent import fit_prevalent_module
from trace_agent.training.modules.settings import ModuleTrainingSettings

REFERENCE_SHA256 = "62fac5a04483c04f107040ebaeae84622021d9119c2780a3cd51c4938196daf2"
FEATURES = ("c", "a", "b")


def synthetic_original_inputs():
    """A deterministic cohort, declared column order, partial NaNs and future events."""
    rows = np.arange(235)
    a = np.sin(rows * 0.31)
    b = np.cos(rows * 0.17)
    c = ((rows * 17) % 31) / 31
    labels = (0.7 * a - 0.4 * b + 0.8 * c + 0.4 * np.sin(rows * 0.77) > 0.3).astype(float)
    matrix = np.column_stack([c, a, b])
    matrix[::17, 1] = np.nan
    matrix[[7, 137]] = np.nan
    target = labels.copy()
    years = np.where(labels == 1, -2.0, 10.0)
    target[[31, 59]] = 1.0
    years[[31, 59]] = 3.0
    return rows, matrix, target, years


def _reference():
    path = Path(__file__).parents[1] / "fixtures/prevalent_module_reference.json"
    values = json.loads(path.read_text())
    assert values["source_sha256"] == REFERENCE_SHA256
    return values


def test_final_calibrated_model_matches_reference_predictions():
    reference = _reference()
    rows, matrix, target, years = synthetic_original_inputs()
    targets = encode_outcomes(target, years, task="prevalent")
    eligible = targets.label_mask[:210, 0] & np.isfinite(matrix[:210]).any(axis=1)
    # Source disease index 1: final seed = 123 + 1 * 3.
    fitted = fit_prevalent_module(
        matrix[:210][eligible],
        targets.labels[:210, 0][eligible],
        rows[:210][eligible],
        ModuleTrainingSettings(seed=126),
    )
    model = ModuleModel("prevalent", "M1", "D1", FEATURES, fitted.estimator)
    np.testing.assert_allclose(
        model.predict(matrix[210:222]), reference["test_probability"], rtol=0, atol=2e-7
    )
    np.testing.assert_allclose(
        model.predict(matrix[222:]), reference["train_step2_probability"], rtol=0, atol=2e-7
    )
    assert len(fitted.estimator.calibrated_classifiers_) == 5
    assert fitted.metadata["fit_rows"] == reference["eligible_train_rows"]
    for calibrated in fitted.estimator.calibrated_classifiers_:
        parameters = calibrated.estimator.get_params()
        assert parameters["n_estimators"] == 100
        assert parameters["learning_rate"] == 0.1
        assert parameters["num_leaves"] == 31
        assert parameters["min_child_samples"] == 20
        assert parameters["metric"] == "auc"
        assert "early_stopping_rounds" not in parameters


def test_outer_oof_matches_reference_with_same_external_folds_and_seed():
    reference = _reference()
    rows, matrix, target, years = synthetic_original_inputs()
    targets = encode_outcomes(target[:210], years[:210], task="prevalent")
    # Source disease index 1: OOF seed = 123 + 1 * 4.
    result = crossfit_pair(
        task="prevalent",
        module="M1",
        endpoint="D1",
        feature_names=FEATURES,
        matrix=matrix[:210],
        labels=targets.labels[:, 0],
        label_mask=targets.label_mask[:, 0],
        row_indices=rows[:210],
        folds=rows[:210] % 5 + 1,
        settings=ModuleTrainingSettings(seed=127),
    )
    expected_rows = np.asarray(reference["eligible_train_rows"], dtype=int)
    np.testing.assert_allclose(
        result.oof_scores[expected_rows], reference["oof_probability"], rtol=0, atol=2e-7
    )
    # Preserve unavailable rows and score future cases without using their labels.
    assert not result.oof_mask[[7, 137]].any()
    assert np.isnan(result.oof_scores[[7, 137]]).all()
    assert result.oof_mask[[31, 59]].all()
    for record in result.metadata["folds"]:
        assert set(record["fit_rows"]).isdisjoint(record["held_rows"])
        assert not {31, 59}.intersection(record["fit_rows"])


def test_finite_prevalent_labels_match_reference_time_rules():
    target = np.array([0, 1, 1, 0, 1, 0], dtype=float)
    years = np.array([5, -2, 3, -1, -0.01, 12], dtype=float)
    # Exact source expression for finite, nonzero event times.
    diagnosis = np.where((target == 1) & (years < 0), 1, np.where((target == 1) & (years > 0), 2, 0))
    targets = encode_outcomes(target, years, task="prevalent")
    np.testing.assert_array_equal(targets.label_mask[:, 0], np.isin(diagnosis, [0, 1]))
    np.testing.assert_array_equal(targets.labels[targets.label_mask], diagnosis[np.isin(diagnosis, [0, 1])])
