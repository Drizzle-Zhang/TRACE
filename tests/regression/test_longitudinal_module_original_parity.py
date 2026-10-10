"""Cox source-kernel evidence, including explicitly approved native changes."""

import json
from pathlib import Path

import numpy as np
import pytest

from trace_agent.data.module_outcomes import encode_outcomes
from trace_agent.risk_skills.module_model import ModuleModel
from trace_agent.training.modules.crossfit import crossfit_pair
from trace_agent.training.modules.longitudinal import fit_longitudinal_module
from trace_agent.training.modules.settings import ModuleTrainingSettings

REFERENCE_SHA256 = "85c887b818b68a8107725ac215af1d0b7a822aa93b2dde221a21e6cd518664a8"
FEATURES = ("c", "a", "b")


def synthetic_original_inputs():
    """Selected feature order, partial NaNs, all-NaN rows and baseline cases."""
    rows = np.arange(215)
    a = np.sin(rows * 0.31)
    b = np.cos(rows * 0.17)
    c = ((rows * 17) % 31) / 31
    labels = (0.7 * a - 0.4 * b + 0.8 * c + 0.4 * np.sin(rows * 0.77) > 0.3).astype(float)
    matrix = np.column_stack([c, a, b])
    matrix[::17, 1] = np.nan
    matrix[[7, 137]] = np.nan
    years = 1.0 + ((rows * 11) % 71) / 7
    labels[[31, 59]] = 1.0
    years[[31, 59]] = -2.0
    return rows, matrix, labels, years


def reference():
    path = Path(__file__).parents[1] / "fixtures/longitudinal_module_original_reference.json"
    values = json.loads(path.read_text())
    assert values["source_sha256"] == REFERENCE_SHA256
    return values


def pair_arguments():
    rows, matrix, target, years = synthetic_original_inputs()
    targets = encode_outcomes(target[:190], years[:190], task="longitudinal")
    return {
        "task": "longitudinal",
        "module": "M1",
        "endpoint": "D1",
        "feature_names": FEATURES,
        "matrix": matrix[:190],
        "labels": targets.labels[:, 0],
        "label_mask": targets.label_mask[:, 0],
        "signed_time": targets.signed_time[:, 0],
        "row_indices": rows[:190],
        "folds": rows[:190] % 5 + 1,
        "settings": ModuleTrainingSettings.from_mapping({"seed": 127}, task="longitudinal"),
    }


def test_final_cox_matches_original_kernel_after_approved_selection_and_refit():
    pytest.importorskip("xgboost")
    expected = reference()
    rows, matrix, target, years = synthetic_original_inputs()
    targets = encode_outcomes(target[:190], years[:190], task="longitudinal")
    eligible = targets.label_mask[:, 0] & np.isfinite(matrix[:190]).any(axis=1)
    fitted = fit_longitudinal_module(
        matrix[:190][eligible],
        targets.signed_time[:, 0][eligible],
        rows[:190][eligible],
        ModuleTrainingSettings.from_mapping({"seed": 126}, task="longitudinal"),
    )
    assert fitted.metadata["fit_rows"] == expected["eligible_train_rows"]
    assert fitted.metadata["selected_rounds"] == expected["approved_final_kernel"]["selected_rounds"]
    model = ModuleModel("longitudinal", "M1", "D1", FEATURES, fitted.estimator)
    margins = model.predict(matrix[190:])
    np.testing.assert_allclose(margins, expected["approved_final_kernel"]["log_risk"], rtol=0, atol=2e-7)
    np.testing.assert_allclose(np.exp(margins), expected["approved_final_kernel"]["hr"], rtol=2e-7)
    assert model.score_scale == "log_risk"
    assert np.any(margins < 0)
    assert np.any(margins > 0)
    assert fitted.estimator.get_params()["eval_metric"] == "cox-nloglik"
    assert fitted.metadata["validation_rows_used"] == fitted.metadata["test_rows_used"] == 0


def test_oof_cox_matches_original_kernel_with_train_only_round_selection():
    pytest.importorskip("xgboost")
    expected = reference()["approved_oof_kernel"]
    result = crossfit_pair(**pair_arguments())
    np.testing.assert_allclose(result.oof_scores[expected["rows"]], expected["log_risk"], rtol=0, atol=2e-7)
    assert [item["selected_rounds"] for item in result.metadata["folds"]] == expected["rounds"]
    assert not result.oof_mask[[7, 137]].any()
    assert np.isnan(result.oof_scores[[7, 137]]).all()
    # Baseline cases are ineligible for incident fitting but remain scoreable.
    assert result.oof_mask[[31, 59]].all()
    for fold in result.metadata["folds"]:
        held = set(fold["held_rows"])
        assert held.isdisjoint(fold["fit_rows"])
        assert held.isdisjoint(fold["inner_validation_rows"])
        assert not {31, 59}.intersection(fold["fit_rows"])


def test_held_out_event_times_do_not_change_their_own_oof_predictions():
    pytest.importorskip("xgboost")
    arguments = pair_arguments()
    original = crossfit_pair(**arguments)
    held = arguments["folds"] == 1
    arguments["signed_time"] = arguments["signed_time"].copy()
    held_times = arguments["signed_time"][held]
    arguments["signed_time"][held] = np.sign(held_times) * (20 - np.abs(held_times))
    changed = crossfit_pair(**arguments)
    np.testing.assert_array_equal(original.oof_scores[held], changed.oof_scores[held])
    assert original.metadata["folds"][0] == changed.metadata["folds"][0]


def test_finite_incident_labels_and_signed_times_match_original_rules():
    target = np.array([0, 1, 1, 0, 1, 0], dtype=float)
    years = np.array([5, -2, 3, 0.1, -0.01, 12], dtype=float)
    diagnosis = np.where((target == 1) & (years < 0), 1, np.where((target == 1) & (years > 0), 2, 0))
    eligible = np.isin(diagnosis, [0, 2])
    old_event = np.where(diagnosis[eligible] == 2, 1, 0)
    old_signed = np.where(old_event == 1, years[eligible], -years[eligible])
    targets = encode_outcomes(target, years, task="incident")
    np.testing.assert_array_equal(targets.label_mask[:, 0], eligible)
    np.testing.assert_array_equal(targets.labels[eligible, 0], old_event)
    np.testing.assert_allclose(targets.signed_time[eligible, 0], old_signed, rtol=1e-7)
    np.testing.assert_allclose(targets.event_time[eligible, 0], np.abs(old_signed), rtol=1e-7)
