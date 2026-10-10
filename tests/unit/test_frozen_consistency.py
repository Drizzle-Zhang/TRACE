"""Golden cases from the frozen original scripts; no original-project imports."""

import json
from pathlib import Path

import numpy as np
import pytest


def _api():
    from trace_agent.consistency import frozen

    return frozen


def test_probability_cutoff_tie_keeps_highest_score_and_tied_groups():
    api = _api()
    result = api.select_probability_cutoff(
        np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 0, 0, 1]), np.ones(4, bool)
    )
    assert result["cutoff"] == pytest.approx(0.9)
    # A tied score cannot be split to manufacture a perfect classification.
    tied = api.select_probability_cutoff(np.array([0.8, 0.8]), np.array([1, 0]), np.ones(2, bool))
    assert tied["metrics"]["tp"] == 1
    assert tied["metrics"]["fp"] == 1
    assert tied["metrics"]["f1"] == pytest.approx(2 / 3)


def test_probability_cutoff_preserves_no_positive_and_no_observed_fallbacks():
    api = _api()
    for mask, reason in [(np.ones(2, bool), "no_positive_labels"), (np.zeros(2, bool), "no_observed_labels")]:
        result = api.select_probability_cutoff(np.array([0.2, 0.8]), np.zeros(2), mask)
        assert result["cutoff"] is None
        assert result["invalid_reason"] == reason


def _select(**overrides):
    api = _api()
    values = {
        "predictions": np.full((4, 2), [0.7, 0.3]),
        "labels": np.array([[1, 1], [1, 0], [0, 0], [0, 0]]),
        "label_mask": np.ones((4, 2), bool),
        "kg_scores": np.array([[1, 1], [1, 1], [0, 1], [0, 1]]),
        "feature_scores": np.ones((4, 2)),
        "evidence_scores": np.array([[1, 1], [1, 1], [0, 1], [0, 1]]),
        "endpoints": ("D1", "D2"),
        "split_role": "validation",
        "threshold_grid": [0.8, 0.2],
        "min_validation_positives": 1,
        "min_validation_base_positives": 1,
    }
    values.update(overrides)
    return api.select_frozen_rules(**values)


def test_per_endpoint_selection_preserves_original_tie_order_and_no_improvement():
    rules = _select()["rules"]
    assert rules[0]["cutoff"] == pytest.approx(0.7)
    assert rules[0]["selected_strategy"] == "kg_mcs_product"
    assert rules[0]["selected_threshold"] == 0.2
    assert rules[0]["use_consistency_cutoff"] is True
    assert rules[0]["validation_filtered"]["f1"] == 1.0
    assert rules[1]["cutoff"] == pytest.approx(0.3)
    assert rules[1]["use_consistency_cutoff"] is False
    assert rules[1]["fallback_reason"] == "no_validation_f1_improvement"


def test_minimum_eligibility_and_strict_f1_delta_preserve_baseline():
    rules = _select(min_validation_positives=3)["rules"]
    assert all(row["fallback_reason"] == "too_few_validation_positives" for row in rules)
    assert not _select(min_f1_delta=1.0)["rules"][0]["use_consistency_cutoff"]
    assert _select(min_validation_base_positives=5)["rules"][0]["fallback_reason"] == (
        "too_few_validation_base_predictions"
    )


def test_fitting_rejects_test_selection_and_frozen_filter_honors_invalid_predictions():
    api = _api()
    with pytest.raises(ValueError, match="validation"):
        _select(split_role="test")
    artifact = _select()
    baseline, filtered = api.apply_frozen_rules(
        artifact,
        predictions=np.full((2, 2), 0.8),
        valid_mask=np.array([[True, False], [True, True]]),
        kg_scores=np.array([[0, 1], [1, 1]]),
        feature_scores=np.ones((2, 2)),
        evidence_scores=np.ones((2, 2)),
    )
    np.testing.assert_array_equal(baseline, [[True, False], [True, True]])
    np.testing.assert_array_equal(filtered, [[False, False], [True, True]])


def test_original_three_strategy_formulas_and_artifact_roundtrip(tmp_path):
    api = _api()
    expected = {
        "kg_mcs_product": 0.2,
        "medical_knowledge_consistency": 0.8,
        "kg_mcs_product_x_medical_knowledge": 0.16,
    }
    for strategy, value in expected.items():
        assert api.strategy_score(
            strategy, np.array([0.4]), np.array([0.5]), np.array([0.8])
        ) == pytest.approx(value)
    path = tmp_path / "rules.json"
    api.write_frozen_rules(path, _select())
    assert api.load_frozen_rules(path)["rules"] == _select()["rules"]
    payload = json.loads(path.read_text())
    payload["endpoints"] = ["D2", "D1"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="endpoint"):
        api.load_frozen_rules(path)


def test_original_reference_golden_cases_without_original_project():
    reference = json.loads(
        (Path(__file__).parents[1] / "fixtures/frozen_consistency_reference.json").read_text()
    )
    assert reference["source_hashes"]["scripts/build_aligned_probability_cutoffs.py"] == (
        "3ee317dde6ce67d7e76ef23cf1e9c00dd7bea82f3c9b5747a0c850790ac9bfee"
    )
    for case in reference["cases"]:
        result = _api().select_frozen_rules(
            predictions=np.asarray(case["probability"]),
            labels=np.asarray(case["labels"]),
            label_mask=np.asarray(case["label_mask"]),
            kg_scores=np.asarray(case["kg"]),
            feature_scores=np.asarray(case["feature"]),
            evidence_scores=np.asarray(case["evidence"]),
            endpoints=tuple(case["endpoints"]),
            split_role="validation",
            threshold_grid=case["threshold_grid"],
            min_validation_positives=1,
            min_validation_base_positives=1,
        )
        for expected, actual in zip(case["expected"], result["rules"]):
            assert {name: actual[name] for name in expected} == expected
