"""Actual old CLI references, with an explicit approved missing-input correction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trace_agent.consistency.survival_policies import (
    apply_survival_policies,
    evaluate_survival_policies,
    load_survival_policies,
    select_survival_policies,
    write_survival_policies,
)

REFERENCE = json.loads(
    (Path(__file__).parents[1] / "fixtures/survival_consistency_original_reference.json").read_text()
)
NAN_CORRECTION_REFERENCE = json.loads(
    (Path(__file__).parents[1] / "fixtures/survival_consistency_approved_nan_reference.json").read_text()
)


def _expected_case(case: dict) -> dict:
    # The original fixture is immutable. Only cases containing missing KG/med
    # inputs use the independently generated old-CLI endpoint correction.
    missing = any(
        not np.isfinite(np.asarray(case[split][field], dtype=float)).all()
        for split in ("validation", "test")
        for field in ("kg", "med")
    )
    if not missing:
        return case
    return next(value for value in NAN_CORRECTION_REFERENCE["cases"] if value["name"] == case["name"])


def test_nan_correction_reference_preserves_inputs_and_original_finite_results():
    corrected = {case["name"]: case for case in NAN_CORRECTION_REFERENCE["cases"]}
    assert corrected.keys() == {case["name"] for case in REFERENCE["cases"]}
    assert NAN_CORRECTION_REFERENCE["source"]["approved_nan_correction"]
    changed = []
    for original in REFERENCE["cases"]:
        updated = corrected[original["name"]]
        for key in ("name", "endpoints", "settings", "validation", "test"):
            assert updated[key] == original[key]
        if _expected_case(original) is original:
            assert updated == original
        elif updated != original:
            changed.append(original["name"])
    assert set(changed) == {"random_ties_0", "random_ties_1"}


def _select(case: dict, **kwargs: object) -> dict:
    arrays = {key: np.asarray(value) for key, value in case["validation"].items()}
    return select_survival_policies(
        predictions=arrays["risk"],
        labels=arrays["y"],
        label_mask=arrays["known"],
        kg_scores=arrays["kg"],
        evidence_scores=arrays["med"],
        endpoints=tuple(case["endpoints"]),
        split_role="validation",
        horizon="10y",
        **case["settings"],
        **kwargs,
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_default_survival_candidates_finalists_and_selector_match_source_reference(
    case: dict, tmp_path: Path
):
    artifact = _select(case)
    expected_case = _expected_case(case)
    assert artifact["all_strategy_specs"] == expected_case["all_strategy_specs"]
    assert artifact["finalists"] == expected_case["finalists"]
    assert artifact["selector_candidates"] == expected_case["selector_candidates"]
    assert [row["selected_strategy"] for row in artifact["rules"]] == expected_case["selected_strategies"]
    assert artifact["kg_q99"] == expected_case["kg_q99"]
    path = write_survival_policies(tmp_path / "policies.json", artifact)
    assert load_survival_policies(path) == artifact
    arrays = {key: np.asarray(value) for key, value in case["test"].items()}
    baseline, selected, strategies = apply_survival_policies(
        artifact,
        predictions=arrays["risk"],
        valid_mask=arrays["known"],
        kg_scores=arrays["kg"],
        evidence_scores=arrays["med"],
    )
    np.testing.assert_array_equal(selected, expected_case["expected_selected_mask"])
    for name, expected in expected_case["expected_strategy_masks"].items():
        np.testing.assert_array_equal(strategies[name], expected)
    assert not np.any(baseline & ~arrays["known"].astype(bool))


def test_survival_kg_only_strategy_restores_the_lost_true_positive():
    case = REFERENCE["cases"][0]
    artifact = _select(case)
    arrays = {key: np.asarray(value) for key, value in case["test"].items()}
    _, selected, _ = apply_survival_policies(
        artifact,
        predictions=arrays["risk"],
        valid_mask=arrays["known"],
        kg_scores=arrays["kg"],
        evidence_scores=arrays["med"],
    )
    np.testing.assert_array_equal(selected[:, 0], [True, False])


def test_survival_policy_selection_rejects_test_role():
    case = REFERENCE["cases"][0]
    arrays = {key: np.asarray(value) for key, value in case["validation"].items()}
    with pytest.raises(ValueError, match="validation"):
        select_survival_policies(
            predictions=arrays["risk"],
            labels=arrays["y"],
            label_mask=arrays["known"],
            kg_scores=arrays["kg"],
            evidence_scores=arrays["med"],
            endpoints=("D0",),
            split_role="test",
            horizon="10y",
        )


def test_survival_application_can_compute_only_selected_policies():
    case = REFERENCE["cases"][1]
    artifact = _select(case)
    arrays = {key: np.asarray(value) for key, value in case["test"].items()}
    _, selected, strategies = apply_survival_policies(
        artifact,
        predictions=arrays["risk"],
        valid_mask=arrays["known"],
        kg_scores=arrays["kg"],
        evidence_scores=arrays["med"],
        include_strategies=False,
    )
    assert set(strategies) == {"baseline_risk_only"} | {row["selected_strategy"] for row in artifact["rules"]}
    np.testing.assert_array_equal(selected, _expected_case(case)["expected_selected_mask"])


def test_survival_finalist_evaluation_accumulates_identical_counts_by_chunk():
    case = REFERENCE["cases"][1]
    artifact = _select(case)
    arrays = {key: np.asarray(value) for key, value in case["test"].items()}
    kwargs = {
        "predictions": arrays["risk"],
        "labels": arrays["y"],
        "label_mask": arrays["known"],
        "kg_scores": arrays["kg"],
        "evidence_scores": arrays["med"],
    }
    tiny = evaluate_survival_policies(artifact, **kwargs, chunk_size=1)
    larger = evaluate_survival_policies(artifact, **kwargs, chunk_size=8)
    assert tiny == larger
    assert len(tiny["summary"]) == len(artifact["finalists"])


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_survival_finalist_and_selector_reports_match_source_reference(case: dict):
    artifact = _select(case)
    expected_case = _expected_case(case)
    arrays = {key: np.asarray(value) for key, value in case["test"].items()}
    report = evaluate_survival_policies(
        artifact,
        predictions=arrays["risk"],
        labels=arrays["y"],
        label_mask=arrays["known"],
        kg_scores=arrays["kg"],
        evidence_scores=arrays["med"],
        chunk_size=3,
    )
    for name in ("summary", "per_disease", "selector_summary", "selector_per_disease"):
        expected = (
            expected_case["expected_test_" + name]
            if name in {"summary", "per_disease"}
            else expected_case["expected_" + name]
        )
        observed_rows = report[name] if isinstance(report[name], list) else [report[name]]
        expected_rows = expected if isinstance(expected, list) else [expected]
        assert len(observed_rows) == len(expected_rows)
        for actual, original in zip(observed_rows, expected_rows):
            # CSV unions optional policy columns across the whole table and
            # represents both empty text and absent values as blank cells.
            assert actual.keys() <= original.keys()
            for field, value in original.items():
                observed = actual.get(field)
                if isinstance(value, float):
                    assert observed == pytest.approx(value)
                elif value is None:
                    assert observed in (None, "")
                else:
                    assert observed == value
