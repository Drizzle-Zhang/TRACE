"""Track original rows through real LightGBM predictions into real isotonic fits."""

import json
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from lightgbm import LGBMClassifier
from sklearn import __version__ as sklearn_version
from sklearn import calibration
from sklearn.isotonic import IsotonicRegression

from tests.module_fixtures import module_config, write_module_inputs
from trace_agent.training.modules import crossfit
from trace_agent.training.modules.service import prepare_module_data, train_modules


@pytest.fixture
def cohort(tmp_path):
    write_module_inputs(tmp_path)
    config = module_config(tmp_path, task="prevalent")
    catalog_path = tmp_path / "catalog.yaml"
    catalog = yaml.safe_load(catalog_path.read_text())
    catalog["modules"] = catalog["modules"][:1]
    catalog["pairs"] = catalog["pairs"][:1]
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False))

    frame = pd.read_csv(tmp_path / "features.csv")
    frame["original_row"] = np.arange(len(frame))
    splits = pd.read_csv(tmp_path / "splits.csv")
    # Noncontiguous train rows expose confusion between local and cohort indices.
    permutation = np.random.default_rng(17).permutation(len(frame))
    roles = np.empty(len(frame), dtype=object)
    roles[permutation[:120]] = "train"
    roles[permutation[120:150]] = "validation"
    roles[permutation[150:]] = "test"
    splits["split"] = roles
    splits.to_csv(tmp_path / "splits.csv", index=False)
    train_rows = np.flatnonzero(roles == "train")
    folds = np.arange(train_rows.size) % 5 + 1
    fold_path = tmp_path / "folds.csv"
    pd.DataFrame({"row_index": train_rows, "eid": frame.eid.iloc[train_rows], "fold": folds}).to_csv(
        fold_path, index=False
    )
    config["training"]["modules"]["fold_table"] = {"path": str(fold_path)}

    unavailable, unknown, future = train_rows[:3]
    names = ["original_row", "b", "a"]
    frame.loc[unavailable, names] = np.nan
    frame.to_csv(tmp_path / "features.csv", index=False)
    pd.DataFrame({"feature_code": names}).to_csv(tmp_path / "M1_D1.csv", index=False)
    outcomes = pd.read_csv(tmp_path / "outcome.csv")
    outcomes.loc[unknown, "target_y"] = np.nan
    outcomes.loc[future, ["target_y", "BL2Target_yrs"]] = [1.0, 3.0]
    outcomes.to_csv(tmp_path / "outcome.csv", index=False)
    prepare_module_data(config)
    labels = np.load(tmp_path / "prepared/cohort_y.npy")[:, 0]
    eligible = set(train_rows.tolist()) - {int(unavailable), int(unknown), int(future)}
    return {
        "config": config,
        "matrix": frame[names].to_numpy(),
        "labels": labels,
        "train_rows": train_rows,
        "folds": folds,
        "eligible": eligible,
        "roles": roles,
        "unavailable": int(unavailable),
        "unlabelled": {int(unknown), int(future)},
    }


def trace_actual_fits(monkeypatch, cohort, *, inject_row=None):
    """Spies delegate all numerical work; row identity follows array memory, not scores."""
    groups = []
    active_group = None
    active_calibrator = None
    original_module_fit = crossfit.fit_prevalent_module
    original_base_fit = LGBMClassifier.fit
    original_predict = LGBMClassifier.predict_proba
    original_calibrator = calibration._fit_calibrator
    original_isotonic_fit = IsotonicRegression.fit

    @wraps(original_module_fit)
    def module_fit(matrix, labels, row_indices, settings):
        nonlocal active_group
        group = {"provided_rows": row_indices.tolist(), "pairs": {}}
        groups.append(group)
        active_group = group
        try:
            return original_module_fit(matrix, labels, row_indices, settings)
        finally:
            active_group = None

    @wraps(original_base_fit)
    def base_fit(self, matrix, labels, *args, **kwargs):
        assert active_group is not None
        rows = np.asarray(matrix)[:, 0].astype(int)
        np.testing.assert_array_equal(labels, cohort["labels"][rows])
        active_group["pairs"][id(self)] = {
            "fit_rows": rows.tolist(),
            "eval_rows": [
                np.asarray(values)[:, 0].astype(int).tolist() for values, _ in (kwargs.get("eval_set") or [])
            ],
            "predictions": [],
            "isotonic_rows": [],
        }
        return original_base_fit(self, matrix, labels, *args, **kwargs)

    @wraps(original_predict)
    def predict(self, matrix, *args, **kwargs):
        probabilities = original_predict(self, matrix, *args, **kwargs)
        if active_group is not None:
            rows = np.asarray(matrix)[:, 0].astype(int)
            # Keep the actual returned array alive: equal/repeated scores cannot identify rows.
            active_group["pairs"][id(self)]["predictions"].append((rows, probabilities))
        return probabilities

    @wraps(original_calibrator)
    def calibrator(clf, predictions, labels, *args, **kwargs):
        nonlocal active_calibrator
        assert active_group is not None
        record = active_group["pairs"][id(clf)]
        if inject_row is not None and len(groups) == 1 and len(active_group["pairs"]) == 1:
            # Calibration-only leakage: leave the real base fit untouched, add one forbidden row.
            rows = np.append(record["predictions"][-1][0], inject_row)
            predictions = clf.predict_proba(cohort["matrix"][rows])[:, 1:]
            labels = cohort["labels"][rows]
        active_calibrator = record
        try:
            return original_calibrator(clf, predictions, labels, *args, **kwargs)
        finally:
            active_calibrator = None

    @wraps(original_isotonic_fit)
    def isotonic_fit(self, scores, labels, *args, **kwargs):
        assert active_calibrator is not None
        matching = [
            (rows, probabilities)
            for rows, probabilities in active_calibrator["predictions"]
            if np.shares_memory(np.asarray(scores), probabilities)
        ]
        assert len(matching) == 1, "actual isotonic scores must originate from one recorded prediction"
        rows, probabilities = matching[0]
        np.testing.assert_array_equal(scores, probabilities[:, 1])
        np.testing.assert_array_equal(labels, cohort["labels"][rows])
        active_calibrator["isotonic_rows"].append(rows.tolist())
        return original_isotonic_fit(self, scores, labels, *args, **kwargs)

    monkeypatch.setattr(crossfit, "fit_prevalent_module", module_fit)
    monkeypatch.setattr(LGBMClassifier, "fit", base_fit)
    monkeypatch.setattr(LGBMClassifier, "predict_proba", predict)
    monkeypatch.setattr(calibration, "_fit_calibrator", calibrator)
    monkeypatch.setattr(IsotonicRegression, "fit", isotonic_fit)
    return groups


def assert_isolation(groups, cohort):
    assert len(groups) == 6  # Five outer models, then the distinct final deployment model.
    for index, group in enumerate(groups):
        held = set(cohort["train_rows"][cohort["folds"] == index + 1].tolist()) if index < 5 else set()
        allowed = cohort["eligible"] - held
        assert set(group["provided_rows"]) == allowed
        pairs = list(group["pairs"].values())
        assert len(pairs) == 5
        calibrated_once = []
        for pair in pairs:
            fitted = set(pair["fit_rows"])
            assert fitted <= allowed, "base fit contains a forbidden row"
            assert pair["eval_rows"] == []  # This LightGBM path does not use early stopping.
            assert len(pair["isotonic_rows"]) == 1
            calibrated = set(pair["isotonic_rows"][0])
            assert calibrated.isdisjoint(held), "isotonic calibration contains an outer held row"
            assert calibrated <= allowed, "isotonic calibration contains a nontraining/ineligible row"
            assert fitted.isdisjoint(calibrated), "inner classifier fit and calibration overlap"
            assert fitted | calibrated == allowed
            calibrated_once.extend(pair["isotonic_rows"][0])
        # Every eligible row calibrates exactly one of the five inner ensemble members.
        assert sorted(calibrated_once) == sorted(allowed)


def test_real_nested_isotonic_calibration_rows_are_isolated(monkeypatch, cohort):
    groups = trace_actual_fits(monkeypatch, cohort)
    manifest = json.loads(train_modules(cohort["config"]).read_text())
    assert_isolation(groups, cohort)
    metadata = manifest["pairs"][0]["metadata"]
    for index, record in enumerate(metadata["folds"]):
        assert record["fit_rows"] == groups[index]["provided_rows"]
        assert set(record["held_rows"]) == set(cohort["train_rows"][cohort["folds"] == index + 1].tolist())
    assert set(metadata["final_fit"]["fit_rows"]) == cohort["eligible"]
    assert sum(len(group["pairs"]) for group in groups) == 30
    cache_path = Path(cohort["config"]["artifacts"]["module_model_dir"]) / manifest["oof_cache"]["file"]
    with np.load(cache_path) as cache:
        np.testing.assert_array_equal(cache["train_rows"], cohort["train_rows"])
        mask = cache["pred_mask"][:, 0, 0]
        rows = cohort["train_rows"]
        assert not mask[rows == cohort["unavailable"]].any()
        assert mask[np.isin(rows, list(cohort["unlabelled"]))].all()
    # Save the synthetic row trace in pytest's output directory for diagnostics.
    trace = {
        "sklearn_version": sklearn_version,
        "train_rows": cohort["train_rows"].tolist(),
        "validation_rows": np.flatnonzero(cohort["roles"] == "validation").tolist(),
        "test_rows": np.flatnonzero(cohort["roles"] == "test").tolist(),
        "eligible_train_rows": sorted(cohort["eligible"]),
        "groups": [
            {
                "stage": "outer_oof" if index < 5 else "final_deployment",
                "fold": index + 1 if index < 5 else None,
                "provided_rows": group["provided_rows"],
                "held_rows": metadata["folds"][index]["held_rows"] if index < 5 else [],
                "inner_pairs": [
                    {name: pair[name] for name in ("fit_rows", "eval_rows", "isotonic_rows")}
                    for pair in group["pairs"].values()
                ],
            }
            for index, group in enumerate(groups)
        ],
    }
    (cache_path.parent / "isotonic_row_trace.json").write_text(json.dumps(trace) + "\n")


@pytest.mark.parametrize("leak_source", ["outer_held", "validation", "test"])
def test_row_trace_detects_calibration_only_leakage(monkeypatch, cohort, leak_source):
    if leak_source == "outer_held":
        candidates = set(cohort["train_rows"][cohort["folds"] == 1]) & cohort["eligible"]
        injected = min(candidates)
        message = "isotonic calibration contains an outer held row"
    else:
        injected = int(np.flatnonzero(cohort["roles"] == leak_source)[0])
        message = "isotonic calibration contains a nontraining/ineligible row"
    groups = trace_actual_fits(monkeypatch, cohort, inject_row=injected)
    train_modules(cohort["config"])
    first_pair = next(iter(groups[0]["pairs"].values()))
    assert injected not in first_pair["fit_rows"]
    assert injected in first_pair["isotonic_rows"][0]
    with pytest.raises(AssertionError, match=message):
        assert_isolation(groups, cohort)
