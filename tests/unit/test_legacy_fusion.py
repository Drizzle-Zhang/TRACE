"""Synthetic contracts for portable model-compatible fusion predictors."""

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression


def api():
    from trace_agent.integration import legacy_fusion

    return legacy_fusion


class CapturingClassifier:
    def __init__(self, margin=0.0):
        self.margin = margin
        self.values = None

    def predict(self, values, *, raw_score):
        assert raw_score
        self.values = np.asarray(values).copy()
        return np.nan_to_num(values, nan=0.0).sum(axis=1) + self.margin


class CapturingCox:
    def __init__(self, margin):
        self.margin = margin
        self.values = None

    def predict(self, values, *, output_margin):
        assert output_margin
        self.values = np.asarray(values).copy()
        return np.full(values.shape[0], self.margin, dtype=np.float32)


def cpu_expf(values):
    from xgboost.core import _LIB

    exponential = _LIB.expf
    exponential.argtypes = [ctypes.c_float]
    exponential.restype = ctypes.c_float
    values = np.asarray(values, dtype=np.float32)
    return np.asarray([exponential(float(v)) for v in values.flat], dtype=np.float32).reshape(values.shape)


def test_prevalent_selects_saved_indices_and_hides_unused_or_missing_features():
    first, second = CapturingClassifier(0.25), CapturingClassifier(0.5)
    fusion = api().LegacyPrevalentFusion(
        modules=("A", "B", "C"),
        diseases=("D1", "D2", "D3"),
        models=(first, second, None),
        model_feature_indices=(np.asarray([2, 0]), np.asarray([1]), None),
        model_input_space="prob",
    )
    scores = np.asarray([[0.2, 0.7, 0.9], [0.3, 0.8, 0.4], [0.6, 0.4, 0.5]], dtype=np.float32)
    prediction_mask = np.ones_like(scores, dtype=bool)
    prediction_mask[0, 0] = False
    logits = fusion.fuse_logits(scores, np.asarray([True, False, True]), prediction_mask)
    np.testing.assert_array_equal(first.values, np.asarray([[0.6, np.nan]], dtype=np.float32))
    np.testing.assert_array_equal(second.values, np.asarray([[np.nan]], dtype=np.float32))
    np.testing.assert_allclose(logits, np.asarray([0.85, 0.5, 0.0], dtype=np.float32), rtol=1e-6)
    assert fusion.M == 3 and fusion.D == 3


def test_prevalent_clips_and_transforms_all_features_in_float32():
    estimator = CapturingClassifier()
    fusion = api().LegacyPrevalentFusion(("A", "B", "C"), ("D",), (estimator,))
    scores = np.asarray([[0.0], [1.0], [0.25]], dtype=np.float32)
    fusion.fuse_logits(scores, np.ones(3))
    clipped = np.clip(scores[:, 0], 1e-6, 1.0 - 1e-6)
    expected = np.log(clipped / (1.0 - clipped))[None, :]
    np.testing.assert_array_equal(estimator.values, expected)
    assert estimator.values.dtype == np.float32


def test_prevalent_applies_calibration_before_clipped_logits_and_probabilities():
    calibrator = IsotonicRegression(out_of_bounds="clip").fit([0.0, 0.5, 1.0], [0.0, 0.2, 1.0])
    fusion = api().LegacyPrevalentFusion(
        modules=("A",),
        diseases=("LOW", "HIGH", "MIDDLE", "UNCALIBRATED"),
        models=tuple(CapturingClassifier(v) for v in [-100.0, 100.0, 0.0, 0.0]),
        calibrators={0: calibrator, 1: calibrator, 2: calibrator},
    )
    scores = np.full((1, 4), np.nan, dtype=np.float32)
    probability = np.asarray([1e-6, 1.0 - 1e-6, 0.2, 0.5], dtype=np.float32)
    expected_logits = np.log(probability / (1.0 - probability))
    with np.errstate(over="ignore"):
        np.testing.assert_array_equal(fusion.fuse_logits(scores, np.ones(1)), expected_logits)
        np.testing.assert_array_equal(fusion.fuse(scores, np.ones(1)), 1.0 / (1.0 + np.exp(-expected_logits)))


def test_prevalent_empty_calibration_preserves_the_calibrated_wrapper_roundtrip():
    fusion = api().LegacyPrevalentFusion(("A",), ("D",), (CapturingClassifier(100.0),))
    scores = np.asarray([[np.nan]], dtype=np.float32)
    probability = np.asarray([1.0 - 1e-6], dtype=np.float32)
    np.testing.assert_array_equal(
        fusion.fuse_logits(scores, np.ones(1)), np.log(probability / (1.0 - probability))
    )


def test_cox_uses_selected_hr_features_and_leaves_unfitted_endpoints_nan():
    estimator = CapturingCox(np.log(np.float32(7.0)))
    fusion = api().LegacyCoxFusion(
        modules=("A", "B", "C"),
        diseases=("D1", "D2"),
        models=(estimator, None),
        active_module_indices=(np.asarray([2, 0]), np.asarray([])),
        risk_loc=np.asarray([1.0, 0.0]),
        risk_scale=np.asarray([2.0, 1.0]),
    )
    scores = np.log(np.arange(1, 13, dtype=np.float32).reshape(2, 3, 2))
    mask = np.ones_like(scores, dtype=bool)
    mask[1, 0, 0] = False
    observed = np.asarray([[True, True, False], [True, True, True]])
    result = fusion.predict(scores, pred_mask_nmd=mask, used_mask_nm=observed)
    hr = cpu_expf(scores)
    expected_features = np.asarray([[np.nan, hr[0, 0, 0]], [hr[1, 2, 0], np.nan]], dtype=np.float32)
    np.testing.assert_array_equal(estimator.values, expected_features)
    np.testing.assert_array_equal(result[:, 0], np.full(2, estimator.margin, dtype=np.float32))
    assert np.isnan(result[:, 1]).all()
    np.testing.assert_array_equal(
        fusion.predict(scores, pred_mask_nmd=mask, used_mask_nm=observed, standardized=True),
        (cpu_expf(result) - np.asarray([1.0, 0.0], dtype=np.float32))
        / np.asarray([2.0, 1.0], dtype=np.float32),
    )


def test_cox_hr_rounding_matches_the_xgboost_cpu_symbol():
    values = np.linspace(-8.0, 8.0, 301, dtype=np.float32).reshape(7, 43)
    np.testing.assert_array_equal(api().cox_margin_to_legacy_hr(values), cpu_expf(values))


@pytest.mark.parametrize("batch", [False, True])
def test_cox_standardization_preserves_saved_scale_fallback(batch):
    fusion = api().LegacyCoxFusion(
        modules=("A",),
        diseases=("D1", "D2", "D3", "D4"),
        models=(None,) * 4,
        active_module_indices=(np.asarray([]),) * 4,
        risk_loc=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        risk_scale=np.asarray([0.0, np.nan, -1.0, 2.0], dtype=np.float32),
    )
    margins = np.asarray([0.0, 0.5, 1.0, np.nan], dtype=np.float32)
    if batch:
        margins = margins[None, :]
    expected = (cpu_expf(margins) - fusion.risk_loc) / np.asarray([1.0, 1.0, -1.0, 2.0], dtype=np.float32)
    np.testing.assert_array_equal(fusion.standardize(margins), expected)


def test_cox_rejects_finite_margins_that_overflow_saved_hr_features():
    fusion = api().LegacyCoxFusion(("A",), ("D",), (None,), (np.asarray([]),), [0.0], [1.0])
    with pytest.raises(FloatingPointError, match="overflow"):
        fusion.predict(np.asarray([[[1000.0]]], dtype=np.float32))


def test_pickles_reload_in_an_isolated_process(tmp_path):
    from xgboost import XGBRegressor

    values = np.arange(16, dtype=np.float32).reshape(8, 2) / 10.0
    classifier = LGBMClassifier(n_estimators=2, num_leaves=3, min_child_samples=1, n_jobs=1, verbosity=-1)
    classifier.fit(values, np.asarray([0, 0, 0, 0, 1, 1, 1, 1]))
    cox = XGBRegressor(objective="survival:cox", n_estimators=2, max_depth=1, n_jobs=1)
    cox.fit(values, np.asarray([1, -2, 3, -4, 5, -6, 7, -8], dtype=np.float32))
    calibrator = IsotonicRegression(out_of_bounds="clip").fit([0.0, 1.0], [0.1, 0.9])
    prevalence = api().LegacyPrevalentFusion(
        ("A", "B"), ("D",), (classifier,), model_input_space="prob", calibrators={0: calibrator}
    )
    survival = api().LegacyCoxFusion(("A", "B"), ("D",), (cox,), (np.asarray([0, 1]),), [0.0], [1.0])
    path = tmp_path / "portable.joblib"
    joblib.dump((prevalence, survival), path)
    payload = path.read_bytes()
    for forbidden in (b"ukbpp", b"code_root", b"legacy_model_support"):
        assert forbidden not in payload
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    script = """
import sys
sys.path.insert(0, sys.argv[2])
import joblib
import numpy as np
prevalence, survival = joblib.load(sys.argv[1])
np.testing.assert_array_equal(prevalence.fuse(np.full((2, 1), .4, np.float32), np.ones(2)), [float(sys.argv[3])])
np.testing.assert_array_equal(survival.predict(np.zeros((1, 2, 1), np.float32)), [[float(sys.argv[4])]])
assert not any(name == 'ukbpp' or name.startswith('ukbpp.') for name in sys.modules)
assert 'legacy_model_support' not in sys.modules
assert set(vars(prevalence)) == {'modules', 'diseases', 'models', 'model_feature_indices', 'model_input_space', 'calibrators'}
assert set(vars(survival)) == {'modules', 'diseases', 'models', 'active_module_indices', 'risk_loc', 'risk_scale'}
print('portable reload passed')
"""
    expected_p = prevalence.fuse(np.full((2, 1), 0.4, np.float32), np.ones(2))[0]
    expected_r = survival.predict(np.zeros((1, 2, 1), np.float32))[0, 0]
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(path),
            str(Path(__file__).resolve().parents[2] / "src"),
            str(float(expected_p)),
            str(float(expected_r)),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == "portable reload passed"
