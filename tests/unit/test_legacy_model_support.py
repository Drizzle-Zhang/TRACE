"""The audit bridge must preserve HR inputs and learned missing-model semantics."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def support():
    spec = importlib.util.spec_from_file_location(
        "legacy_model_support_test", Path(__file__).resolve().parents[2] / "tools/legacy_model_support.py"
    )
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MarginEstimator:
    def predict(self, x, *, output_margin=False):
        assert output_margin
        np.testing.assert_allclose(x[0, 0], 2.0)
        assert np.isnan(x[0, 1])  # Unused module stays hidden.
        assert np.isnan(x[1, 0])  # Missing prediction stays hidden.
        np.testing.assert_allclose(x[1, 1], 5.0)
        return np.asarray([np.log(7.0), np.log(11.0)], np.float32)


def test_hr_bridge_uses_xgboost_cpu_expf_at_tree_split_boundaries():
    import ctypes

    pytest.importorskip("xgboost")
    from xgboost.core import _LIB

    expf = _LIB.expf
    expf.argtypes = [ctypes.c_float]
    expf.restype = ctypes.c_float
    margins = np.linspace(-2, 2, 101, dtype=np.float32)
    expected = np.asarray([expf(float(value)) for value in margins], np.float32)
    # Close float values can lie on opposite sides of a saved tree cut.
    np.testing.assert_array_equal(support().cox_margin_to_legacy_hr(margins), expected)


def test_cox_bridge_hides_unused_and_missing_hr_features_and_returns_margin():
    pytest.importorskip("xgboost")
    source = SimpleNamespace(
        M=2,
        D=2,
        models=[MarginEstimator(), None],
        active_module_indices=[np.asarray([0, 1]), np.asarray([])],
        standardize=lambda hr: (hr - 1) / 2,
    )
    bridge = support().LegacyCoxFusion(source)
    scores = np.log(np.asarray([[[2, 3], [4, 5]], [[3, 4], [5, 6]]], np.float32))
    mask = np.ones_like(scores, bool)
    mask[1, 0, 0] = False
    observed = np.asarray([[True, False], [True, True]])
    margin = bridge.predict(scores, pred_mask_nmd=mask, used_mask_nm=observed)
    np.testing.assert_allclose(np.exp(margin[:, 0]), [7, 11], rtol=1e-6)
    assert np.isnan(margin[:, 1]).all()
    np.testing.assert_allclose(bridge.standardize(margin)[:, 0], [3, 5], rtol=1e-6)


def test_cox_bridge_rejects_wrong_axes():
    bridge = support().LegacyCoxFusion(SimpleNamespace(M=2, D=2))
    with pytest.raises(ValueError, match="axis mismatch"):
        bridge.predict(np.zeros((2, 3, 2)))


def test_fusion_load_does_not_mmap_thousands_of_small_arrays(monkeypatch, tmp_path):
    module = support()
    calls = []
    monkeypatch.setattr(module, "bind_calibrated_class", lambda code: None)
    monkeypatch.setattr(module.joblib, "load", lambda path, **kwargs: calls.append(kwargs) or object())
    module.load_original(tmp_path / "fusion_model.joblib", tmp_path)
    module.load_original(tmp_path / "q_model.joblib", tmp_path)
    assert calls == [{"mmap_mode": None}, {"mmap_mode": "r"}]
