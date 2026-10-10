"""Native prediction must retain the fitted LightGBM CPU allocation."""

import numpy as np
from lightgbm import Booster, LGBMClassifier, LGBMRegressor

from trace_agent.training.planner import ActionValueModel
from trace_agent.training.prevalent.integration import _positive_probability


def _observe_prediction_threads(monkeypatch):
    calls = []
    original = Booster.predict

    def predict(self, data, *args, **kwargs):
        calls.append(kwargs.get("num_threads"))
        return original(self, data, *args, **kwargs)

    monkeypatch.setattr(Booster, "predict", predict)
    return calls


def test_fusion_prediction_respects_fitted_thread_budget(monkeypatch):
    rows = np.arange(90, dtype=np.float32).reshape(30, 3)
    estimator = LGBMClassifier(n_estimators=5, n_jobs=2, min_child_samples=2, verbosity=-1)
    estimator.fit(rows, np.arange(30) % 2)
    expected = estimator.booster_.predict(rows, num_threads=2).astype(np.float32)
    calls = _observe_prediction_threads(monkeypatch)

    actual = _positive_probability(estimator, rows)

    assert calls == [2]
    np.testing.assert_array_equal(actual, expected)


def test_q_prediction_respects_fitted_thread_budget(monkeypatch):
    rows = np.arange(90, dtype=np.float32).reshape(30, 3)
    estimator = LGBMRegressor(n_estimators=5, n_jobs=2, min_child_samples=2, verbosity=-1)
    estimator.fit(rows, np.arange(30, dtype=np.float32) / 30)
    expected = estimator.booster_.predict(rows, num_threads=2).astype(np.float32)
    calls = _observe_prediction_threads(monkeypatch)

    actual = ActionValueModel(estimator, feature_width=3).predict(rows)

    assert calls == [2]
    np.testing.assert_array_equal(actual, expected)
