"""Restore the frozen source's optional validation-only Cox round selection."""

import numpy as np
import pytest

from tests.unit.test_longitudinal_training import _view
from tests.unit.test_training_service import _write_training_bundle
from trace_agent.contracts.task import SplitRole
from trace_agent.runtime.model_artifact import load_trace_artifact
from trace_agent.training.longitudinal.integration import (
    LongitudinalIntegrationConfig,
    train_longitudinal_integration,
)
from trace_agent.training.service import train_integration


def test_cox_early_stopping_uses_only_validation_for_selection(monkeypatch):
    xgb = pytest.importorskip("xgboost")
    train = _view(SplitRole.TRAIN, offset=0, n=24)
    validation = _view(SplitRole.VALIDATION, offset=100, n=12)
    # Opposite validation associations make further boosting worsen held-out loss.
    validation.targets.labels[:] = 1 - validation.targets.labels
    validation.targets.signed_time[:] *= -1
    validation.targets.label_mask[0, 0] = False
    calls = []
    original = xgb.XGBRegressor.fit

    def recording_fit(self, x, y, **kwargs):
        calls.append((x.copy(), y.copy(), kwargs.get("eval_set")))
        return original(self, x, y, **kwargs)

    monkeypatch.setattr(xgb.XGBRegressor, "fit", recording_fit)
    result = train_longitudinal_integration(
        train,
        validation,
        config=LongitudinalIntegrationConfig(
            subsets_per_participant=0,
            min_events=2,
            n_estimators=20,
            early_stopping_rounds=2,
        ),
    )
    assert len(calls) == 2
    for endpoint, (_, labels, eval_set) in enumerate(calls):
        np.testing.assert_array_equal(labels, train.targets.signed_time[:, endpoint])
        assert len(eval_set) == 1
        rows = validation.targets.label_mask[:, endpoint]
        np.testing.assert_array_equal(eval_set[0][1], validation.targets.signed_time[rows, endpoint])
        selected = result.metrics["selected_rounds"][train.scores.endpoints[endpoint]]
        assert selected == result.model.estimators[endpoint].best_iteration + 1
        assert 1 <= selected < 20
        estimator = result.model.estimators[endpoint]
        assert estimator.get_booster().num_boosted_rounds() < 20
        np.testing.assert_allclose(
            estimator.predict(calls[endpoint][0], output_margin=True),
            estimator.predict(calls[endpoint][0], output_margin=True, iteration_range=(0, selected)),
        )


def test_fixed_cox_rounds_remain_the_default_and_are_recorded():
    result = train_longitudinal_integration(
        _view(SplitRole.TRAIN, offset=0, n=12),
        _view(SplitRole.VALIDATION, offset=12, n=6),
        config=LongitudinalIntegrationConfig(min_events=2, n_estimators=3),
    )
    assert result.metrics["selected_rounds"] == {"D1": 3, "D2": 3}
    assert all(model.get_params().get("early_stopping_rounds") is None for model in result.model.estimators)


def test_validation_without_events_retains_fixed_rounds():
    validation = _view(SplitRole.VALIDATION, offset=12, n=6)
    validation.targets.label_mask[:] = False
    result = train_longitudinal_integration(
        _view(SplitRole.TRAIN, offset=0, n=12),
        validation,
        config=LongitudinalIntegrationConfig(min_events=2, n_estimators=3, early_stopping_rounds=2),
    )
    assert result.metrics["selected_rounds"] == {"D1": 3, "D2": 3}


def test_selected_cox_rounds_persist_in_native_artifact(tmp_path):
    config = _write_training_bundle(tmp_path / "data", task="longitudinal")
    config["training"]["integration"].update(min_events=1, early_stopping_rounds=2)
    result = train_integration(config)
    artifact = load_trace_artifact(result.artifact_path)
    assert artifact.metrics["selected_rounds"]["D1"] == artifact.model.estimators[0].best_iteration + 1
    assert artifact.training["settings"]["early_stopping_rounds"] == 2
    assert artifact.training["test_rows_used"] == 0


@pytest.mark.parametrize("patience", [-1, 1.5, True])
def test_early_stopping_patience_must_be_a_nonnegative_integer(patience):
    with pytest.raises(ValueError, match="early_stopping_rounds"):
        LongitudinalIntegrationConfig.from_mapping({"early_stopping_rounds": patience})


def test_fixed_round_training_keeps_validation_without_signed_targets_compatible():
    from dataclasses import replace

    validation = _view(SplitRole.VALIDATION, offset=12, n=6)
    validation = replace(validation, targets=replace(validation.targets, signed_time=None))
    result = train_longitudinal_integration(
        _view(SplitRole.TRAIN, offset=0, n=12),
        validation,
        config=LongitudinalIntegrationConfig(min_events=2, n_estimators=3),
    )
    assert result.metrics["selected_rounds"] == {"D1": 3, "D2": 3}
