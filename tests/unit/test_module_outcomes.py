from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.module_fixtures import write_module_inputs


def test_prevalent_excludes_future_events():
    from trace_agent.data.module_outcomes import encode_outcomes

    targets = encode_outcomes(
        np.array([0.0, 1.0, 1.0, np.nan, 0.0, 1.0]),
        np.array([12.0, -2.0, 3.0, np.nan, np.nan, np.nan]),
        task="prevalent",
    )
    assert targets.label_mask[:, 0].tolist() == [True, True, False, False, True, False]
    assert targets.labels[:2, 0].tolist() == [0.0, 1.0]
    assert np.isnan(targets.labels[2:4]).all()


def test_incident_encodes_censoring_and_excludes_baseline_cases():
    from trace_agent.data.module_outcomes import encode_outcomes

    targets = encode_outcomes(
        np.array([0.0, 1.0, 1.0, np.nan, 0.0]),
        np.array([12.0, 3.0, -2.0, np.nan, np.nan]),
        task="incident",
    )
    assert targets.label_mask[:, 0].tolist() == [True, True, False, False, False]
    assert targets.signed_time[:2, 0].tolist() == [-12.0, 3.0]
    np.testing.assert_allclose(np.abs(targets.signed_time[:2]), targets.event_time[:2])
    assert np.isnan(targets.event_time[2:]).all()


@pytest.mark.parametrize(
    ("target", "time", "task", "message"),
    [
        (1.0, 0.0, "prevalent", "zero"),
        (1.0, 0.0, "longitudinal", "zero"),
        (0.0, -1.0, "longitudinal", "positive"),
        (0.0, 0.0, "longitudinal", "positive"),
        (2.0, 3.0, "prevalent", "0 or 1"),
        (-1.0, 3.0, "longitudinal", "0 or 1"),
        (np.inf, 3.0, "prevalent", "infinity"),
        (1.0, np.inf, "longitudinal", "infinity"),
    ],
)
def test_outcomes_reject_invalid_encoding(target, time, task, message):
    from trace_agent.data.module_outcomes import encode_outcomes

    with pytest.raises(ValueError, match=message):
        encode_outcomes(np.array([target]), np.array([time]), task=task)


def test_outcome_shape_mismatch_errors():
    from trace_agent.data.module_outcomes import encode_outcomes

    with pytest.raises(ValueError, match="shape"):
        encode_outcomes(np.zeros(2), np.zeros(3), task="prevalent")


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
def test_outcomes_explicitly_align_eids_and_keep_missing_rows(tmp_path, task):
    from trace_agent.data.module_features import load_catalog, load_module_features
    from trace_agent.data.module_outcomes import load_module_outcomes

    path = write_module_inputs(tmp_path, task=task)
    frame = pd.read_csv(tmp_path / "outcome.csv").iloc[1:][::-1]
    frame.to_csv(tmp_path / "outcome.csv", index=False)
    features = load_module_features(load_catalog(path))
    targets = load_module_outcomes(features.catalog, features.participant_ids, task=task)
    assert targets.labels.shape == (180, 1)
    assert not targets.label_mask[0, 0]
    assert targets.label_mask[1:, 0].all()
    assert targets.labels[1, 0] == 1


@pytest.mark.parametrize("change", ["duplicate", "unknown"])
def test_outcome_eid_errors(tmp_path, change):
    from trace_agent.data.module_features import load_catalog, load_module_features
    from trace_agent.data.module_outcomes import load_module_outcomes

    path = write_module_inputs(tmp_path)
    frame = pd.read_csv(tmp_path / "outcome.csv")
    frame.loc[0, "eid"] = frame.loc[1, "eid"] if change == "duplicate" else 999999
    frame.to_csv(tmp_path / "outcome.csv", index=False)
    features = load_module_features(load_catalog(path))
    with pytest.raises(ValueError, match=change):
        load_module_outcomes(features.catalog, features.participant_ids, task="prevalent")
