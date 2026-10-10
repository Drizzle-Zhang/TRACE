import json

import numpy as np
import pytest

from tests.unit.test_cli import _evaluation_config
from trace_agent.cli.operations import evaluate
from trace_agent.planning.rewards.longitudinal import build_horizon_labels


def _config(root):
    config = _evaluation_config(root)
    config["task"] = "longitudinal"
    # Four test rows: early event, late event, early censoring, censoring at 5y.
    event = np.array([[0], [1], [1], [1], [0], [0]], dtype=np.float32)
    time = np.array([[12], [2], [2], [8], [1], [5]], dtype=np.float32)
    np.save(root / "cohort_event.npy", event)
    np.save(root / "cohort_time.npy", time)
    np.save(root / "cohort_signed_time.npy", np.where(event == 1, time, -time))
    np.save(root / "cohort_label_mask.npy", np.ones((6, 1), dtype=bool))
    np.save(root / "cohort_scores.npy", np.zeros((6, 1, 1), dtype=np.float32))
    (root / "splits.csv").write_text(
        "row_index,eid,split\n0,10,train\n1,11,validation\n" "2,12,test\n3,13,test\n4,14,test\n5,15,test\n"
    )
    # Deliberately impossible probabilities in the raw Cox score file.
    np.save(root / "predictions.npy", np.full((6, 1), -2.0))
    horizon_path = root / "horizons.npz"
    np.savez(
        horizon_path,
        participant_ids=np.arange(10, 16),
        endpoints=np.array(["D1"]),
        valid_mask=np.ones((6, 1), dtype=bool),
        **{"5y": np.array([[0.1], [0.9], [0.8], [0.2], [0.9], [0.1]])},
    )
    config["evaluation"]["horizon_predictions_path"] = str(horizon_path)
    return config


def test_incident_evaluation_uses_horizon_probabilities_and_censoring(tmp_path):
    report = json.loads(evaluate(_config(tmp_path)).read_text())
    assert report["task"] == "longitudinal"
    assert report["participants"] == 4
    metrics = report["horizons"]["5y"]
    assert metrics["n_valid"] == 3
    assert metrics["endpoints"][0]["positive"] == 1
    assert metrics["endpoints"][0]["negative"] == 2
    assert metrics["macro_auroc"] == 1.0
    assert metrics["masked_log_loss"] == pytest.approx(-np.log([0.8, 0.8, 0.9]).mean())


def test_censoring_changes_horizon_evaluation_counts(tmp_path):
    config = _config(tmp_path)
    first = json.loads(evaluate(config).read_text())
    times = np.load(tmp_path / "cohort_time.npy")
    times[4] = 12
    events = np.load(tmp_path / "cohort_event.npy")
    np.save(tmp_path / "cohort_time.npy", times)
    np.save(tmp_path / "cohort_signed_time.npy", np.where(events == 1, times, -times))
    second = json.loads(evaluate(config).read_text())
    assert first["horizons"]["5y"]["n_valid"] == 3
    assert second["horizons"]["5y"]["n_valid"] == 4
    assert second["horizons"]["5y"]["masked_log_loss"] != first["horizons"]["5y"]["masked_log_loss"]


@pytest.mark.parametrize("change", ["eids", "endpoints", "shape", "range", "nan", "missing"])
def test_incident_evaluation_rejects_invalid_horizon_artifacts(tmp_path, change):
    config = _config(tmp_path)
    path = tmp_path / "horizons.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    if change == "eids":
        payload["participant_ids"] = payload["participant_ids"][::-1]
    elif change == "endpoints":
        payload["endpoints"] = np.array(["D2"])
    elif change == "shape":
        payload["5y"] = payload["5y"][:-1]
    elif change == "range":
        payload["5y"][2] = 1.5
    elif change == "nan":
        payload["5y"][2] = np.nan
    elif change == "missing":
        payload.pop("5y")
    np.savez(path, **payload)
    with pytest.raises(ValueError):
        evaluate(config)


def test_incident_evaluation_cannot_fall_back_to_cox_log_risk(tmp_path):
    config = _config(tmp_path)
    (tmp_path / "horizons.npz").unlink()
    with pytest.raises(FileNotFoundError):
        evaluate(config)


def test_horizon_boundary_includes_censoring_at_horizon():
    labels, known = build_horizon_labels(
        event=np.array([0, 1, 1]),
        time=np.array([5, 5, 8]),
        label_mask=np.ones(3, dtype=bool),
        horizons=np.array([5]),
    )
    np.testing.assert_array_equal(labels[:, 0], [0, 1, 0])
    assert known.all()


def test_incident_evaluation_preserves_multiple_endpoint_and_horizon_axes(tmp_path):
    config = _config(tmp_path)
    for name in ("event", "time", "signed_time", "label_mask"):
        path = tmp_path / f"cohort_{name}.npy"
        np.save(path, np.repeat(np.load(path), 2, axis=1))
    np.save(tmp_path / "cohort_diseases.npy", np.array(["D1", "D2"]))
    np.save(tmp_path / "cohort_scores.npy", np.zeros((6, 1, 2)))
    np.save(tmp_path / "cohort_pred_mask.npy", np.ones((6, 1, 2), dtype=bool))
    np.savez(
        tmp_path / "horizons.npz",
        participant_ids=np.arange(10, 16),
        endpoints=np.array(["D1", "D2"]),
        valid_mask=np.ones((6, 2), dtype=bool),
        **{"5y": np.full((6, 2), 0.2), "10y": np.full((6, 2), 0.6)},
    )
    report = json.loads(evaluate(config).read_text())
    assert report["horizons"]["5y"]["n_valid"] == 6
    assert report["horizons"]["10y"]["n_valid"] == 4
    assert [entry["positive"] for entry in report["horizons"]["5y"]["endpoints"]] == [1, 1]
    assert [entry["positive"] for entry in report["horizons"]["10y"]["endpoints"]] == [2, 2]


def test_incident_evaluation_honors_prediction_validity_mask(tmp_path):
    config = _config(tmp_path)
    with np.load(tmp_path / "horizons.npz", allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    payload["valid_mask"][2] = False
    payload["5y"][2] = np.nan
    np.savez(tmp_path / "horizons.npz", **payload)
    report = json.loads(evaluate(config).read_text())
    assert report["horizons"]["5y"]["n_valid"] == 2
    assert report["horizons"]["5y"]["macro_auroc"] is None


@pytest.mark.parametrize("name", ["5", "0y", "-1y", "nany", "infy", "fivey"])
def test_incident_evaluation_rejects_ambiguous_horizon_names(tmp_path, name):
    config = _config(tmp_path)
    with np.load(tmp_path / "horizons.npz", allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    payload[name] = payload.pop("5y")
    np.savez(tmp_path / "horizons.npz", **payload)
    with pytest.raises(ValueError, match="horizon"):
        evaluate(config)
