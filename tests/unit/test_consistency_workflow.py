import csv
import json

import numpy as np
import pytest

from trace_agent.training.service import dataset_from_config


def _workflow():
    from trace_agent.cli import consistency_workflow

    return consistency_workflow


def _inputs(root, task="prevalent", *, eids=None, endpoints=("D1",)):
    base_labels = np.array([0, 1, 1, 0, 0, 1, 0, 1], dtype=np.float32)
    labels = np.column_stack(
        [base_labels if index % 2 == 0 else 1 - base_labels for index in range(len(endpoints))]
    )
    eids = np.arange(10, 18) if eids is None else np.asarray(eids)
    scores = np.full((8, 1, len(endpoints)), 0.7, np.float32)
    for name, values in {
        "eids": eids,
        "modules": np.array(["M1"]),
        "diseases": np.array(endpoints),
        "oof_probs" if task == "prevalent" else "scores": scores,
        "pred_mask": np.ones_like(scores, bool),
        "avail": np.ones((8, 1), bool),
        "y": labels,
        "y_mask": np.ones(labels.shape, bool),
        "event": labels,
        "time": np.full(labels.shape, 2.0, np.float32),
        "signed_time": np.where(labels == 1, 2.0, -2.0),
        "label_mask": np.ones(labels.shape, bool),
    }.items():
        np.save(root / f"cohort_{name}.npy", values)
    (root / "splits.csv").write_text(
        "row_index,eid,split\n"
        + "".join(
            f"{i},{eid},{'train' if i == 0 else 'validation' if i < 5 else 'test'}\n"
            for i, eid in enumerate(eids)
        )
    )
    output = root / "output"
    output.mkdir()
    for name, values in {
        "final_scores": np.full(labels.shape, 0.7 if task == "prevalent" else -2.0),
        "participant_ids": eids,
        "prediction_endpoints": np.array(endpoints),
        "final_valid_mask": np.ones(labels.shape, bool),
    }.items():
        np.save(output / f"{name}.npy", values)
    np.savez(
        output / "consistency_scores.npz",
        participant_ids=eids,
        endpoints=np.array(endpoints),
        kg=labels,
        feature=np.ones(labels.shape),
        evidence=labels,
    )
    np.savez(
        output / "horizon_risks.npz",
        participant_ids=eids,
        endpoints=np.array(endpoints),
        valid_mask=np.ones(labels.shape, bool),
        **{"5y": np.full(labels.shape, 0.7)},
    )
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "task": task,
                "endpoints": list(endpoints),
                "artifact_provenance": {
                    "fusion_model": {"sha256": "a" * 64},
                    "consistency_scale": {"sha256": "b" * 64},
                },
            }
        )
    )
    config = {
        "task": task,
        "data": {
            "input_dir": str(root),
            "prefix": "cohort",
            "split_table": {"path": str(root / "splits.csv")},
        },
        "output": {"directory": str(output)},
        "consistency": {
            "enabled": True,
            "threshold_grid": [0.2, 0.8],
            "min_validation_positives": 1,
            "min_validation_base_positives": 1,
        },
    }
    if task == "longitudinal":
        config["consistency"]["horizon"] = "5y"
        times = np.broadcast_to(
            np.array([[8], [2], [2], [8], [8], [2], [1], [8]], dtype=np.float32), labels.shape
        )
        np.save(root / "cohort_time.npy", times)
        np.save(root / "cohort_signed_time.npy", np.where(labels == 1, times, -times))
    return config


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
@pytest.mark.parametrize("custom_stem", [False, True])
def test_apply_exports_masked_binary_csvs_with_complete_axes(tmp_path, monkeypatch, task, custom_stem):
    api = _workflow()
    eids = ["001", "1", "NA", "sample-1", "α", "003", "z", "last"]
    endpoints = ("D,2", "D1")
    config = _inputs(tmp_path, task, eids=eids, endpoints=endpoints)
    api.calibrate_consistency(config)
    output = tmp_path / "output"
    valid = np.ones((8, 2), bool)
    valid[0, 1] = False
    valid[1, 0] = False
    valid[-1] = False
    np.save(output / "final_valid_mask.npy", valid)
    if task == "longitudinal":
        horizon_valid = np.ones_like(valid)
        horizon_valid[2, 0] = False
        np.savez(
            output / "horizon_risks.npz",
            participant_ids=eids,
            endpoints=endpoints,
            valid_mask=horizon_valid,
            **{"5y": np.full(valid.shape, 0.7)},
        )
        valid &= horizon_valid
    expected_path = output / "consistency_predictions.npz"
    if custom_stem:
        expected_path = tmp_path / "custom/classification.v2.npz"
        config["consistency"]["predictions_path"] = str(expected_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("CSV export opened training data")

    monkeypatch.setattr(api, "dataset_from_config", forbidden)
    config.pop("data")
    assert api.apply_consistency(config) == expected_path

    with np.load(expected_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["valid_mask"], valid)
        for name in ("baseline", "filtered"):
            csv_path = expected_path.with_name(f"{expected_path.stem}_{name}.csv")
            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            assert rows[0] == ["participant_id", *endpoints]
            assert [row[0] for row in rows[1:]] == eids
            assert [row[1:] for row in rows[1:]] == [
                [str(int(value)) if known else "" for value, known in zip(values, mask)]
                for values, mask in zip(archive[name], valid)
            ]


@pytest.mark.parametrize("change", ["order", "shape", "duplicate", "missing_components"])
def test_inference_identity_errors_use_participant_id_terminology(tmp_path, change):
    api = _workflow()
    config = _inputs(tmp_path)
    output = tmp_path / "output"
    if change in {"shape", "duplicate"}:
        ids = np.arange(10, 18)[None, :] if change == "shape" else np.repeat(10, 8)
        np.save(output / "participant_ids.npy", ids)
    else:
        with np.load(output / "consistency_scores.npz") as archive:
            arrays = {name: archive[name] for name in archive.files}
        if change == "order":
            arrays["participant_ids"] = arrays["participant_ids"][::-1]
        else:
            arrays.pop("participant_ids")
        np.savez(output / "consistency_scores.npz", **arrays)

    with pytest.raises(ValueError, match="participant IDs"):
        api._read_inference(config)


def test_validation_rules_ignore_test_labels_and_frozen_apply_is_label_free(tmp_path, monkeypatch):
    api = _workflow()
    config = _inputs(tmp_path)
    artifact = json.loads(api.calibrate_consistency(config).read_text())
    labels = np.load(tmp_path / "cohort_y.npy")
    labels[5:] = 1 - labels[5:]
    np.save(tmp_path / "cohort_y.npy", labels)
    second = json.loads(api.calibrate_consistency(config).read_text())
    assert artifact["rules"] == second["rules"]

    def forbidden(*args, **kwargs):
        raise AssertionError("frozen application read training data")

    monkeypatch.setattr(api, "dataset_from_config", forbidden)
    config.pop("data")
    path = api.apply_consistency(config)
    with np.load(path) as saved:
        assert saved["baseline"].sum() == 8
        assert saved["filtered"].sum() == 4


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
def test_test_only_before_after_metrics_and_censoring(tmp_path, task):
    api = _workflow()
    config = _inputs(tmp_path, task)
    api.calibrate_consistency(config)
    api.apply_consistency(config)
    dataset = dataset_from_config(config, require_training_scores=False)
    report = api.evaluate_frozen_consistency(config, dataset)
    assert report["split"] == "test"
    assert report["participants"] == 3
    assert report["baseline"]["n_valid"] == (3 if task == "prevalent" else 2)
    if task == "prevalent":
        assert report["baseline"]["micro"]["f1"] == pytest.approx(0.8)
        assert report["filtered"]["micro"]["f1"] == 1.0
    else:
        assert report["horizon"] == "5y"
        assert report["baseline"]["micro"]["tp"] == 1
        assert report["baseline"]["micro"]["fp"] == 1


@pytest.mark.parametrize("change", ["eids", "endpoints", "model", "horizon"])
def test_apply_rejects_wrong_axes_model_or_horizon(tmp_path, change):
    api = _workflow()
    config = _inputs(tmp_path, "longitudinal")
    api.calibrate_consistency(config)
    output = tmp_path / "output"
    if change == "model":
        path = output / "run_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["artifact_provenance"]["fusion_model"]["sha256"] = "c" * 64
        path.write_text(json.dumps(manifest))
    elif change == "horizon":
        config["consistency"]["horizon"] = "10y"
    else:
        path = output / "consistency_scores.npz"
        with np.load(path) as archive:
            values = {key: archive[key] for key in archive.files}
        key = "participant_ids" if change == "eids" else "endpoints"
        values[key] = values[key][::-1] if change == "eids" else np.array(["D2"])
        np.savez(path, **values)
    with pytest.raises(ValueError):
        api.apply_consistency(config)


def test_evaluation_rejects_stale_filtered_predictions(tmp_path):
    api = _workflow()
    config = _inputs(tmp_path)
    api.calibrate_consistency(config)
    api.apply_consistency(config)
    np.save(tmp_path / "output/final_scores.npy", np.full((8, 1), 0.2))
    with pytest.raises(ValueError, match="stale|input"):
        api.evaluate_frozen_consistency(config, dataset_from_config(config, require_training_scores=False))


def test_longitudinal_never_falls_back_to_raw_scores(tmp_path):
    api = _workflow()
    config = _inputs(tmp_path, "longitudinal")
    (tmp_path / "output/horizon_risks.npz").unlink()
    with pytest.raises(FileNotFoundError):
        api.calibrate_consistency(config)


def test_longitudinal_cli_searches_original_hard_kg_and_freezes_the_selector(tmp_path, monkeypatch):
    api = _workflow()
    config = _inputs(tmp_path, "longitudinal")
    output = tmp_path / "output"
    kg = np.load(tmp_path / "cohort_event.npy")
    np.savez(
        output / "consistency_scores.npz",
        participant_ids=np.arange(10, 18),
        endpoints=["D1"],
        kg=kg,
        feature=np.ones_like(kg),
        evidence=1 - kg,
    )
    rules = json.loads(api.calibrate_consistency(config).read_text())
    assert rules["schema"] == "trace.survival_consistency_policies.v1"
    assert any(row["kind"] == "hard_kg" for row in rules["all_strategy_specs"].values())
    assert rules["rules"][0]["validation_selected_f1"] == 1
    original_rules = rules["rules"]
    event = np.load(tmp_path / "cohort_event.npy")
    event[5:] = 1 - event[5:]
    np.save(tmp_path / "cohort_event.npy", event)
    time = np.load(tmp_path / "cohort_time.npy")
    np.save(tmp_path / "cohort_signed_time.npy", np.where(event == 1, time, -time))
    assert json.loads(api.calibrate_consistency(config).read_text())["rules"] == original_rules

    def forbidden(*args, **kwargs):
        raise AssertionError("applying survival rules opened labels")

    monkeypatch.setattr(api, "dataset_from_config", forbidden)
    api.apply_consistency(config)
    with np.load(output / "consistency_predictions.npz") as saved:
        assert saved["filtered"][1:5, 0].tolist() == [True, True, False, False]
