import json

import numpy as np
import pytest

from trace_agent.training.service import dataset_from_config


def _workflow():
    from trace_agent.cli import consistency_workflow

    return consistency_workflow


def _inputs(root, task="prevalent"):
    labels = np.array([[0], [1], [1], [0], [0], [1], [0], [1]], dtype=np.float32)
    eids = np.arange(10, 18)
    scores = np.full((8, 1, 1), 0.7, np.float32)
    for name, values in {
        "eids": eids,
        "modules": np.array(["M1"]),
        "diseases": np.array(["D1"]),
        "oof_probs" if task == "prevalent" else "scores": scores,
        "pred_mask": np.ones_like(scores, bool),
        "avail": np.ones((8, 1), bool),
        "y": labels,
        "y_mask": np.ones((8, 1), bool),
        "event": labels,
        "time": np.full((8, 1), 2.0, np.float32),
        "signed_time": np.where(labels == 1, 2.0, -2.0),
        "label_mask": np.ones((8, 1), bool),
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
        "final_scores": np.full((8, 1), 0.7 if task == "prevalent" else -2.0),
        "participant_ids": eids,
        "prediction_endpoints": np.array(["D1"]),
        "final_valid_mask": np.ones((8, 1), bool),
    }.items():
        np.save(output / f"{name}.npy", values)
    np.savez(
        output / "consistency_scores.npz",
        participant_ids=eids,
        endpoints=np.array(["D1"]),
        kg=labels,
        feature=np.ones((8, 1)),
        evidence=labels,
    )
    np.savez(
        output / "horizon_risks.npz",
        participant_ids=eids,
        endpoints=np.array(["D1"]),
        valid_mask=np.ones((8, 1), bool),
        **{"5y": np.full((8, 1), 0.7)},
    )
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "task": task,
                "endpoints": ["D1"],
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
        times = np.array([[8], [2], [2], [8], [8], [2], [1], [8]], dtype=np.float32)
        np.save(root / "cohort_time.npy", times)
        np.save(root / "cohort_signed_time.npy", np.where(labels == 1, times, -times))
    return config


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
