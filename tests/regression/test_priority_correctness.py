"""Regression cases for the six confirmed formal-release correctness findings."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.module_fixtures import write_full_module_pipeline
from tests.training_fixtures import rewrite_training_artifact
from trace_agent.cli.main import main
from trace_agent.cli.operations import evaluate, run_inference
from trace_agent.data.arrays import select_participant
from trace_agent.runtime.artifacts import build_trace_agent
from trace_agent.runtime.config import load_config
from trace_agent.runtime.model_artifact import load_trace_artifact
from trace_agent.training.modules.service import prepare_module_data, score_modules, train_modules
from trace_agent.training.service import calibrate, dataset_from_config, train_integration, train_planner


def _scored_config(root, task="prevalent"):
    config = load_config(write_full_module_pipeline(root, task=task))
    prepare_module_data(config)
    train_modules(config)
    score_modules(config)
    return config


@pytest.mark.parametrize(
    "task,command",
    [
        ("prevalent", train_integration),
        ("prevalent", calibrate),
        ("prevalent", train_planner),
        ("longitudinal", train_integration),
        ("longitudinal", train_planner),
    ],
)
def test_downstream_rejects_reordered_native_targets(tmp_path, command, task):
    config = _scored_config(tmp_path, task)
    if command is not train_integration:
        train_integration(config)
    if command is train_planner and task == "prevalent":
        calibrate(config)
    name = "y" if task == "prevalent" else "event"
    path = tmp_path / f"prepared/cohort_{name}.npy"
    labels = np.load(path)
    labels[:120] = np.roll(labels[:120], 1, axis=0)
    np.save(path, labels)
    with pytest.raises(ValueError, match="target.*hash"):
        command(config)


@pytest.mark.parametrize("change", ["missing", "eids", "endpoints", "split"])
def test_native_target_manifest_identity_is_required(tmp_path, change):
    config = _scored_config(tmp_path)
    path = tmp_path / "prepared/cohort_module_targets_manifest.json"
    if change == "missing":
        path.unlink()
    else:
        manifest = json.loads(path.read_text())
        manifest[{"eids": "eids_sha256", "endpoints": "endpoints", "split": "split_sha256"}[change]] = []
        path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError), match="target"):
        dataset_from_config(config)


@pytest.mark.parametrize("command", [train_integration, calibrate, train_planner])
def test_training_rejects_native_inference_scores(tmp_path, command):
    config = _scored_config(tmp_path)
    if command is not train_integration:
        train_integration(config)
    if command is train_planner:
        calibrate(config)
    config["module_data"]["score_mode"] = "inference"
    score_modules(config)
    with pytest.raises(ValueError, match="OOF|score_mode"):
        command(config)


@pytest.mark.parametrize("field", ["split_sha256", "train_rows_sha256", "train_eids_sha256", "score_mode"])
@pytest.mark.parametrize("change", ["wrong", "missing"])
def test_native_oof_provenance_is_bound_to_the_split(tmp_path, field, change):
    config = _scored_config(tmp_path)
    path = tmp_path / "prepared/cohort_score_bundle_manifest.json"
    manifest = json.loads(path.read_text())
    if change == "missing":
        manifest["provenance"].pop(field, None)
    else:
        manifest["provenance"][field] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="OOF"):
        dataset_from_config(config)


def test_native_targets_do_not_silently_accept_missing_score_manifest(tmp_path):
    config = _scored_config(tmp_path)
    (tmp_path / "prepared/cohort_score_bundle_manifest.json").unlink()
    with pytest.raises(ValueError, match="score manifest"):
        dataset_from_config(config)


@pytest.mark.parametrize("axis", ["modules", "endpoints"])
def test_runtime_rejects_wrong_q_model_axis(tmp_path, axis):
    config_path = write_full_module_pipeline(tmp_path, task="prevalent")
    assert main(["run", "--config", str(config_path)]) == 0
    config = load_config(config_path)
    dataset = dataset_from_config(config)
    path = config["artifacts"]["q_model"]
    artifact = load_trace_artifact(path)
    names = tuple(reversed(artifact.modules)) if axis == "modules" else ("OTHER_ENDPOINT",)
    rewrite_training_artifact(replace(artifact, **{axis: names}), path)
    with pytest.raises(ValueError, match="Q-model.*order"):
        build_trace_agent(config, batch=dataset.scores)


def test_default_prevalent_inference_loads_trained_calibration(tmp_path, monkeypatch):
    config_path = write_full_module_pipeline(tmp_path, task="prevalent")
    assert main(["run", "--config", str(config_path)]) == 0
    train_config = load_config(config_path)
    for key, value in {
        "TRACE_INPUT_DIR": tmp_path / "prepared",
        "TRACE_INPUT_PREFIX": "cohort",
        "TRACE_FUSION_MODEL": tmp_path / "output/fusion_model.joblib",
        "TRACE_Q_MODEL": tmp_path / "output/q_model.joblib",
        "TRACE_CALIBRATORS": tmp_path / "output/calibration_model.joblib",
        "TRACE_KNOWLEDGE_MATRIX": tmp_path / "kg.npy",
        "TRACE_OUTPUT_DIR": tmp_path / "inference",
    }.items():
        monkeypatch.setenv(key, str(value))
    config = load_config(Path("pipelines/prevalent/infer.yaml"))
    assert config["artifacts"].get("calibrators") == train_config["artifacts"]["calibrators"]
    config["artifacts"]["burden_profile"] = train_config["artifacts"]["burden_profile"]
    batch = dataset_from_config(train_config).scores
    agent = build_trace_agent(config, batch=batch)
    result = agent.assess(select_participant(batch, 0), initial_used_modules=np.ones(2, dtype=bool))
    profile = result.planning.final_profile
    assert profile.calibrated_scores is not None
    assert not np.allclose(profile.raw_scores, profile.calibrated_scores)


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
def test_native_full_pipeline_remains_evaluable(tmp_path, task):
    config_path = write_full_module_pipeline(tmp_path, task=task)
    assert main(["run", "--config", str(config_path)]) == 0
    config = load_config(config_path)
    # Evaluation is allowed on frozen inference scores, unlike training.
    config["module_data"]["score_mode"] = "inference"
    score_modules(config)
    config["output"]["top_endpoints"] = 1
    run_inference(config)
    report = json.loads(evaluate(config).read_text())
    assert report["participants"] == 30
    if task == "longitudinal":
        assert set(report["horizons"]) == {"5y", "10y"}
        with np.load(tmp_path / "output/horizon_risks.npz", allow_pickle=False) as archive:
            assert archive["5y"].shape == (180, 1)
            assert archive["participant_ids"].size == 180
            assert archive["endpoints"].tolist() == ["D1"]
        assert report["horizons"]["5y"]["n_valid"] == 30
