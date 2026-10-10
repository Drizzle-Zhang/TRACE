"""Shipped follow-up workflows retain frozen settings and select the right cohort."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.module_fixtures import write_full_module_pipeline
from trace_agent.cli.main import main
from trace_agent.runtime.config import load_config

PROJECT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
def test_environment_evaluation_templates_reuse_prediction_models(monkeypatch, tmp_path, task):
    values = {
        "TRACE_INPUT_DIR": str(tmp_path / "scores"),
        "TRACE_INPUT_PREFIX": "cohort",
        "TRACE_SPLIT_TABLE": str(tmp_path / "splits.csv"),
        "TRACE_FUSION_MODEL": str(tmp_path / "models/fusion.joblib"),
        "TRACE_Q_MODEL": str(tmp_path / "models/planner.joblib"),
        "TRACE_CALIBRATORS": str(tmp_path / "models/calibration.joblib"),
        "TRACE_KNOWLEDGE_MATRIX": str(tmp_path / "knowledge.csv"),
        "TRACE_OUTPUT_DIR": str(tmp_path / "evaluation"),
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    config = load_config(PROJECT / f"pipelines/{task}/evaluate.yaml")
    inference = load_config(PROJECT / f"pipelines/{task}/infer.yaml")
    assert config["pipeline"]["steps"] == ["infer", "evaluate"]
    assert config["artifacts"] == inference["artifacts"]
    assert config["data"]["split_table"]["path"] == values["TRACE_SPLIT_TABLE"]
    assert config["output"]["directory"] == values["TRACE_OUTPUT_DIR"]


def _template(name):
    path = PROJECT / "configs/examples" / name
    assert path.is_file(), f"missing shipped workflow: {path.name}"
    return path


def _copied_templates(root):
    for directory in ("configs", "pipelines"):
        shutil.copytree(PROJECT / directory, root / directory)
    destination = root / "data/configs"
    destination.mkdir(parents=True)
    for source in (root / "configs/examples").glob("*.yaml"):
        shutil.copy(source, destination / source.name)
    return destination


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
def test_evaluation_templates_reuse_frozen_training_inputs(task):
    config = load_config(_template(f"{task}-evaluate.yaml"))
    training = load_config(_template(f"{task}-train.yaml"))

    assert config["pipeline"]["steps"] == ["infer", "evaluate"]
    assert config["data"] == training["data"]
    assert config["artifacts"] == training["artifacts"]
    assert config["output"]["directory"] == str(PROJECT / f"data/study/{task}/cohort_evaluation")


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
def test_deployment_templates_inherit_validation_settings_and_new_cohort_paths(tmp_path, task):
    _template(f"{task}-new-consistency.yaml")
    directory = _copied_templates(tmp_path)
    validation_path = directory / f"{task}-consistency.yaml"
    values = yaml.safe_load(validation_path.read_text())
    custom_paths = {
        "evidence_bundle": "custom/evidence",
        "feature_weights": "custom/weights.npy",
        "feature_names": "custom/names.npy",
        "feature_schema_path": "custom/schema.json",
        "rules_path": "custom/rules.json",
    }
    values["consistency"].update(custom_paths)
    values["consistency"]["thresholds"] = {"kg": 0.23}
    values["artifacts"].update(
        fusion_model="custom/fusion.joblib",
        q_model="custom/planner.joblib",
        knowledge_matrix="custom/knowledge.npy",
        burden_profile="custom/frozen_burden.yaml",
    )
    if task == "longitudinal":
        values["consistency"].update(horizon="5y", longitudinal_scale_path="custom/scale.json")
    validation_path.write_text(yaml.safe_dump(values))

    validation = load_config(validation_path)
    prediction = load_config(directory / f"{task}-predict.yaml")
    deployment = load_config(directory / f"{task}-new-consistency.yaml")

    assert deployment["pipeline"]["steps"] == ["score-modules", "infer", "apply-consistency"]
    assert deployment["module_data"] == prediction["module_data"]
    for key in ("input_dir", "prefix", "split_table"):
        assert deployment["data"][key] == prediction["data"][key]
    assert deployment["data"]["split_table"] is None
    assert deployment["output"] == prediction["output"]
    for key, value in validation["artifacts"].items():
        assert deployment["artifacts"][key] == value
    for key in (*custom_paths, "thresholds", "horizon", "longitudinal_scale_path"):
        if key in validation["consistency"]:
            assert deployment["consistency"][key] == validation["consistency"][key]
    assert deployment["consistency"]["enabled"] is True
    assert deployment["consistency"]["contributions"] == str(
        tmp_path / f"data/study/{task}/new_consistency_inputs/contributions.npy"
    )


@pytest.mark.parametrize("priority", ["performance", "balanced", "cost"])
def test_trained_survival_deployment_reuses_scale_without_validation_rows(priority):
    root = PROJECT / "configs/trained_models"
    path = root / f"survival-{priority}-consistency-predict.yaml"
    assert path.is_file(), f"missing shipped workflow: {path.name}"
    deployment = load_config(path)
    validation = load_config(root / f"survival-{priority}-consistency.yaml")

    assert deployment["pipeline"]["steps"] == ["score-modules", "consistency"]
    assert deployment["data"]["split_table"] is None
    assert deployment["artifacts"] == validation["artifacts"]
    assert deployment["planner"] == validation["planner"]
    assert deployment["consistency"] == validation["consistency"]


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
def test_new_consistency_pipeline_runs_after_outcomes_and_splits_are_removed(tmp_path, task):
    _template(f"{task}-new-consistency.yaml")
    directory = _copied_templates(tmp_path)
    study = tmp_path / "study"
    training_path = write_full_module_pipeline(study, task=task)
    training = yaml.safe_load(training_path.read_text())
    copied_train = directory / f"{task}-train.yaml"
    copied_train.write_text(yaml.safe_dump(training))
    assert main(["run", "--config", str(copied_train)]) == 0

    for name, values in {
        "contributions": np.ones((180, 1, 1), dtype=np.float32),
        "feature_weights": np.ones((1, 1), dtype=np.float32),
        "feature_names": np.array(["a"]),
    }.items():
        np.save(study / f"{name}.npy", values)
    validation_path = directory / f"{task}-consistency.yaml"
    validation = yaml.safe_load(validation_path.read_text())
    validation.update(data=training["data"], artifacts=training["artifacts"])
    validation["consistency"].update(
        contributions=str(study / "contributions.npy"),
        feature_weights=str(study / "feature_weights.npy"),
        feature_names=str(study / "feature_names.npy"),
        evidence_bundle=None,
    )
    validation_path.write_text(yaml.safe_dump(validation))
    assert main(["run", "--config", str(validation_path)]) == 0
    frozen = load_config(validation_path)
    scale_path = frozen["consistency"].get("longitudinal_scale_path")
    original_scale = Path(scale_path).read_bytes() if scale_path else None

    prediction_path = directory / f"{task}-predict.yaml"
    prediction = yaml.safe_load(prediction_path.read_text())
    prediction["module_data"]["catalog_path"] = str(study / "catalog.yaml")
    prediction_path.write_text(yaml.safe_dump(prediction))
    deployment_path = directory / f"{task}-new-consistency.yaml"
    deployment = yaml.safe_load(deployment_path.read_text())
    deployment["consistency"]["contributions"] = str(study / "contributions.npy")
    deployment_path.write_text(yaml.safe_dump(deployment))
    (study / "outcome.csv").unlink()
    (study / "splits.csv").unlink()

    assert main(["run", "--config", str(deployment_path)]) == 0

    config = load_config(deployment_path)
    output = Path(config["output"]["directory"])
    assert (output / "consistency_predictions.npz").is_file()
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["participants"] == 180
    if scale_path:
        assert Path(scale_path).read_bytes() == original_scale
