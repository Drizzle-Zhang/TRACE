"""Configuration, artifact dependency, and inference regression cases."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.training_fixtures import rewrite_training_artifact
from tests.unit.test_training_service import _write_training_bundle
from trace_agent.cli.operations import run_inference
from trace_agent.integration.absolute_risk import CoxAbsoluteRiskMapper
from trace_agent.runtime.artifacts import build_trace_agent, load_absolute_risk_mapper
from trace_agent.runtime.config import load_config
from trace_agent.runtime.model_artifact import load_trace_artifact
from trace_agent.training.service import calibrate, dataset_from_config, train_integration, train_planner


def test_inherited_paths_belong_to_the_file_that_declares_them(tmp_path):
    shared = tmp_path / "shared"
    child = tmp_path / "pipeline"
    shared.mkdir()
    child.mkdir()
    (shared / "base.yaml").write_text("artifacts:\n  burden_profile: resources/burden.yaml\n")
    path = child / "infer.yaml"
    path.write_text("extends: ../shared/base.yaml\noutput:\n  directory: results\n")
    config = load_config(path)
    assert config["artifacts"]["burden_profile"] == str(shared / "resources/burden.yaml")
    assert config["output"]["directory"] == str(child / "results")


def test_profile_relative_paths_belong_to_the_profile_directory(tmp_path):
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text("output:\n  directory: original\n")
    folder = tmp_path / "machine"
    folder.mkdir()
    profile = folder / "profile.yaml"
    profile.write_text("output:\n  directory: local_results\n")
    assert load_config(pipeline, profile_path=profile)["output"]["directory"] == str(folder / "local_results")


def _trained(root, task="prevalent"):
    config = _write_training_bundle(root, task=task)
    train_integration(config)
    if task == "prevalent":
        calibrate(config)
    train_planner(config)
    return config


@pytest.mark.parametrize("operation", ["planner", "inference"])
def test_same_axes_and_split_do_not_allow_calibration_from_another_fusion(tmp_path, operation):
    config = _trained(tmp_path / "data")
    path = config["artifacts"]["fusion_model"]
    artifact = load_trace_artifact(path)
    artifact.model.estimators[0].probability = 0.9
    rewrite_training_artifact(artifact, path)
    with pytest.raises(ValueError, match="upstream.*integration|integration.*identity"):
        if operation == "planner":
            train_planner(config)
        else:
            build_trace_agent(config, batch=dataset_from_config(config).scores)


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
def test_planner_fusion_dependency_is_validated_at_inference(tmp_path, task):
    config = _trained(tmp_path / "data", task)
    if task == "prevalent":
        # Refresh the calibrator for the new fusion, leaving the old Q-model.
        path = config["artifacts"]["fusion_model"]
        artifact = load_trace_artifact(path)
        artifact.model.estimators[0].probability = 0.9
        rewrite_training_artifact(artifact, path)
        calibrate(config)
    else:
        path = config["artifacts"]["fusion_model"]
        artifact = load_trace_artifact(path)
        artifact.model.estimators[0].value = 1.0
        rewrite_training_artifact(artifact, path)
    with pytest.raises(ValueError, match="upstream.*integration|integration.*identity"):
        build_trace_agent(config, batch=dataset_from_config(config).scores)


def test_planner_dependency_rejects_a_replaced_calibration(tmp_path):
    config = _trained(tmp_path / "data")
    path = config["artifacts"]["calibrators"]
    artifact = load_trace_artifact(path)
    rewrite_training_artifact(replace(artifact, created_at="different calibration run"), path)
    with pytest.raises(ValueError, match="upstream.*calibration|calibration.*identity"):
        build_trace_agent(config, batch=dataset_from_config(config).scores)


@pytest.mark.parametrize("role,row", [("train", 0), ("validation", 5)])
@pytest.mark.parametrize("stage", [calibrate, train_planner])
def test_changed_supervised_scores_under_the_same_split_require_fusion_refit(tmp_path, role, row, stage):
    config = _trained(tmp_path / "data")
    path = Path(config["data"]["input_dir"]) / "cohort_oof_probs.npy"
    values = np.load(path)
    values[row, 0, 0] = 0.99
    np.save(path, values)
    with pytest.raises(ValueError, match="training input"):
        stage(config)


def test_changing_only_test_targets_does_not_invalidate_supervised_inputs(tmp_path):
    config = _trained(tmp_path / "data")
    path = Path(config["data"]["input_dir"]) / "cohort_y.npy"
    values = np.load(path)
    values[7:] = 1 - values[7:]
    np.save(path, values)
    assert calibrate(config).artifact_path.is_file()
    assert train_planner(config).artifact_path.is_file()


def test_small_baseline_hazard_is_not_rounded_to_zero_risk():
    cumulative = 1e-8
    mapper = CoxAbsoluteRiskMapper(endpoints=("D",), baseline_survival={"5y": np.exp([-cumulative])})
    result = mapper.transform(np.array([[20.0]]), ("D",))["5y"]
    expected = -np.expm1(-cumulative * np.exp(20.0))
    assert result[0, 0] == pytest.approx(expected, rel=1e-6)


def test_small_external_cumulative_hazard_survives_artifact_loading(tmp_path):
    np.save(tmp_path / "hazard.npy", [[1e-8]])
    np.save(tmp_path / "time.npy", [5.0])
    mapper = load_absolute_risk_mapper(
        {"baseline_cumhaz": str(tmp_path / "hazard.npy"), "time_grid": str(tmp_path / "time.npy")},
        endpoints=("D",),
    )
    assert mapper.transform(np.array([[20.0]]), ("D",))["5y"][0, 0] > 0.99


def test_cli_restores_schema_fallback_consensus_aliases_and_threshold_diagnostics(tmp_path):
    config = _trained(tmp_path / "data")
    bundle = tmp_path / "evidence"
    metadata = tmp_path / "metadata"
    bundle.mkdir()
    metadata.mkdir()
    tables = (
        (
            bundle / "consensus_evidence.csv",
            [
                {
                    "disease": "Long Disease",
                    "evidence_bucket": "risk_factor",
                    "factor": "bmi",
                    "strength_rank": 3,
                    "certainty_rank": 3,
                    "source": "original",
                }
            ],
        ),
        (
            bundle / "decision_screening_evidence.csv",
            [
                {
                    "disease": "Long Disease",
                    "evidence_bucket": "decision_screening",
                    "factor": "bmi",
                    "threshold": ">= 25",
                    "source": "original",
                }
            ],
        ),
        (metadata / "outcome_filter.csv", [{"LONGNAME": "Long Disease", "NAME": "D1"}]),
    )
    for path, rows in tables:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    np.save(tmp_path / "names.npy", ["p21001"])
    np.save(tmp_path / "contrib.npy", np.ones((9, 1, 1)))
    np.save(tmp_path / "values.npy", np.array([[24.0], [27.0]] + [[30.0]] * 7))
    (tmp_path / "schema.json").write_text(
        json.dumps({"feature_code_to_name": {"p21001": "Body mass index (BMI)"}})
    )
    config["planner"] = {"initial_modules": ["M1"]}
    config["consistency"] = {
        "enabled": True,
        "evidence_bundle": str(bundle),
        "disease_metadata_dir": str(metadata),
        "feature_names": str(tmp_path / "names.npy"),
        "feature_schema_path": str(tmp_path / "schema.json"),
        "feature_values_path": str(tmp_path / "values.npy"),
        "contributions": str(tmp_path / "contrib.npy"),
    }
    summary = run_inference(config)
    output = summary.manifest_path.parent
    with np.load(output / "consistency_scores.npz") as archive:
        np.testing.assert_allclose(archive["evidence"], 0.754914998867, atol=1e-6)
    with (output / "threshold_checks.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    assert [row["threshold_status"] for row in rows[:2]] == ["fail", "pass"]
    manifest = json.loads(summary.manifest_path.read_text())
    sources = manifest["artifact_provenance"]
    assert any(key.endswith("consensus_evidence.csv") for key in sources)
    assert any(key.endswith("outcome_filter.csv") for key in sources)
    assert sources["feature_schema_path"]["sha256"]


@pytest.mark.parametrize(
    "values",
    [
        np.arange(24).reshape(4, 2, 3),
        np.array(["\u00e9", "01", "a,b"]),
        np.array(["a", 5, None], dtype=object),
        np.zeros((4, 0)),
    ],
)
def test_row_fingerprint_matches_the_materialized_selection(values):
    from trace_agent.runtime.provenance import array_sha256, row_array_sha256

    rows = np.array([2, 0, 2])
    assert row_array_sha256(values, rows) == array_sha256(values[rows])
    assert row_array_sha256(values, np.array([], dtype=int)) == array_sha256(values[:0])


def test_unused_inherited_environment_path_is_not_expanded(tmp_path):
    (tmp_path / "base.yaml").write_text("output:\n  directory: ${TRACE_UNUSED_FRESH_TEST}/results\n")
    path = tmp_path / "child.yaml"
    path.write_text("extends: base.yaml\noutput:\n  directory: local\n")
    assert load_config(path)["output"]["directory"] == str(tmp_path / "local")
