"""Native output boundaries needed by the frozen-source validation workflow."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.isotonic import IsotonicRegression

from tests.module_fixtures import write_full_module_pipeline
from tests.training_fixtures import rewrite_training_artifact
from trace_agent.cli.main import main
from trace_agent.cli.operations import evaluate, run_inference
from trace_agent.runtime.artifacts import _load_calibrators
from trace_agent.runtime.config import load_config
from trace_agent.runtime.model_artifact import TraceModelArtifact, dump_trace_artifact, load_trace_artifact
from trace_agent.training.prevalent.calibration import PrevalentCalibrationModel


@pytest.mark.parametrize(
    "modules,endpoints",
    [(("OLD_MODULE",), ("D1",)), (("M1", "M2"), ("D1",))],
)
def test_native_calibrator_must_match_both_complete_axes(tmp_path, modules, endpoints):
    model = PrevalentCalibrationModel(
        endpoints=endpoints,
        calibrators={"D1": IsotonicRegression().fit([0.0, 1.0], [0.1, 0.2])},
    )
    path = tmp_path / "calibration.joblib"
    dump_trace_artifact(
        TraceModelArtifact(
            component="calibration", task="prevalent", model=model, modules=modules, endpoints=endpoints
        ),
        path,
    )
    with pytest.raises(ValueError, match="calibration artifact.*order"):
        _load_calibrators(path, modules=("M1", "M2"), endpoints=("D1", "D2"))


def test_empty_planner_output_is_not_evaluated_as_a_finite_prior(tmp_path):
    path = write_full_module_pipeline(tmp_path, task="prevalent")
    assert main(["run", "--config", str(path)]) == 0
    config = load_config(path)
    artifact = load_trace_artifact(config["artifacts"]["q_model"])
    model = artifact.model
    model.estimator = DummyRegressor(strategy="constant", constant=0).fit(
        np.zeros((1, model.feature_width)), np.zeros(1)
    )
    rewrite_training_artifact(artifact, config["artifacts"]["q_model"])
    run_inference(config)
    output = tmp_path / "output"
    records = [json.loads(line) for line in (output / "agent_results.jsonl").read_text().splitlines()]
    assert all(not row["endpoints"] for row in records)
    assert np.isnan(np.load(output / "final_scores.npy")).all()
    assert not np.load(output / "final_valid_mask.npy").any()
    assert np.load(output / "prediction_endpoints.npy").tolist() == ["D1"]
    report = json.loads(evaluate(config).read_text())
    assert report["n_valid"] == 0


def test_inference_persists_named_consistency_components(tmp_path):
    path = write_full_module_pipeline(tmp_path, task="prevalent")
    assert main(["run", "--config", str(path)]) == 0
    config = load_config(path)
    config["planner"]["initial_modules"] = ["M1"]
    config["consistency"] = {"enabled": True}
    run_inference(config)
    with np.load(tmp_path / "output/consistency_scores.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"participant_ids", "endpoints", "kg", "feature", "evidence"}
        assert archive["endpoints"].tolist() == ["D1"]
        for name in ("kg", "feature", "evidence"):
            assert archive[name].shape == (180, 1)


def test_inference_routes_evidence_bundle_through_individual_rules(tmp_path):
    reference = json.loads(
        (Path(__file__).parents[1] / "fixtures/migration_evidence_reference.json").read_text()
    )
    case = next(item for item in reference["cases"] if item["name"] == "repeated_support_rules")
    path = write_full_module_pipeline(tmp_path, task="prevalent")
    assert main(["run", "--config", str(path)]) == 0
    config = load_config(path)
    bundle = tmp_path / "evidence"
    bundle.mkdir()
    for filename, rows in (
        ("risk_factor_evidence.csv", case["rules"]),
        ("disease_evidence_feature_map.csv", case["mappings"]),
    ):
        with (bundle / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    np.save(tmp_path / "names.npy", np.array(case["features"]))
    np.save(tmp_path / "contrib.npy", np.repeat(np.asarray(case["contributions"]), 180, axis=0))
    config["consistency"] = {
        "enabled": True,
        "evidence_bundle": str(bundle),
        "feature_names": str(tmp_path / "names.npy"),
        "contributions": str(tmp_path / "contrib.npy"),
    }
    run_inference(config)
    with np.load(tmp_path / "output/consistency_scores.npz", allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["evidence"], np.repeat(case["expected_scores"], 180, axis=0))
