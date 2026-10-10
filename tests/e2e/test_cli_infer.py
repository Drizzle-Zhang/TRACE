from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.isotonic import IsotonicRegression

from trace_agent.cli.main import main
from trace_agent.contracts.task import TaskType
from trace_agent.identity import TRACE_FULL_NAME
from trace_agent.runtime.model_artifact import TraceModelArtifact, dump_trace_artifact
from trace_agent.training.prevalent.calibration import PrevalentCalibrationModel


class MeanFusion:
    def fuse(self, scores, used, pred_mask_md=None):
        active = np.asarray(pred_mask_md, dtype=bool) & np.asarray(used)[:, None]
        return np.nanmean(np.where(active, scores, np.nan), axis=0)

    def fuse_logits(self, scores, used, pred_mask_md=None):
        probability = np.clip(self.fuse(scores, used, pred_mask_md), 1e-5, 1.0 - 1e-5)
        return np.log(probability / (1.0 - probability))


class MeanFusionWithPrior(MeanFusion):
    def fuse(self, scores, used, pred_mask_md=None):
        active = np.asarray(pred_mask_md, dtype=bool) & np.asarray(used)[:, None]
        return np.divide(
            np.where(active, scores, 0.0).sum(axis=0),
            active.sum(axis=0),
            out=np.full(np.asarray(scores).shape[-1], 0.42),
            where=active.any(axis=0),
        )


class AddQ:
    def predict(self, rows):
        matrix = np.asarray(rows)
        action = matrix[:, -4:]
        return np.where(np.any(action[:, :2] > 0, axis=1), 1.0, 0.0)


class MeanSurvivalFusion:
    def predict(
        self,
        scores,
        *,
        pred_mask_nmd,
        used_mask_nm,
        standardized=False,
    ):
        active = np.asarray(pred_mask_nmd, dtype=bool) & np.asarray(used_mask_nm, dtype=bool)[:, :, None]
        return np.divide(
            np.where(active, scores, 0.0).sum(axis=1),
            active.sum(axis=1),
            out=np.full((np.asarray(scores).shape[0], np.asarray(scores).shape[-1]), 0.42),
            where=active.any(axis=1),
        )

    def standardize(self, raw):
        return np.asarray(raw)


def _dump_model(
    path: Path,
    *,
    model: object,
    component: str,
    task: TaskType,
    endpoints: tuple[str, ...] = ("D1", "D2"),
) -> None:
    dump_trace_artifact(
        TraceModelArtifact(
            component=component,
            task=task,
            model=model,
            modules=("basic", "blood"),
            endpoints=endpoints,
        ),
        path,
    )


def _write_bundle(root: Path) -> None:
    np.save(root / "demo_eids.npy", np.array([501]))
    np.save(root / "demo_modules.npy", np.array(["basic", "blood"]))
    np.save(root / "demo_diseases.npy", np.array(["D1", "D2"]))
    np.save(
        root / "demo_oof_probs.npy",
        np.array([[[0.2, 0.3], [0.8, 0.9]]], dtype=np.float32),
    )
    np.save(root / "demo_pred_mask.npy", np.ones((1, 2, 2), dtype=bool))
    np.save(root / "demo_avail.npy", np.ones((1, 2), dtype=bool))


@pytest.mark.parametrize("custom_classification_path", [False, True])
def test_infer_command_runs_agent_and_writes_structured_output(
    tmp_path: Path,
    custom_classification_path: bool,
) -> None:
    _write_bundle(tmp_path)
    _dump_model(
        tmp_path / "fusion.joblib",
        model=MeanFusion(),
        component="integration",
        task=TaskType.PREVALENT,
    )
    _dump_model(
        tmp_path / "q.joblib",
        model=AddQ(),
        component="planner",
        task=TaskType.PREVALENT,
    )
    np.save(tmp_path / "kg.npy", np.ones((2, 2), dtype=np.float32))
    (tmp_path / "burden.yaml").write_text("basic: 1.0\nblood: 2.0\n", encoding="utf-8")
    config_path = tmp_path / "pipeline.yaml"
    classification_paths = [tmp_path / "output/consistency_predictions.npz"]
    if custom_classification_path:
        classification_paths.append(tmp_path / "other/custom.npz")
    stale_files = []
    for path in classification_paths:
        stale_files.extend(
            [path, path.with_name(f"{path.stem}_baseline.csv"), path.with_name(f"{path.stem}_filtered.csv")]
        )
    for path in stale_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale previous classifications", encoding="utf-8")
    notes = tmp_path / "output/consistency_notes.csv"
    notes.write_text("keep unrelated output", encoding="utf-8")
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "task": "prevalent",
                "data": {"input_dir": str(tmp_path), "prefix": "demo"},
                "artifacts": {
                    "fusion_model": str(tmp_path / "fusion.joblib"),
                    "q_model": str(tmp_path / "q.joblib"),
                    "knowledge_matrix": str(tmp_path / "kg.npy"),
                    "burden_profile": str(tmp_path / "burden.yaml"),
                },
                "planner": {
                    "initial_modules": ["basic"],
                    "candidate_actions": 2,
                    "max_steps": 0,
                },
                "output": {
                    "directory": str(tmp_path / "output"),
                    "top_endpoints": 2,
                },
                "consistency": {"predictions_path": str(classification_paths[-1])},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert main(["infer", "--config", str(config_path)]) == 0
    assert not any(path.exists() for path in stale_files)
    assert notes.read_text(encoding="utf-8") == "keep unrelated output"

    rows = [
        json.loads(line)
        for line in (tmp_path / "output" / "agent_results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["participant_id"] == 501
    assert rows[0]["selected_modules"] == ["basic", "blood"]
    assert rows[0]["trajectory"][-1]["action"] == "stop"
    manifest = json.loads((tmp_path / "output" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["framework"] == {
        "name": "TRACE",
        "full_name": TRACE_FULL_NAME,
    }
    with (tmp_path / "output" / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.reader(handle))
    assert predictions[0] == ["participant_id", "D1", "D2"]
    assert predictions[1][0] == "501"
    np.testing.assert_allclose([float(value) for value in predictions[1][1:]], [0.5, 0.6])
    assert manifest["outputs"]["predictions_csv"] == str(tmp_path / "output" / "predictions.csv")


def test_prevalent_csv_preserves_axes_and_uses_final_calibrated_probabilities(
    tmp_path: Path,
) -> None:
    participant_ids = np.array(['participant, "B"', "00042", "A7"])
    endpoints = ("Disease, alpha", 'Disease "beta"')
    scores = np.array(
        [
            [[0.1, 0.2], [0.5, 0.8]],
            [[np.nan, 0.3], [np.nan, 0.7]],
            [[0.2, np.nan], [0.6, np.nan]],
        ],
        dtype=np.float32,
    )
    for suffix, values in (
        ("eids", participant_ids),
        ("modules", np.array(["basic", "blood"])),
        ("diseases", np.array(endpoints)),
        ("oof_probs", scores),
        ("pred_mask", np.isfinite(scores)),
        ("avail", np.ones((3, 2), dtype=bool)),
    ):
        np.save(tmp_path / f"cohort_{suffix}.npy", values)
    _dump_model(
        tmp_path / "fusion.joblib",
        model=MeanFusionWithPrior(),
        component="integration",
        task=TaskType.PREVALENT,
        endpoints=endpoints,
    )
    _dump_model(
        tmp_path / "q.joblib",
        model=AddQ(),
        component="planner",
        task=TaskType.PREVALENT,
        endpoints=endpoints,
    )
    _dump_model(
        tmp_path / "calibration.joblib",
        model=PrevalentCalibrationModel(
            endpoints=endpoints,
            calibrators={
                endpoints[0]: IsotonicRegression(out_of_bounds="clip").fit([0, 1], [0.2, 0.8]),
                endpoints[1]: IsotonicRegression(out_of_bounds="clip").fit([0, 1], [0.1, 0.7]),
            },
        ),
        component="calibration",
        task=TaskType.PREVALENT,
        endpoints=endpoints,
    )
    np.save(tmp_path / "kg.npy", np.ones((2, 2), dtype=np.float32))
    (tmp_path / "burden.yaml").write_text("basic: 1.0\nblood: 2.0\n", encoding="utf-8")
    config_path = tmp_path / "pipeline.yaml"
    output = tmp_path / "output"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "task": "prevalent",
                "data": {"input_dir": str(tmp_path), "prefix": "cohort"},
                "artifacts": {
                    "fusion_model": str(tmp_path / "fusion.joblib"),
                    "q_model": str(tmp_path / "q.joblib"),
                    "calibrators": str(tmp_path / "calibration.joblib"),
                    "knowledge_matrix": str(tmp_path / "kg.npy"),
                    "burden_profile": str(tmp_path / "burden.yaml"),
                },
                "planner": {"initial_modules": ["basic"], "candidate_actions": 2, "max_steps": 0},
                "output": {"directory": str(output), "top_endpoints": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert main(["infer", "--config", str(config_path)]) == 0

    records = [json.loads(line) for line in (output / "agent_results.jsonl").read_text().splitlines()]
    assert [row["participant_id"] for row in records] == participant_ids.tolist()
    assert all(len(row["endpoints"]) == 1 for row in records)
    assert all(row["selected_modules"] == ["basic", "blood"] for row in records)
    # An invalid endpoint still has a finite model prior; the validity mask must
    # control export rather than the value's finiteness alone.
    assert records[1]["trajectory"][-1]["risk_profile"]["valid_mask"] == [False, True]
    with (output / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.reader(handle))
    assert predictions[0] == ["participant_id", *endpoints]
    assert [row[0] for row in predictions[1:]] == participant_ids.tolist()
    assert all(len(row) == 3 for row in predictions)
    np.testing.assert_allclose([float(value) for value in predictions[1][1:]], [0.38, 0.4])
    assert predictions[2][1] == ""
    np.testing.assert_allclose(float(predictions[2][2]), 0.4)
    np.testing.assert_allclose(float(predictions[3][1]), 0.44)
    assert predictions[3][2] == ""
    np.testing.assert_allclose(
        np.load(output / "final_scores.npy"),
        [[0.38, 0.4], [np.nan, 0.4], [0.44, np.nan]],
        equal_nan=True,
    )
    assert np.load(output / "participant_ids.npy").tolist() == participant_ids.tolist()
    assert np.load(output / "prediction_endpoints.npy").tolist() == list(endpoints)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["outputs"]["predictions_csv"] == str(output / "predictions.csv")


def test_longitudinal_infer_includes_absolute_horizon_risk(
    tmp_path: Path,
) -> None:
    np.save(tmp_path / "incident_eids.npy", np.array(["701", "00042"]))
    np.save(tmp_path / "incident_modules.npy", np.array(["basic", "blood"]))
    np.save(tmp_path / "incident_diseases.npy", np.array(["D1", "D2"]))
    np.save(
        tmp_path / "incident_scores.npy",
        np.array(
            [[[0.1, 0.2], [0.5, 0.8]], [[0.2, np.nan], [0.6, np.nan]]],
            dtype=np.float32,
        ),
    )
    np.save(
        tmp_path / "incident_pred_mask.npy",
        np.array([[[True, True], [True, True]], [[True, False], [True, False]]]),
    )
    np.save(
        tmp_path / "incident_avail.npy",
        np.ones((2, 2), dtype=bool),
    )
    _dump_model(
        tmp_path / "survival_fusion.joblib",
        model=MeanSurvivalFusion(),
        component="integration",
        task=TaskType.LONGITUDINAL,
    )
    _dump_model(
        tmp_path / "survival_q.joblib",
        model=AddQ(),
        component="planner",
        task=TaskType.LONGITUDINAL,
    )
    np.save(
        tmp_path / "survival_kg.npy",
        np.ones((2, 2), dtype=np.float32),
    )
    (tmp_path / "survival_burden.yaml").write_text("basic: 1.0\nblood: 2.0\n", encoding="utf-8")
    np.savez(
        tmp_path / "absolute_risk.npz",
        endpoints=np.array(["D1", "D2"]),
        baseline_survival_5y=np.array([0.9, 0.8], dtype=np.float32),
        baseline_survival_10y=np.array([0.8, 0.6], dtype=np.float32),
    )
    config_path = tmp_path / "longitudinal.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "task": "longitudinal",
                "data": {
                    "input_dir": str(tmp_path),
                    "prefix": "incident",
                },
                "artifacts": {
                    "fusion_model": str(tmp_path / "survival_fusion.joblib"),
                    "q_model": str(tmp_path / "survival_q.joblib"),
                    "knowledge_matrix": str(tmp_path / "survival_kg.npy"),
                    "burden_profile": str(tmp_path / "survival_burden.yaml"),
                    "absolute_risk": str(tmp_path / "absolute_risk.npz"),
                },
                "planner": {
                    "initial_modules": ["basic"],
                    "candidate_actions": 2,
                    "max_steps": 0,
                },
                "output": {
                    "directory": str(tmp_path / "longitudinal_output"),
                    "top_endpoints": 1,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert main(["infer", "--config", str(config_path)]) == 0

    row = json.loads(
        (tmp_path / "longitudinal_output" / "agent_results.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert row["task"] == "longitudinal"
    assert "5y" in row["endpoints"][0]["horizon_risks"]
    assert len(row["endpoints"]) == 1
    with np.load(tmp_path / "longitudinal_output/horizon_risks.npz", allow_pickle=False) as archive:
        assert archive["endpoints"].tolist() == ["D1", "D2"]
        assert archive["5y"].shape == (2, 2)
        np.testing.assert_allclose(
            archive["5y"][0], 1 - np.array([0.9, 0.8]) ** np.exp([0.3, 0.5]), atol=1e-7
        )
        assert archive["participant_ids"].tolist() == ["701", "00042"]
        assert not archive["valid_mask"][1, 1]
        assert np.isfinite(archive["5y"][1, 1])
        assert np.isfinite(archive["10y"][1, 1])
    manifest = json.loads((tmp_path / "longitudinal_output/run_manifest.json").read_text())
    assert manifest["outputs"]["horizon_risks"].endswith("horizon_risks.npz")
    output = tmp_path / "longitudinal_output"
    for horizon, survival in (("5y", [0.9, 0.8]), ("10y", [0.8, 0.6])):
        path = output / f"predictions_{horizon}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            predictions = list(csv.reader(handle))
        assert predictions[0] == ["participant_id", "D1", "D2"]
        assert predictions[1][0] == "701"
        assert predictions[2][0] == "00042"
        assert predictions[2][2] == ""
        assert len(predictions) == 3
        np.testing.assert_allclose(
            [float(value) for value in predictions[1][1:]],
            1 - np.array(survival) ** np.exp([0.3, 0.5]),
            atol=1e-7,
        )
        np.testing.assert_allclose(float(predictions[2][1]), 1 - survival[0] ** np.exp(0.4), atol=1e-7)
        assert manifest["outputs"]["horizon_predictions_csv"][horizon] == str(path)
    assert set(manifest["outputs"]["horizon_predictions_csv"]) == {"5y", "10y"}
    assert "predictions_csv" not in manifest["outputs"]
    assert not (output / "predictions.csv").exists()

    notes = output / "predictions_notes.csv"
    notes.write_text("notes\nkeep this file\n", encoding="utf-8")
    np.savez(
        tmp_path / "absolute_risk.npz",
        endpoints=np.array(["D1", "D2"]),
        baseline_survival_5y=np.array([0.9, 0.8], dtype=np.float32),
    )
    assert main(["infer", "--config", str(config_path)]) == 0
    assert (output / "predictions_5y.csv").is_file()
    assert not (output / "predictions_10y.csv").exists()
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["outputs"]["horizon_predictions_csv"] == {"5y": str(output / "predictions_5y.csv")}

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    del config["artifacts"]["absolute_risk"]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["infer", "--config", str(config_path)]) == 0
    assert not (output / "predictions_5y.csv").exists()
    assert not (output / "predictions_10y.csv").exists()
    assert not (output / "predictions.csv").exists()
    assert not (output / "horizon_risks.npz").exists()
    assert notes.read_text(encoding="utf-8") == "notes\nkeep this file\n"
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert not manifest["outputs"].get("horizon_predictions_csv")
    assert "predictions_csv" not in manifest["outputs"]


def test_consistency_command_writes_three_evidence_chains(
    tmp_path: Path,
) -> None:
    _write_bundle(tmp_path)
    _dump_model(
        tmp_path / "fusion.joblib",
        model=MeanFusion(),
        component="integration",
        task=TaskType.PREVALENT,
    )
    _dump_model(
        tmp_path / "q.joblib",
        model=AddQ(),
        component="planner",
        task=TaskType.PREVALENT,
    )
    np.save(tmp_path / "kg.npy", np.ones((2, 2), dtype=np.float32))
    np.save(
        tmp_path / "contributions.npy",
        np.array([[[1.0, 0.5], [0.8, 0.2]]], dtype=np.float32),
    )
    np.save(
        tmp_path / "feature_weights.npy",
        np.ones((2, 2), dtype=np.float32),
    )
    np.save(
        tmp_path / "evidence_weights.npy",
        np.ones((2, 2), dtype=np.float32),
    )
    (tmp_path / "burden.yaml").write_text("basic: 1.0\nblood: 2.0\n", encoding="utf-8")
    config_path = tmp_path / "consistency.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "task": "prevalent",
                "data": {"input_dir": str(tmp_path), "prefix": "demo"},
                "artifacts": {
                    "fusion_model": str(tmp_path / "fusion.joblib"),
                    "q_model": str(tmp_path / "q.joblib"),
                    "knowledge_matrix": str(tmp_path / "kg.npy"),
                    "burden_profile": str(tmp_path / "burden.yaml"),
                },
                "planner": {
                    "initial_modules": ["basic"],
                    "candidate_actions": 2,
                },
                "consistency": {
                    "enabled": True,
                    "strategy": "combined",
                    "contributions": str(tmp_path / "contributions.npy"),
                    "feature_weights": str(tmp_path / "feature_weights.npy"),
                    "evidence_weights": str(tmp_path / "evidence_weights.npy"),
                    "thresholds": {
                        "prediction": 0.5,
                        "kg": 0.5,
                        "feature": 0.5,
                        "evidence": 0.5,
                        "combined": 0.5,
                    },
                },
                "output": {"directory": str(tmp_path / "consistency_output")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert main(["consistency", "--config", str(config_path)]) == 0

    row = json.loads(
        (tmp_path / "consistency_output" / "agent_results.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert row["consistency"] is not None
    assert len(row["consistency"]["kg_scores"]) == 2
    assert len(row["consistency"]["feature_scores"]) == 2
    assert len(row["consistency"]["evidence_scores"]) == 2
