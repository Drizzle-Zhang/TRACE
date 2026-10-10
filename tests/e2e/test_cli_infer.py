from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from trace_agent.cli.main import main
from trace_agent.contracts.task import TaskType
from trace_agent.identity import TRACE_FULL_NAME
from trace_agent.runtime.model_artifact import TraceModelArtifact, dump_trace_artifact


class MeanFusion:
    def fuse(self, scores, used, pred_mask_md=None):
        active = np.asarray(pred_mask_md, dtype=bool) & np.asarray(used)[:, None]
        return np.nanmean(np.where(active, scores, np.nan), axis=0)

    def fuse_logits(self, scores, used, pred_mask_md=None):
        probability = np.clip(self.fuse(scores, used, pred_mask_md), 1e-5, 1.0 - 1e-5)
        return np.log(probability / (1.0 - probability))


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
        return np.nanmean(np.where(active, scores, np.nan), axis=1)

    def standardize(self, raw):
        return np.asarray(raw)


def _dump_model(
    path: Path,
    *,
    model: object,
    component: str,
    task: TaskType,
) -> None:
    dump_trace_artifact(
        TraceModelArtifact(
            component=component,
            task=task,
            model=model,
            modules=("basic", "blood"),
            endpoints=("D1", "D2"),
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


def test_infer_command_runs_agent_and_writes_structured_output(
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
    (tmp_path / "burden.yaml").write_text("basic: 1.0\nblood: 2.0\n", encoding="utf-8")
    config_path = tmp_path / "pipeline.yaml"
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
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert main(["infer", "--config", str(config_path)]) == 0

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


def test_longitudinal_infer_includes_absolute_horizon_risk(
    tmp_path: Path,
) -> None:
    np.save(tmp_path / "incident_eids.npy", np.array([701]))
    np.save(tmp_path / "incident_modules.npy", np.array(["basic", "blood"]))
    np.save(tmp_path / "incident_diseases.npy", np.array(["D1", "D2"]))
    np.save(
        tmp_path / "incident_scores.npy",
        np.array([[[0.1, 0.2], [0.5, 0.8]]], dtype=np.float32),
    )
    np.save(
        tmp_path / "incident_pred_mask.npy",
        np.ones((1, 2, 2), dtype=bool),
    )
    np.save(
        tmp_path / "incident_avail.npy",
        np.ones((1, 2), dtype=bool),
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
        assert archive["5y"].shape == (1, 2)
        np.testing.assert_allclose(
            archive["5y"][0], 1 - np.array([0.9, 0.8]) ** np.exp([0.3, 0.5]), atol=1e-7
        )
    manifest = json.loads((tmp_path / "longitudinal_output/run_manifest.json").read_text())
    assert manifest["outputs"]["horizon_risks"].endswith("horizon_risks.npz")


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
