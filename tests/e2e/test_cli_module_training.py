import json

import numpy as np
import pytest
import yaml

from tests.module_fixtures import module_config, write_full_module_pipeline, write_module_inputs
from trace_agent.cli.main import main
from trace_agent.data.arrays import load_module_score_bundle
from trace_agent.runtime.model_artifact import load_trace_artifact


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
def test_cli_native_prepare_train_score_pipeline(tmp_path, monkeypatch, task):
    if task == "longitudinal":
        pytest.importorskip("xgboost")
    import subprocess

    def forbidden(*args, **kwargs):
        pytest.fail("native CLI must not execute external scripts")

    write_module_inputs(tmp_path, task=task)
    config = module_config(tmp_path, task=task)
    config["module_data"]["score_mode"] = "training"
    config["pipeline"] = {"steps": ["prepare-data", "train-modules", "score-modules"]}
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(subprocess, "run", forbidden)
    assert main(["run", "--config", str(path)]) == 0
    batch = load_module_score_bundle(tmp_path / "prepared", prefix="cohort", task=task)
    assert batch.scores.shape == (180, 2, 1)
    assert batch.pred_mask.all()


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
def test_cli_from_raw_features_through_agent_inference(tmp_path, task):
    if task == "longitudinal":
        pytest.importorskip("xgboost")
    path = write_full_module_pipeline(tmp_path, task=task)
    assert main(["run", "--config", str(path)]) == 0
    bank = json.loads((tmp_path / "bank/module_bank_manifest.json").read_text(encoding="utf-8"))
    assert all(pair["status"] == "trained" for pair in bank["pairs"])
    fusion = load_trace_artifact(tmp_path / "output/fusion_model.joblib", component="integration", task=task)
    assert fusion.training["test_rows_used"] == 0
    if task == "prevalent":
        from lightgbm import LGBMClassifier

        assert isinstance(fusion.model.estimators[0], LGBMClassifier)
    else:
        from xgboost import XGBRegressor

        assert isinstance(fusion.model.estimators[0], XGBRegressor)
    load_trace_artifact(tmp_path / "output/q_model.joblib", component="planner", task=task)
    if task == "prevalent":
        calibration = load_trace_artifact(
            tmp_path / "output/calibration_model.joblib", component="calibration", task=task
        )
        assert calibration.training["test_rows_used"] == 0
    final_scores = np.load(tmp_path / "output/final_scores.npy", allow_pickle=False)
    assert final_scores.shape == (180, 1)
    assert np.isfinite(final_scores).all()
    assert (tmp_path / "output/agent_results.jsonl").is_file()
