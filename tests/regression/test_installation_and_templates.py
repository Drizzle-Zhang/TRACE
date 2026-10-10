"""Exercise shipped templates and a fresh process without optional XGBoost."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.module_fixtures import write_full_module_pipeline
from trace_agent.cli.main import main
from trace_agent.runtime.config import load_config
from trace_agent.training.modules.service import score_modules

PROJECT = Path(__file__).resolve().parents[2]
BLOCK_XGBOOST = """
import importlib.abc
import sys

class NoXGBoost(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'xgboost' or fullname.startswith('xgboost.'):
            raise ModuleNotFoundError("No module named 'xgboost'", name='xgboost')

sys.meta_path.insert(0, NoXGBoost())
"""


def _without_xgboost(code, *args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT / "src"), str(PROJECT)])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-c", BLOCK_XGBOOST + code, *map(str, args)],
        cwd=PROJECT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_all_cli_dry_runs_work_without_xgboost():
    result = _without_xgboost(
        """
from trace_agent.cli.main import COMMANDS, main
for name, _ in COMMANDS:
    assert main([name, '--dry-run']) == 0
assert 'xgboost' not in sys.modules
"""
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_prevalent_training_and_inference_work_without_xgboost(tmp_path):
    config_path = write_full_module_pipeline(tmp_path, task="prevalent")
    result = _without_xgboost(
        """
from trace_agent.cli.main import main
assert main(['run', '--config', sys.argv[1]]) == 0
assert 'xgboost' not in sys.modules
""",
        config_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "output/calibration_model.joblib").is_file()
    assert np.load(tmp_path / "output/final_scores.npy").shape == (180, 1)


def test_actual_cox_fit_without_xgboost_reports_installation_hint():
    result = _without_xgboost(
        """
import numpy as np
try:
    from trace_agent.training.longitudinal.integration import _fit_endpoint, LongitudinalIntegrationConfig
    _fit_endpoint(np.ones((4, 2)), np.array([1., -2., 3., -4.]), np.array([1., 0., 1., 0.]),
                  config=LongitudinalIntegrationConfig(min_events=1, n_estimators=1), endpoint=0)
except ImportError as exc:
    assert 'trace-agent[longitudinal]' in str(exc), str(exc)
    assert 'pip install' in str(exc), str(exc)
else:
    raise AssertionError('Cox fitting must not silently replace the requested estimator')
"""
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _model_environment(root, monkeypatch):
    for name, value in {
        "TRACE_INPUT_DIR": root / "prepared",
        "TRACE_INPUT_PREFIX": "cohort",
        "TRACE_FUSION_MODEL": root / "output/fusion_model.joblib",
        "TRACE_CALIBRATORS": root / "output/calibration_model.joblib",
        "TRACE_Q_MODEL": root / "output/q_model.joblib",
        "TRACE_KNOWLEDGE_MATRIX": root / "kg.npy",
        "TRACE_OUTPUT_DIR": root / "validation_output",
        "TRACE_SPLIT_TABLE": root / "splits.csv",
    }.items():
        monkeypatch.setenv(name, str(value))
    profile = root / "profile.yaml"
    profile.write_text(yaml.safe_dump({"artifacts": {"burden_profile": str(root / "burden.yaml")}}))
    return profile


@pytest.mark.parametrize("score_mode", ["training", "inference"])
def test_full_validation_template_runs_with_shared_split_and_no_legacy_label_variables(
    tmp_path,
    monkeypatch,
    score_mode,
):
    train_path = write_full_module_pipeline(tmp_path, task="prevalent")
    assert main(["run", "--config", str(train_path)]) == 0
    train_config = load_config(train_path)
    train_config["module_data"]["score_mode"] = score_mode
    score_modules(train_config)
    profile = _model_environment(tmp_path, monkeypatch)
    np.save(tmp_path / "contributions.npy", np.ones((180, 1, 1), dtype=np.float32))
    np.save(tmp_path / "weights.npy", np.ones((1, 1), dtype=np.float32))
    np.save(tmp_path / "features.npy", np.array(["a"]))
    monkeypatch.setenv("TRACE_CONTRIBUTIONS", str(tmp_path / "contributions.npy"))
    monkeypatch.setenv("TRACE_FEATURE_WEIGHTS", str(tmp_path / "weights.npy"))
    monkeypatch.setenv("TRACE_FEATURE_NAMES", str(tmp_path / "features.npy"))
    monkeypatch.delenv("TRACE_LABELS", raising=False)
    monkeypatch.delenv("TRACE_LABEL_MASK", raising=False)
    template = PROJECT / "pipelines/full_validation.yaml"
    config = load_config(template, profile_path=profile)
    assert config["data"]["split_table"]["path"] == str(tmp_path / "splits.csv")
    assert config["artifacts"]["calibrators"] == train_config["artifacts"]["calibrators"]
    assert "labels" not in config.get("evaluation", {})
    assert main(["run", "--config", str(template), "--profile", str(profile)]) == 0
    report = json.loads((tmp_path / "validation_output/evaluation.json").read_text())
    assert report["split"] == "test"
    assert report["participants"] == 30
    assert report["consistency"]["split"] == "test"
    assert (tmp_path / "validation_output/consistency_rules.json").is_file()
    assert np.load(tmp_path / "validation_output/consistency_combined.npy").shape == (180, 1)


def test_longitudinal_full_validation_freezes_scale_rules_and_evaluates_test(tmp_path, monkeypatch):
    train_path = write_full_module_pipeline(tmp_path, task="longitudinal")
    assert main(["run", "--config", str(train_path)]) == 0
    profile = _model_environment(tmp_path, monkeypatch)
    np.save(tmp_path / "contributions.npy", np.ones((180, 1, 1), dtype=np.float32))
    np.save(tmp_path / "weights.npy", np.ones((1, 1), dtype=np.float32))
    np.save(tmp_path / "features.npy", np.array(["a"]))
    for key, name in (
        ("TRACE_CONTRIBUTIONS", "contributions"),
        ("TRACE_FEATURE_WEIGHTS", "weights"),
        ("TRACE_FEATURE_NAMES", "features"),
    ):
        monkeypatch.setenv(key, str(tmp_path / f"{name}.npy"))
    template = PROJECT / "pipelines/longitudinal/full_validation.yaml"
    assert main(["run", "--config", str(template), "--profile", str(profile)]) == 0
    output = tmp_path / "validation_output"
    report = json.loads((output / "evaluation.json").read_text())
    assert report["consistency"]["split"] == "test"
    assert report["consistency"]["horizon"] == "10y"
    rules = json.loads((output / "consistency_rules.json").read_text())
    assert rules["source"]["validation_participants"] == 30
    assert rules["source"]["test_rows_used"] == 0
    assert "consistency_scale" in rules["model_identity"]


def test_native_longitudinal_inference_template_uses_embedded_baseline(tmp_path, monkeypatch):
    train_path = write_full_module_pipeline(tmp_path, task="longitudinal")
    assert main(["run", "--config", str(train_path)]) == 0
    profile = _model_environment(tmp_path, monkeypatch)
    monkeypatch.delenv("TRACE_ABSOLUTE_RISK", raising=False)
    template = PROJECT / "pipelines/longitudinal/infer.yaml"
    assert main(["infer", "--config", str(template), "--profile", str(profile)]) == 0
    with np.load(tmp_path / "validation_output/horizon_risks.npz", allow_pickle=False) as archive:
        assert archive["5y"].shape == (180, 1)
        assert archive["10y"].shape == (180, 1)


def test_external_absolute_risk_is_an_explicit_profile_override(tmp_path, monkeypatch):
    _model_environment(tmp_path, monkeypatch)
    monkeypatch.setenv("TRACE_ABSOLUTE_RISK", str(tmp_path / "external.npz"))
    template = PROJECT / "pipelines/longitudinal/infer.yaml"
    default_config = load_config(template)
    assert default_config["artifacts"].get("absolute_risk") is None
    config = load_config(
        template,
        profile_path=PROJECT / "configs/profiles/absolute_risk.example.yaml",
    )
    assert config["artifacts"]["absolute_risk"] == str(tmp_path / "external.npz")
