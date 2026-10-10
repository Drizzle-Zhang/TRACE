from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trace_agent.cli import runner
from trace_agent.cli.main import build_parser, main
from trace_agent.cli.operations import evaluate
from trace_agent.identity import TRACE_FULL_NAME
from trace_agent.training.service import TrainingSummary


def test_cli_exposes_stage_zero_to_eight_commands() -> None:
    parser = build_parser()
    help_text = parser.format_help()

    for command in (
        "prepare-data",
        "train-integration",
        "calibrate",
        "train-planner",
        "infer",
        "consistency",
        "evaluate",
        "run",
    ):
        assert command in help_text


def test_cli_without_command_prints_help(capsys) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    normalized_output = " ".join(output.split())
    assert "TRACE" in output
    assert TRACE_FULL_NAME in normalized_output
    assert build_parser().prog == "trace"


def test_cli_accepts_an_explicit_log_level() -> None:
    args = build_parser().parse_args(["infer", "--dry-run", "--log-level", "DEBUG"])
    assert args.log_level == "DEBUG"


def test_module_commands_support_dry_run(capsys, monkeypatch):
    from trace_agent.training.modules import service

    def forbidden(*args, **kwargs):
        pytest.fail("dry-run must not open module inputs or fit models")

    monkeypatch.setattr(service, "module_context", forbidden)
    for command in ("train-modules", "score-modules"):
        assert main([command, "--dry-run"]) == 0
        assert f"command={command}" in capsys.readouterr().out


def test_module_commands_dispatch_without_external_scripts(monkeypatch, tmp_path):
    from trace_agent.training.modules import service

    calls = []

    def train(config):
        calls.append("train")
        return tmp_path / "module_bank_manifest.json"

    def score(config):
        calls.append("score")
        return tmp_path / "cohort_score_bundle_manifest.json"

    monkeypatch.setattr(service, "train_modules", train)
    monkeypatch.setattr(service, "score_modules", score)
    assert runner._execute("train-modules", {}) is not None
    assert runner._execute("score-modules", {}) is not None
    assert calls == ["train", "score"]


def test_unknown_prepare_input_mode_errors():
    from trace_agent.cli.operations import prepare_data

    with pytest.raises(ValueError, match="input_mode"):
        prepare_data({"module_data": {"input_mode": "guess"}})


def test_training_commands_dispatch_native_services(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def summary(name: str) -> TrainingSummary:
        calls.append(name)
        return TrainingSummary(
            artifact_path=tmp_path / f"{name}.joblib",
            manifest_path=tmp_path / f"{name}.json",
            train_rows=1,
            validation_rows=1,
        )

    monkeypatch.setattr(runner, "train_integration", lambda config: summary("integration"))
    monkeypatch.setattr(runner, "calibrate", lambda config: summary("calibration"))
    monkeypatch.setattr(runner, "train_planner", lambda config: summary("planner"))

    for command in ("train-integration", "calibrate", "train-planner"):
        assert runner._execute(command, {}) is not None

    assert calls == ["integration", "calibration", "planner"]


def _evaluation_config(tmp_path: Path) -> dict[str, object]:
    prefix = "cohort"
    eids = np.array([10, 11, 12, 13, 14, 15])
    labels = np.array([[0], [1], [0], [1], [0], [1]], dtype=np.float32)
    scores = np.broadcast_to(labels[:, None, :], (6, 1, 1)).copy()
    np.save(tmp_path / f"{prefix}_eids.npy", eids)
    np.save(tmp_path / f"{prefix}_modules.npy", np.array(["M1"]))
    np.save(tmp_path / f"{prefix}_diseases.npy", np.array(["D1"]))
    np.save(tmp_path / f"{prefix}_oof_probs.npy", scores)
    np.save(tmp_path / f"{prefix}_pred_mask.npy", np.ones_like(scores, dtype=bool))
    np.save(tmp_path / f"{prefix}_avail.npy", np.ones((6, 1), dtype=bool))
    np.save(tmp_path / f"{prefix}_y.npy", labels)
    np.save(tmp_path / f"{prefix}_y_mask.npy", np.ones_like(labels, dtype=bool))
    (tmp_path / "splits.csv").write_text(
        "row_index,eid,split\n"
        "0,10,train\n1,11,train\n2,12,validation\n3,13,validation\n"
        "4,14,test\n5,15,test\n",
        encoding="utf-8",
    )
    predictions = np.array([[0.2], [0.8], [0.3], [0.7], [0.1], [0.9]], dtype=np.float32)
    np.save(tmp_path / "predictions.npy", predictions)
    np.save(tmp_path / "prediction_eids.npy", eids)
    np.save(tmp_path / "prediction_endpoints.npy", np.array(["D1"]))
    np.save(tmp_path / "prediction_valid.npy", np.ones_like(labels, dtype=bool))
    return {
        "task": "prevalent",
        "data": {
            "input_dir": str(tmp_path),
            "prefix": prefix,
            "split_table": {"path": str(tmp_path / "splits.csv")},
        },
        "evaluation": {
            "predictions": str(tmp_path / "predictions.npy"),
            "participant_ids": str(tmp_path / "prediction_eids.npy"),
            "endpoints_file": str(tmp_path / "prediction_endpoints.npy"),
            "valid_mask_path": str(tmp_path / "prediction_valid.npy"),
        },
        "output": {"directory": str(tmp_path / "output")},
    }


def test_evaluation_uses_only_test_rows_and_records_identities(tmp_path: Path) -> None:
    report_path = evaluate(_evaluation_config(tmp_path))

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["split"] == "test"
    assert report["participants"] == 2
    assert report["n_valid"] == 2
    assert report["row_indices_sha256"]
    assert report["eids_sha256"]


def test_evaluation_rejects_prediction_eid_reordering(tmp_path: Path) -> None:
    config = _evaluation_config(tmp_path)
    np.save(tmp_path / "prediction_eids.npy", np.array([11, 10, 12, 13, 14, 15]))

    with pytest.raises(ValueError, match="EIDs.*order"):
        evaluate(config)
