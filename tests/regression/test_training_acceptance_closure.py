"""Native training metadata, preflight and publication acceptance cases."""

import dataclasses
import json
from pathlib import Path

import pytest

from tests.unit.test_training_service import _write_training_bundle
from trace_agent import __version__
from trace_agent.runtime.model_artifact import dump_trace_artifact, load_trace_artifact
from trace_agent.runtime.provenance import file_provenance
from trace_agent.training import service
from trace_agent.training.longitudinal.integration import LongitudinalIntegrationConfig
from trace_agent.training.planner import PlannerTrainingConfig
from trace_agent.training.prevalent.integration import PrevalentIntegrationConfig


def _prepare_upstream(config, component):
    if component != "integration":
        service.train_integration(config)
    if component == "planner" and config["task"] == "prevalent":
        service.calibrate(config)


OPERATIONS = [
    ("prevalent", "integration", "train-integration", service.train_integration, "fusion_model"),
    ("longitudinal", "integration", "train-integration", service.train_integration, "fusion_model"),
    ("prevalent", "calibration", "calibrate", service.calibrate, "calibrators"),
    ("prevalent", "planner", "train-planner", service.train_planner, "q_model"),
    ("longitudinal", "planner", "train-planner", service.train_planner, "q_model"),
]


@pytest.mark.parametrize("task,component,command,operation,output_key", OPERATIONS)
@pytest.mark.parametrize("seed", [None, 19])
def test_records_actual_defaults_command_and_version(
    tmp_path, task, component, command, operation, output_key, seed
):
    config = _write_training_bundle(tmp_path / "bundle", task=task)
    for name in ("integration", "planner"):
        config["training"][name].pop("seed")
        if seed is not None:
            config["training"][name]["seed"] = seed
    config["training"]["calibration"] = {}
    config["training"]["longitudinal_reward"] = {}
    _prepare_upstream(config, component)
    summary = operation(config)
    artifact = load_trace_artifact(summary.artifact_path)
    manifest = json.loads(summary.manifest_path.read_text())
    assert manifest["command"] == artifact.training["command"] == command
    assert manifest["package_version"] == artifact.training["package_version"] == __version__
    settings = artifact.training["settings"]
    assert manifest["training"]["settings"] == json.loads(json.dumps(settings))
    if component == "integration":
        factory = PrevalentIntegrationConfig if task == "prevalent" else LongitudinalIntegrationConfig
        assert settings == dataclasses.asdict(factory.from_mapping(config["training"][component]))
    elif component == "calibration":
        assert settings == {"min_class_count": 10}
    else:
        expected = dataclasses.asdict(PlannerTrainingConfig.from_mapping(config["training"][component]))
        assert all(settings[key] == value for key, value in expected.items())
        assert settings["candidate_ranker"]["cost_gamma"] == 1.0
        assert settings["candidate_ranker"]["urgency_threshold"] == (0.5 if task == "prevalent" else 0.8)
        if task == "longitudinal":
            assert tuple(settings["longitudinal_reward"]["horizons"]) == (5.0, 10.0)
            assert tuple(settings["longitudinal_reward"]["horizon_weights"]) == (0.5, 0.5)
    expected_seed = None if component == "calibration" else (7 if seed is None else seed)
    assert manifest["random_seed"] == artifact.training["random_seed"] == expected_seed
    assert Path(manifest["metrics_path"]) == summary.manifest_path
    assert manifest["metrics_key"] == "metrics"


@pytest.mark.parametrize("task,component,command,operation,output_key", OPERATIONS)
@pytest.mark.parametrize("missing", [True, False])
def test_required_output_is_checked_before_fitting(
    tmp_path, monkeypatch, task, component, command, operation, output_key, missing
):
    config = _write_training_bundle(tmp_path / "bundle", task=task)
    _prepare_upstream(config, component)
    if missing:
        config["artifacts"].pop(output_key)
    else:
        config["artifacts"][output_key] = ""

    def forbidden_fit(*args, **kwargs):
        pytest.fail("required output config must be checked before fitting")

    names = {
        "integration": ("train_prevalent_integration", "train_longitudinal_integration"),
        "calibration": ("fit_prevalent_calibration",),
        "planner": ("train_prevalent_planner", "train_longitudinal_planner"),
    }
    for name in names[component]:
        monkeypatch.setattr(service, name, forbidden_fit)
    with pytest.raises(ValueError, match=f"artifacts.{output_key}"):
        operation(config)


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["stage_model", "stage_manifest", "publish_model", "publish_manifest"])
def test_failed_publication_preserves_complete_previous_pair(tmp_path, monkeypatch, existing, failure):
    config = _write_training_bundle(tmp_path / "bundle", task="prevalent")
    path = Path(config["artifacts"]["fusion_model"])
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if existing:
        service.train_integration(config)
    before = {file: file.read_bytes() if file.exists() else None for file in (path, manifest_path)}
    config["training"]["integration"]["seed"] = 17
    if failure in {"stage_model", "stage_manifest"}:
        name = "dump_trace_artifact" if failure == "stage_model" else "write_training_manifest"

        def broken_stage(*args, **kwargs):
            raise OSError("injected stage failure")

        monkeypatch.setattr(service, name, broken_stage)
    else:
        import os

        replace = os.replace
        failed = False
        destination = path if failure == "publish_model" else manifest_path

        def broken_publish(source, target):
            nonlocal failed
            if Path(target) == destination and not failed:
                failed = True
                raise OSError("injected publication failure")
            return replace(source, target)

        monkeypatch.setattr(os, "replace", broken_publish)
    with pytest.raises(OSError, match="injected"):
        service.train_integration(config)
    for file, old in before.items():
        if old is None:
            assert not file.exists()
        else:
            assert file.read_bytes() == old
    if existing:
        artifact = load_trace_artifact(path)
        assert artifact.training["settings"]["seed"] == 3
        manifest = json.loads(manifest_path.read_text())
        assert manifest["artifact"]["sha256"] == file_provenance(path)["sha256"]


@pytest.mark.parametrize("task,component,command,operation,output_key", OPERATIONS)
@pytest.mark.parametrize("invalid_path", ["directory", "parent_file"])
def test_invalid_output_destination_is_rejected_before_fit(
    tmp_path, monkeypatch, task, component, command, operation, output_key, invalid_path
):
    config = _write_training_bundle(tmp_path / "bundle", task=task)
    _prepare_upstream(config, component)
    blocked = tmp_path / "blocked"
    if invalid_path == "directory":
        blocked.mkdir()
        config["artifacts"][output_key] = str(blocked)
    else:
        blocked.write_text("not a directory")
        config["artifacts"][output_key] = str(blocked / "nested/model.joblib")

    def forbidden_fit(*args, **kwargs):
        pytest.fail("invalid output path must be rejected before fitting")

    for name in (
        "train_prevalent_integration",
        "train_longitudinal_integration",
        "fit_prevalent_calibration",
        "train_prevalent_planner",
        "train_longitudinal_planner",
    ):
        monkeypatch.setattr(service, name, forbidden_fit)
    with pytest.raises(ValueError, match=f"artifacts.{output_key}"):
        operation(config)


@pytest.mark.parametrize("sidecar", ["missing", "stale", "malformed", "wrong_task"])
def test_incomplete_published_model_is_rejected(tmp_path, sidecar):
    config = _write_training_bundle(tmp_path / "bundle", task="prevalent")
    summary = service.train_integration(config)
    artifact = load_trace_artifact(summary.artifact_path)
    if sidecar == "missing":
        summary.manifest_path.unlink()
    elif sidecar == "stale":
        dump_trace_artifact(
            dataclasses.replace(artifact, created_at="uncommitted replacement"), summary.artifact_path
        )
    elif sidecar == "malformed":
        summary.manifest_path.write_text("{invalid JSON")
    else:
        manifest = json.loads(summary.manifest_path.read_text())
        manifest["task"] = "longitudinal"
        summary.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        load_trace_artifact(summary.artifact_path)


def test_complete_published_pair_can_move_to_another_directory(tmp_path):
    import shutil

    config = _write_training_bundle(tmp_path / "bundle", task="prevalent")
    summary = service.train_integration(config)
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copy2(summary.artifact_path, moved / summary.artifact_path.name)
    shutil.copy2(summary.manifest_path, moved / summary.manifest_path.name)
    assert load_trace_artifact(moved / summary.artifact_path.name).component == "integration"


def test_legacy_envelope_does_not_require_a_sidecar(tmp_path):
    config = _write_training_bundle(tmp_path / "bundle", task="prevalent")
    summary = service.train_integration(config)
    artifact = load_trace_artifact(summary.artifact_path)
    training = dict(artifact.training)
    training.pop("manifest_required", None)
    old = dataclasses.replace(artifact, training=training)
    path = dump_trace_artifact(old, tmp_path / "legacy.joblib")
    assert load_trace_artifact(path).component == "integration"
