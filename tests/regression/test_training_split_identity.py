"""Native upstream artifacts must respect the current supervised split."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from tests.training_fixtures import rewrite_training_artifact
from tests.unit.test_training_service import _write_training_bundle
from trace_agent.runtime.model_artifact import load_trace_artifact
from trace_agent.training.service import calibrate, train_integration, train_planner


def _swap_train_and_test(config):
    path = Path(config["data"]["split_table"]["path"])
    frame = pd.read_csv(path)
    frame.loc[[0, 7], "partition"] = ["test", "train"]
    frame.to_csv(path, index=False)


@pytest.mark.parametrize(
    ("task", "command"),
    [("prevalent", calibrate), ("prevalent", train_planner), ("longitudinal", train_planner)],
)
def test_supervised_stage_rejects_fusion_trained_on_the_new_test_split(tmp_path, task, command):
    config = _write_training_bundle(tmp_path / task, task=task)
    train_integration(config)
    if task == "prevalent":
        calibrate(config)
    _swap_train_and_test(config)
    with pytest.raises(ValueError, match="integration.*split"):
        command(config)


def test_planner_rejects_old_calibration_after_fusion_is_retrained(tmp_path):
    config = _write_training_bundle(tmp_path / "data", task="prevalent")
    train_integration(config)
    calibrate(config)
    _swap_train_and_test(config)
    train_integration(config)
    with pytest.raises(ValueError, match="calibration.*split"):
        train_planner(config)


def test_existing_per_role_fingerprints_are_checked_without_canonical_hash(tmp_path):
    config = _write_training_bundle(tmp_path / "data", task="prevalent")
    train_integration(config)
    path = config["artifacts"]["fusion_model"]
    artifact = load_trace_artifact(path)
    rewrite_training_artifact(replace(artifact, provenance={"splits": artifact.provenance["splits"]}), path)
    _swap_train_and_test(config)
    with pytest.raises(ValueError, match="integration.*split"):
        calibrate(config)


def test_legacy_envelopes_without_split_fingerprints_keep_their_interface(tmp_path):
    config = _write_training_bundle(tmp_path / "data", task="prevalent")
    train_integration(config)
    path = config["artifacts"]["fusion_model"]
    artifact = load_trace_artifact(path)
    rewrite_training_artifact(replace(artifact, provenance={}), path)
    calibrate(config)
    path = config["artifacts"]["calibrators"]
    artifact = load_trace_artifact(path)
    rewrite_training_artifact(replace(artifact, provenance={}), path)
    assert train_planner(config).artifact_path.is_file()


def test_split_file_reordering_does_not_invalidate_canonical_identity(tmp_path):
    config = _write_training_bundle(tmp_path / "data", task="prevalent")
    train_integration(config)
    path = Path(config["data"]["split_table"]["path"])
    pd.read_csv(path).iloc[::-1].to_csv(path, index=False)
    assert calibrate(config).artifact_path.is_file()
