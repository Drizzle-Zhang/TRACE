"""Module OOF scores must describe the labels used for their train fits."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.module_fixtures import module_config, write_module_inputs
from trace_agent.data.training_dataset import load_training_dataset
from trace_agent.training.modules.service import prepare_module_data, score_modules, train_modules


def _trained(root, task):
    if task == "longitudinal":
        pytest.importorskip("xgboost")
    write_module_inputs(root, task=task)
    config = module_config(root, task=task)
    config["module_data"]["score_mode"] = "training"
    prepare_module_data(config)
    train_modules(config)
    score_modules(config)
    return config


def _dataset(root, task):
    return load_training_dataset(
        root / "prepared", prefix="cohort", task=task, split_table=root / "splits.csv"
    )


@pytest.mark.parametrize(
    ("task", "change"),
    [
        ("prevalent", "labels"),
        ("prevalent", "mask"),
        ("longitudinal", "labels"),
        ("longitudinal", "mask"),
        ("longitudinal", "time"),
    ],
)
def test_updated_train_targets_cannot_reuse_old_module_oof(tmp_path, task, change):
    config = _trained(tmp_path, task)
    outcomes = pd.read_csv(tmp_path / "outcome.csv")
    if change == "labels":
        outcomes.loc[:119, "target_y"] = 1 - outcomes.loc[:119, "target_y"]
        if task == "prevalent":
            outcomes.loc[:119, "BL2Target_yrs"] = np.where(outcomes.loc[:119, "target_y"] == 1, -2.0, 12.0)
    elif change == "mask":
        outcomes.loc[0, "target_y"] = np.nan
    else:
        outcomes.loc[0, "BL2Target_yrs"] += 3
    outcomes.to_csv(tmp_path / "outcome.csv", index=False)
    prepare_module_data(config)
    # Scoring remains label-free. The supervised boundary must reject stale OOF.
    score_modules(config)
    with pytest.raises(ValueError, match="module.*train.*target"):
        _dataset(tmp_path, task)


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
def test_updated_validation_test_targets_keep_existing_module_oof_valid(tmp_path, task):
    config = _trained(tmp_path, task)
    before = _dataset(tmp_path, task)
    outcomes = pd.read_csv(tmp_path / "outcome.csv")
    outcomes.loc[120:, "target_y"] = 1 - outcomes.loc[120:, "target_y"]
    if task == "prevalent":
        outcomes.loc[120:, "BL2Target_yrs"] = np.where(outcomes.loc[120:, "target_y"] == 1, -4.0, 15.0)
    else:
        outcomes.loc[120:, "BL2Target_yrs"] = 8.0
    outcomes.loc[[120, 150], "target_y"] = np.nan
    outcomes.to_csv(tmp_path / "outcome.csv", index=False)
    prepare_module_data(config)
    score_modules(config)
    after = _dataset(tmp_path, task)
    np.testing.assert_array_equal(before.scores.scores, after.scores.scores)
    assert not np.array_equal(before.targets.labels[120:], after.targets.labels[120:], equal_nan=True)


def test_legacy_module_bank_without_target_fingerprints_is_marked_unverified(tmp_path):
    config = _trained(tmp_path, "prevalent")
    bank_path = tmp_path / "bank/module_bank_manifest.json"
    manifest = json.loads(bank_path.read_text())
    manifest.pop("train_target_fingerprints", None)
    bank_path.write_text(json.dumps(manifest))
    score_path = score_modules(config)
    provenance = json.loads(score_path.read_text())["provenance"]
    assert provenance["module_training_target_identity"] == "legacy_unverified"
    assert "train_target_fingerprints" not in provenance
    assert _dataset(tmp_path, "prevalent").scores.pred_mask.all()


@pytest.mark.parametrize("change", ["missing_pair", "skipped_pair_has_oof", "model_target_mismatch"])
def test_module_bank_and_cached_pair_contract_must_be_complete(tmp_path, change):
    config = _trained(tmp_path, "prevalent")
    bank_path = tmp_path / "bank/module_bank_manifest.json"
    manifest = json.loads(bank_path.read_text())
    if change == "missing_pair":
        manifest["pairs"].pop()
        message = "module bank.*pair"
    elif change == "skipped_pair_has_oof":
        manifest["pairs"][0]["status"] = "skipped"
        message = "OOF.*pair"
    else:
        manifest["train_target_fingerprints"]["y"] = "0" * 64
        message = "module bank.*train.*target"
    bank_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        score_modules(config)


@pytest.mark.parametrize("case", ["skipped", "omitted"])
def test_declared_skip_and_unconfigured_pairs_are_valid(tmp_path, case):
    write_module_inputs(tmp_path)
    config = module_config(tmp_path, task="prevalent")
    config["module_data"]["score_mode"] = "training"
    if case == "skipped":
        config["training"]["modules"]["min_positive"] = 1000
    else:
        path = tmp_path / "catalog.yaml"
        catalog = yaml.safe_load(path.read_text())
        catalog["pairs"].pop()
        path.write_text(yaml.safe_dump(catalog))
    prepare_module_data(config)
    train_modules(config)
    score_modules(config)
    dataset = _dataset(tmp_path, "prevalent")
    if case == "skipped":
        assert not dataset.scores.pred_mask.any()
    else:
        assert dataset.scores.pred_mask[:, 0, :].all()
        assert not dataset.scores.pred_mask[:, 1, :].any()
