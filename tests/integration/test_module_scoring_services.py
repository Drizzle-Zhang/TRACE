import json

import numpy as np
import pandas as pd
import pytest

from tests.module_fixtures import module_config, write_module_inputs
from trace_agent.data.arrays import load_module_score_bundle
from trace_agent.training.modules.service import prepare_module_data, train_modules


def trained_config(root, task="prevalent"):
    if task == "longitudinal":
        pytest.importorskip("xgboost")
    write_module_inputs(root, task=task)
    config = module_config(root, task=task)
    prepare_module_data(config)
    train_modules(config)
    config["module_data"]["score_mode"] = "training"
    return config


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
def test_training_scoring_does_not_open_outcomes_or_targets(tmp_path, task, monkeypatch):
    from trace_agent.training.modules import service

    config = trained_config(tmp_path, task)
    (tmp_path / "outcome.csv").unlink()
    for path in (tmp_path / "prepared").glob("*.npy"):
        path.unlink()

    def forbidden(*args, **kwargs):
        pytest.fail("scoring must not open training targets")

    monkeypatch.setattr(service, "load_training_targets", forbidden)
    monkeypatch.setattr(service, "load_module_outcomes", forbidden)
    service.score_modules(config)
    batch = load_module_score_bundle(tmp_path / "prepared", prefix="cohort", task=task)
    assert batch.scores.shape == (180, 2, 1)
    assert batch.pred_mask.all()
    with np.load(tmp_path / "bank/train_oof.npz", allow_pickle=False) as cache:
        np.testing.assert_array_equal(batch.scores[:120], cache["scores"])
        np.testing.assert_array_equal(batch.pred_mask[:120], cache["pred_mask"])


def test_training_scoring_missing_oof_is_an_error(tmp_path):
    from trace_agent.training.modules.service import score_modules

    config = trained_config(tmp_path)
    (tmp_path / "bank/train_oof.npz").unlink()
    with pytest.raises(FileNotFoundError):
        score_modules(config)


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
def test_new_cohort_inference_needs_no_outcomes_splits_or_oof(tmp_path, task):
    from trace_agent.training.modules.service import score_modules

    config = trained_config(tmp_path, task)
    config["module_data"]["score_mode"] = "inference"
    (tmp_path / "outcome.csv").unlink()
    (tmp_path / "splits.csv").unlink()
    (tmp_path / "bank/train_oof.npz").unlink()
    for path in (tmp_path / "prepared").iterdir():
        path.unlink()
    for name in ("participants.csv", "features.csv"):
        path = tmp_path / name
        frame = pd.read_csv(path).iloc[:17].copy()
        frame["eid"] += 10000
        if name == "features.csv":
            frame["a"] += 1
        frame.to_csv(path, index=False)
    score_modules(config)
    batch = load_module_score_bundle(tmp_path / "prepared", prefix="cohort", task=task)
    assert batch.scores.shape == (17, 2, 1)
    assert batch.pred_mask.all()
    np.testing.assert_array_equal(batch.participant_ids, np.arange(11000, 11017).astype(str))


@pytest.mark.parametrize("change", ["features", "split", "eids", "folds"])
def test_training_scoring_contract_mismatches_error(tmp_path, change):
    from trace_agent.training.modules.score_bundle import local_record
    from trace_agent.training.modules.service import score_modules

    config = trained_config(tmp_path)
    if change == "features":
        path = tmp_path / "features.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "a"] += 1
        frame.to_csv(path, index=False)
    elif change == "split":
        path = tmp_path / "splits.csv"
        frame = pd.read_csv(path)
        frame.loc[[0, 120], "split"] = ["validation", "train"]
        frame.to_csv(path, index=False)
    else:
        path = tmp_path / "bank/train_oof.npz"
        with np.load(path, allow_pickle=False) as archive:
            cache = {key: archive[key].copy() for key in archive.files}
        if change == "eids":
            cache["train_eids"] = cache["train_eids"][::-1]
        else:
            cache["folds"] = 6 - cache["folds"]
        np.savez(path, **cache)
        manifest_path = path.parent / "module_bank_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["oof_cache"] = local_record(path)
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        score_modules(config)


def test_score_mode_is_explicit(tmp_path):
    from trace_agent.training.modules.service import score_modules

    config = trained_config(tmp_path)
    del config["module_data"]["score_mode"]
    with pytest.raises(ValueError, match="score_mode"):
        score_modules(config)
