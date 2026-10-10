"""Native banks must keep final models and train OOF in one training identity."""

import json
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.module_fixtures import module_config, write_module_inputs
from trace_agent import __version__
from trace_agent.data.arrays import load_module_score_bundle
from trace_agent.risk_skills.module_model import ModuleModel
from trace_agent.runtime.model_artifact import dump_trace_artifact, load_trace_artifact
from trace_agent.training.modules.score_bundle import load_module_bank, load_oof_cache, local_record
from trace_agent.training.modules.service import prepare_module_data, score_modules, train_modules


def _train(root, *, task="prevalent", missing=False, feature_offset=0, partial_available_pair=False):
    if task == "longitudinal":
        pytest.importorskip("xgboost")
    write_module_inputs(root, task=task)
    frame = pd.read_csv(root / "features.csv")
    frame[["a", "b"]] += feature_offset
    if missing:
        frame.loc[0, ["a", "b"]] = np.nan
        outcomes = pd.read_csv(root / "outcome.csv")
        # A future event is inapplicable to prevalence; a baseline case to incidence.
        outcomes.loc[1, "BL2Target_yrs"] = 2.0 if task == "prevalent" else -2.0
        outcomes.to_csv(root / "outcome.csv", index=False)
    frame.to_csv(root / "features.csv", index=False)
    if partial_available_pair:
        path = root / "catalog.yaml"
        catalog = yaml.safe_load(path.read_text())
        catalog["endpoints"].append({"name": "D2", "outcome_path": "outcome.csv"})
        catalog["pairs"].append({"module": "M1", "endpoint": "D2", "features_path": "M2_D1.csv"})
        path.write_text(yaml.safe_dump(catalog, sort_keys=False))
    config = module_config(root, task=task)
    config["module_data"]["score_mode"] = "training"
    prepare_module_data(config)
    train_modules(config)
    return config


@pytest.fixture(scope="module")
def trained_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("bank-integrity")
    _train(root)
    return root


def _copy_bank(trained_root, tmp_path):
    shutil.copytree(trained_root, tmp_path, dirs_exist_ok=True)
    config = module_config(tmp_path, task="prevalent")
    config["module_data"]["score_mode"] = "training"
    path = tmp_path / "bank/module_bank_manifest.json"
    return config, path, json.loads(path.read_text())


def _rewrite_cache(root, manifest_path, manifest, mutate):
    path = root / "bank" / manifest["oof_cache"]["file"]
    with np.load(path, allow_pickle=False) as archive:
        cache = {name: archive[name].copy() for name in archive.files}
    mutate(cache)
    np.savez_compressed(path, **cache)
    manifest["oof_cache"] = local_record(path)
    manifest_path.write_text(json.dumps(manifest))


def test_recorded_oof_coverage_must_match_rehashed_cache(trained_root, tmp_path):
    config, path, manifest = _copy_bank(trained_root, tmp_path)
    assert manifest["pairs"][0]["metadata"]["oof_valid_count"] == 120

    def drop_prediction(cache):
        cache["pred_mask"][0, 0, 0] = False
        cache["scores"][0, 0, 0] = np.nan

    _rewrite_cache(tmp_path, path, manifest, drop_prediction)
    with pytest.raises(ValueError, match="OOF.*coverage"):
        load_oof_cache(tmp_path / "bank", manifest)
    with pytest.raises(ValueError, match="OOF.*coverage"):
        score_modules(config)


def test_missing_expected_oof_cannot_be_hidden_by_lowering_recorded_count(trained_root, tmp_path):
    config, path, manifest = _copy_bank(trained_root, tmp_path)

    def drop_prediction(cache):
        cache["pred_mask"][0, 0, 0] = False
        cache["scores"][0, 0, 0] = np.nan

    manifest["pairs"][0]["metadata"]["oof_valid_count"] = 119
    _rewrite_cache(tmp_path, path, manifest, drop_prediction)
    with pytest.raises(ValueError, match="OOF.*coverage"):
        score_modules(config)


def test_same_count_wrong_oof_rows_are_rejected(tmp_path):
    # M1 remains physically available through D2, so module availability alone
    # cannot catch a prediction moved onto D1's entirely missing feature row.
    config = _train(tmp_path, missing=True, partial_available_pair=True)
    path = tmp_path / "bank/module_bank_manifest.json"
    manifest = json.loads(path.read_text())
    assert manifest["pairs"][0]["metadata"]["oof_valid_count"] == 119

    def move_prediction(cache):
        cache["pred_mask"][0, 0, 0] = True
        cache["scores"][0, 0, 0] = 0.5
        cache["pred_mask"][1, 0, 0] = False
        cache["scores"][1, 0, 0] = np.nan
        assert cache["pred_mask"][:, 0, 0].sum() == 119

    _rewrite_cache(tmp_path, path, manifest, move_prediction)
    with pytest.raises(ValueError, match="OOF.*coverage"):
        score_modules(config)


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
def test_missing_features_and_inapplicable_labels_keep_their_oof_contract(tmp_path, task):
    config = _train(tmp_path, task=task, missing=True)
    label_mask_name = "y_mask" if task == "prevalent" else "label_mask"
    assert not np.load(tmp_path / f"prepared/cohort_{label_mask_name}.npy")[1, 0]
    # Training scoring never needs labels or a replacement train prediction.
    (tmp_path / "outcome.csv").unlink()
    for path in (tmp_path / "prepared").glob("*.npy"):
        path.unlink()
    score_modules(config)
    batch = load_module_score_bundle(tmp_path / "prepared", prefix="cohort", task=task)
    assert not batch.pred_mask[0, 0, 0]
    assert np.isnan(batch.scores[0, 0, 0])
    assert batch.pred_mask[1, 0, 0]
    assert np.isfinite(batch.scores[1, 0, 0])
    with np.load(tmp_path / "bank/train_oof.npz", allow_pickle=False) as cache:
        np.testing.assert_array_equal(batch.scores[:120], cache["scores"])


def test_rehashed_model_from_another_feature_source_cannot_join_bank(trained_root, tmp_path):
    config, path, manifest = _copy_bank(trained_root, tmp_path / "A")
    _train(tmp_path / "B", feature_offset=10000)
    filename = manifest["pairs"][0]["artifact"]["file"]
    target = path.parent / filename
    shutil.copyfile(tmp_path / "B/bank" / filename, target)
    artifact = load_trace_artifact(target, component="module", task="prevalent")
    assert artifact.provenance["train_target_fingerprints"] == manifest["train_target_fingerprints"]
    assert (
        artifact.provenance["inputs"]["table:baseline"]["sha256"]
        != manifest["feature_sources"]["table:baseline"]
    )
    manifest["pairs"][0]["artifact"] = local_record(target)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="module bank.*(input|feature).*mismatch"):
        load_module_bank(path.parent, task="prevalent")
    with pytest.raises(ValueError, match="module bank.*(input|feature).*mismatch"):
        score_modules(config)


@pytest.mark.parametrize("change", ["split", "folds", "fold_source", "settings", "features", "scale"])
def test_bank_checks_recorded_model_identity(trained_root, tmp_path, change):
    _, path, manifest = _copy_bank(trained_root, tmp_path)
    if change in {"split", "folds"}:
        manifest[f"{change}_sha256"] = "0" * 64
    elif change == "fold_source":
        manifest["fold_source"] = {"sha256": "0" * 64}
    elif change == "settings":
        manifest["settings"]["seed"] += 1
    elif change == "features":
        manifest["feature_names"][0]["names"].reverse()
    else:
        manifest["pairs"][0]["score_scale"] = "log_risk"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="module bank.*mismatch"):
        load_module_bank(path.parent, task="prevalent")


@pytest.mark.parametrize("field", ["feature_sources", "split_sha256", "folds_sha256", "settings"])
def test_modern_bank_cannot_omit_recorded_identity(trained_root, tmp_path, field):
    _, path, manifest = _copy_bank(trained_root, tmp_path)
    manifest.pop(field)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="module bank.*(missing|incomplete)"):
        load_module_bank(path.parent, task="prevalent")


@pytest.mark.parametrize("field", ["inputs", "split_table_canonical_sha256", "folds_sha256"])
def test_modern_model_cannot_omit_recorded_identity(trained_root, tmp_path, field):
    _, path, manifest = _copy_bank(trained_root, tmp_path)
    record = manifest["pairs"][0]
    artifact_path = path.parent / record["artifact"]["file"]
    artifact = load_trace_artifact(artifact_path, component="module", task="prevalent")
    provenance = dict(artifact.provenance)
    provenance.pop(field)
    dump_trace_artifact(replace(artifact, provenance=provenance), artifact_path)
    record["artifact"] = local_record(artifact_path)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="module bank.*mismatch"):
        load_module_bank(path.parent, task="prevalent")


def test_modern_bank_requires_oof_coverage_record(trained_root, tmp_path):
    _, _, manifest = _copy_bank(trained_root, tmp_path)
    manifest["pairs"][0]["metadata"].pop("oof_valid_count")
    with pytest.raises(ValueError, match="OOF.*coverage"):
        load_oof_cache(tmp_path / "bank", manifest)


def test_native_module_training_records_command_package_and_resolved_seed(trained_root):
    manifest, _ = load_module_bank(trained_root / "bank", task="prevalent")
    for metadata in [manifest["training"]] + [
        load_trace_artifact(
            trained_root / "bank" / pair["artifact"]["file"], component="module", task="prevalent"
        ).training
        for pair in manifest["pairs"]
    ]:
        assert metadata["command"] == "train-modules"
        assert metadata["package_version"] == __version__
        assert metadata["random_seed"] == manifest["settings"]["seed"] == 7


def test_nonfinite_native_oof_cannot_publish_complete_bank(tmp_path, monkeypatch):
    predict = ModuleModel.predict
    injected = False

    def nonfinite_prediction(model, matrix):
        nonlocal injected
        values = predict(model, matrix)
        if not injected:
            assert np.isfinite(matrix[0]).any()
            values[0] = np.nan
            injected = True
        return values

    # Keep the real native fits; inject one nonfinite outer-held prediction.
    monkeypatch.setattr(ModuleModel, "predict", nonfinite_prediction)
    with pytest.raises(ValueError, match="OOF.*coverage"):
        _train(tmp_path)
    assert injected
    assert not (tmp_path / "bank/module_bank_manifest.json").exists()
    assert not list((tmp_path / "bank").glob("*.joblib"))
