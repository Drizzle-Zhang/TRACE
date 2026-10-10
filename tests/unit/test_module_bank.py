import json

import numpy as np
import pytest

from trace_agent.contracts.task import TaskType
from trace_agent.risk_skills.module_model import ModuleModel
from trace_agent.runtime.model_artifact import TraceModelArtifact, dump_trace_artifact


class BankEstimator:
    classes_ = np.array([0, 1])

    def predict_proba(self, matrix):
        return np.tile([0.6, 0.4], (matrix.shape[0], 1))


def write_bank(root):
    from trace_agent.training.modules.score_bundle import local_record, pair_filename

    model = ModuleModel("prevalent", "M1", "D1", ("x",), BankEstimator())
    path = dump_trace_artifact(
        TraceModelArtifact("module", TaskType.PREVALENT, model, ("M1",), ("D1",)),
        root / pair_filename("M1", "D1"),
    )
    manifest = {
        "schema_version": 1,
        "task": "prevalent",
        "modules": ["M1", "M2"],
        "endpoints": ["D1"],
        "pairs": [
            {"module": "M1", "endpoint": "D1", "status": "trained", "artifact": local_record(path)},
            {"module": "M2", "endpoint": "D1", "status": "skipped", "reason": "rare"},
        ],
    }
    (root / "module_bank_manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_pair_filename_does_not_expose_path_components():
    from trace_agent.training.modules.score_bundle import pair_filename

    name = pair_filename("../../M1", "disease/name")
    assert "/" not in name and name.endswith(".joblib")
    assert name == pair_filename("../../M1", "disease/name")


@pytest.mark.parametrize("name", ["../model.joblib", "/model.joblib", "..", ""])
def test_checked_file_rejects_non_local_filename(tmp_path, name):
    from trace_agent.training.modules.score_bundle import checked_file

    with pytest.raises(ValueError, match="local filename"):
        checked_file(tmp_path, {"file": name, "sha256": "none"})


def test_checked_file_rejects_changed_content(tmp_path):
    from trace_agent.training.modules.score_bundle import checked_file, local_record

    path = tmp_path / "model.joblib"
    path.write_bytes(b"before")
    record = local_record(path)
    path.write_bytes(b"after")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        checked_file(tmp_path, record)


def test_bank_model_roundtrip_and_skipped_pair(tmp_path):
    from trace_agent.training.modules.score_bundle import load_module_bank

    write_bank(tmp_path)
    manifest, models = load_module_bank(tmp_path, task="prevalent")
    assert len(manifest["pairs"]) == 2
    assert set(models) == {("M1", "D1")}
    np.testing.assert_allclose(models[("M1", "D1")].predict(np.ones((2, 1))), 0.4)


@pytest.mark.parametrize("change", ["schema", "task", "identity", "duplicate_skipped", "status"])
def test_invalid_bank_manifest_errors(tmp_path, change):
    from trace_agent.training.modules.score_bundle import load_module_bank

    manifest = write_bank(tmp_path)
    if change == "schema":
        manifest["schema_version"] = 2
    elif change == "task":
        manifest["task"] = "longitudinal"
    elif change == "identity":
        manifest["pairs"][0]["module"] = "M2"
    elif change == "duplicate_skipped":
        manifest["pairs"].append(manifest["pairs"][1])
    else:
        manifest["pairs"][0]["status"] = "other"
    (tmp_path / "module_bank_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_module_bank(tmp_path, task="prevalent")


def test_missing_bank_manifest_errors(tmp_path):
    from trace_agent.training.modules.score_bundle import load_module_bank

    with pytest.raises(FileNotFoundError):
        load_module_bank(tmp_path, task="prevalent")


def test_bank_rejects_wrong_artifact_component(tmp_path):
    from trace_agent.training.modules.score_bundle import load_module_bank, local_record

    manifest = write_bank(tmp_path)
    path = tmp_path / manifest["pairs"][0]["artifact"]["file"]
    dump_trace_artifact(
        TraceModelArtifact("integration", TaskType.PREVALENT, BankEstimator(), ("M1",), ("D1",)), path
    )
    manifest["pairs"][0]["artifact"] = local_record(path)
    (tmp_path / "module_bank_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="expected component module"):
        load_module_bank(tmp_path, task="prevalent")


@pytest.mark.parametrize("change", [None, "fields", "axes", "mask", "hash"])
def test_oof_cache_validation(tmp_path, change):
    from trace_agent.training.modules.score_bundle import load_oof_cache, local_record, write_oof_cache

    path = tmp_path / "oof.npz"
    scores = np.ones((5, 2, 1), dtype=np.float32)
    mask = np.ones(scores.shape, dtype=bool)
    if change == "mask":
        mask[0] = False
    write_oof_cache(
        path,
        train_rows=np.arange(5),
        train_eids=np.arange(5).astype(str),
        folds=np.arange(1, 6),
        scores=scores[:, :1] if change == "axes" else scores,
        pred_mask=mask,
    )
    if change == "fields":
        np.savez(path, scores=scores)
    manifest = {"modules": ["M1", "M2"], "endpoints": ["D1"], "oof_cache": local_record(path)}
    if change == "hash":
        path.write_bytes(b"tampered")
    if change is None:
        cache = load_oof_cache(tmp_path, manifest)
        np.testing.assert_array_equal(cache["scores"], scores)
        assert cache["train_eids"].dtype.kind == "U"
    else:
        with pytest.raises(ValueError):
            load_oof_cache(tmp_path, manifest)
