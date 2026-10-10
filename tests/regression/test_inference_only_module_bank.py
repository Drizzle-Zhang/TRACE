"""Inference exports require feature identity without invented training records."""

import json
from dataclasses import replace

import numpy as np
import pytest
import yaml

from tests.module_fixtures import module_config, write_module_inputs
from trace_agent.data.arrays import load_module_score_bundle
from trace_agent.data.module_features import load_catalog, load_module_features
from trace_agent.risk_skills.module_model import ModuleModel
from trace_agent.runtime.model_artifact import TraceModelArtifact, dump_trace_artifact, load_trace_artifact
from trace_agent.training.modules.score_bundle import local_record, pair_filename, score_module_features
from trace_agent.training.modules.service import score_modules


class InferenceEstimator:
    classes_ = np.array([0, 1])

    def predict_proba(self, matrix):
        positive = 1.0 / (1.0 + np.exp(-matrix[:, 0]))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, matrix, *, output_margin):
        assert output_margin
        return matrix[:, 0]


def _write_inference_bank(root, *, task="prevalent"):
    write_module_inputs(root, task=task)
    config = module_config(root, task=task)
    config["module_data"]["score_mode"] = "inference"
    features = load_module_features(load_catalog(root / "catalog.yaml"))
    bank = root / "bank"
    records = []
    models = {}
    for key, names in features.feature_names.items():
        model = ModuleModel(task, *key, names, InferenceEstimator())
        path = dump_trace_artifact(
            TraceModelArtifact("module", task, model, (key[0],), (key[1],)),
            bank / pair_filename(*key),
        )
        records.append(
            {
                "module": key[0],
                "endpoint": key[1],
                "status": "trained",
                "selected_features": list(names),
                "score_scale": model.score_scale,
                "artifact": local_record(path),
            }
        )
        models[key] = model
    manifest = {
        "schema_version": 1,
        "task": task,
        "modules": [item.name for item in features.catalog.modules],
        "endpoints": [item.name for item in features.catalog.endpoints],
        "inference_only": True,
        "participant_inputs_included": False,
        "pairs": records,
    }
    path = bank / "module_bank_manifest.json"
    path.write_text(json.dumps(manifest))
    (root / "outcome.csv").unlink()
    (root / "splits.csv").unlink()
    return config, path, manifest, features, models


@pytest.mark.parametrize("task", ["prevalent", "longitudinal"])
def test_minimal_inference_bank_scores_without_training_identity(tmp_path, task):
    config, _, _, features, models = _write_inference_bank(tmp_path, task=task)
    expected = score_module_features(features, models, task=task)

    score_modules(config)

    actual = load_module_score_bundle(tmp_path / "prepared", prefix="cohort", task=task)
    np.testing.assert_array_equal(actual.scores, expected.scores)
    np.testing.assert_array_equal(actual.pred_mask, expected.pred_mask)
    np.testing.assert_array_equal(actual.availability, expected.availability)


@pytest.mark.parametrize("change", ["record_order", "model_order", "missing_pair"])
def test_minimal_inference_bank_rejects_mismatched_feature_or_pair_identity(tmp_path, change):
    config, path, manifest, _, _ = _write_inference_bank(tmp_path)
    record = manifest["pairs"][0]
    if change == "record_order":
        record["selected_features"].reverse()
    elif change == "model_order":
        artifact_path = path.parent / record["artifact"]["file"]
        artifact = load_trace_artifact(artifact_path, component="module", task="prevalent")
        altered = replace(artifact.model, feature_names=tuple(reversed(artifact.model.feature_names)))
        dump_trace_artifact(replace(artifact, model=altered), artifact_path)
        record["artifact"] = local_record(artifact_path)
    else:
        manifest["pairs"].pop()
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="module bank.*(feature|pair).*mismatch"):
        score_modules(config)
    assert not (tmp_path / "prepared/cohort_scores.npy").exists()


def test_minimal_inference_bank_allows_skipped_pair_without_selected_features(tmp_path):
    config, path, manifest, _, _ = _write_inference_bank(tmp_path)
    manifest["pairs"][1] = {
        "module": "M2",
        "endpoint": "D1",
        "status": "skipped",
        "reason": "no original fitted model",
    }
    path.write_text(json.dumps(manifest))
    catalog_path = tmp_path / "catalog.yaml"
    catalog = yaml.safe_load(catalog_path.read_text())
    catalog["pairs"].pop()
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False))

    score_modules(config)

    batch = load_module_score_bundle(tmp_path / "prepared", prefix="cohort", task="prevalent")
    assert batch.pred_mask[:, 0, 0].all()
    assert not batch.pred_mask[:, 1, 0].any()
    assert np.isnan(batch.scores[:, 1, 0]).all()


def test_minimal_inference_bank_requires_every_trained_pair_in_catalog(tmp_path):
    config, _, _, _, _ = _write_inference_bank(tmp_path)
    catalog_path = tmp_path / "catalog.yaml"
    catalog = yaml.safe_load(catalog_path.read_text())
    catalog["pairs"].pop()
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False))

    with pytest.raises(ValueError, match="module bank.*pair.*mismatch"):
        score_modules(config)
