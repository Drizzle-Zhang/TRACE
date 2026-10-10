import json

import numpy as np
import pytest

from tests.module_fixtures import write_module_inputs
from trace_agent.data.arrays import load_module_score_bundle
from trace_agent.data.module_features import load_catalog, load_module_features
from trace_agent.risk_skills.module_model import ModuleModel


class ColumnEstimator:
    classes_ = np.array([0, 1])

    def predict_proba(self, matrix):
        probability = 1 / (1 + np.exp(-matrix[:, 0]))
        return np.column_stack([1 - probability, probability])


def make_batch(root):
    from trace_agent.training.modules.score_bundle import score_module_features

    features = load_module_features(load_catalog(write_module_inputs(root)))
    (root / "outcome.csv").unlink()
    model = ModuleModel("prevalent", "M1", "D1", ("b", "a"), ColumnEstimator())
    return features, score_module_features(features, {("M1", "D1"): model}, task="prevalent")


def test_scoring_uses_saved_pair_feature_order_and_not_outcomes(tmp_path):
    features, batch = make_batch(tmp_path)
    expected = 1 / (1 + np.exp(-features.matrices[("M1", "D1")][:, 0]))
    np.testing.assert_allclose(batch.scores[:, 0, 0], expected)
    assert batch.pred_mask[:, 0, 0].all()
    assert not batch.pred_mask[:, 1, 0].any()


def test_oof_cache_reordered_ids_fail():
    from trace_agent.training.modules.score_bundle import validate_oof_identity

    cache = {"train_rows": np.array([0, 1]), "train_eids": np.array(["11", "10"])}
    with pytest.raises(ValueError, match="OOF participant mismatch"):
        validate_oof_identity(cache, np.array([0, 1]), np.array(["10", "11", "12"]))


def test_training_final_model_predicts_only_requested_rows(tmp_path):
    from trace_agent.training.modules.score_bundle import score_module_features

    features, _ = make_batch(tmp_path)
    calls = []

    class SpyEstimator(ColumnEstimator):
        def predict_proba(self, matrix):
            calls.append(matrix.copy())
            return super().predict_proba(matrix)

    model = ModuleModel("prevalent", "M1", "D1", ("b", "a"), SpyEstimator())
    batch = score_module_features(
        features, {("M1", "D1"): model}, task="prevalent", predict_rows=np.arange(120, 180)
    )
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], features.matrices[("M1", "D1")][120:])
    assert not batch.pred_mask[:120].any()


def test_apply_oof_replaces_train_scores_not_validation(tmp_path):
    from trace_agent.training.modules.score_bundle import apply_train_oof

    _, batch = make_batch(tmp_path)
    rows = np.arange(120)
    scores = np.full((120, 2, 1), 0.123, dtype=np.float32)
    mask = np.ones(scores.shape, dtype=bool)
    result = apply_train_oof(
        batch,
        {"train_rows": rows, "train_eids": batch.participant_ids[rows], "scores": scores, "pred_mask": mask},
        rows,
    )
    np.testing.assert_array_equal(result.scores[:120], scores)
    np.testing.assert_array_equal(result.scores[120:], batch.scores[120:])


@pytest.mark.parametrize("legacy", [False, True])
def test_score_bundle_load_without_any_targets(tmp_path, legacy):
    from trace_agent.training.modules.score_bundle import write_score_bundle

    _, batch = make_batch(tmp_path)
    path = write_score_bundle(tmp_path / "bundle", prefix="demo", batch=batch, provenance={})
    if legacy:
        path.unlink()
    loaded = load_module_score_bundle(path.parent, prefix="demo", task="prevalent")
    np.testing.assert_array_equal(loaded.scores, batch.scores)


def test_score_bundle_publishes_canonical_participant_identifier_artifact(tmp_path):
    from trace_agent.contracts.scores import ModuleScoreBatch
    from trace_agent.training.modules.score_bundle import write_score_bundle

    batch = ModuleScoreBatch(
        participant_ids=np.array(["001", "NA", "sample-1", "α"]),
        modules=("M1",),
        endpoints=("D1",),
        scores=np.array([[[0.2]], [[np.nan]], [[0.3]], [[0.4]]]),
        pred_mask=np.array([[[True]], [[False]], [[True]], [[True]]]),
        availability=np.ones((4, 1), bool),
        task="prevalent",
    )
    path = write_score_bundle(tmp_path, prefix="demo", batch=batch, provenance={})

    manifest = json.loads(path.read_text())
    assert "participant_ids" in manifest["files"]
    assert "eids" not in manifest["files"]
    assert manifest["files"]["participant_ids"]["file"] == "demo_participant_ids.npy"
    assert not (tmp_path / "demo_eids.npy").exists()
    loaded = load_module_score_bundle(tmp_path, prefix="demo", task="prevalent")
    np.testing.assert_array_equal(loaded.participant_ids, batch.participant_ids)
    np.testing.assert_array_equal(loaded.scores, batch.scores)
    np.testing.assert_array_equal(loaded.pred_mask, batch.pred_mask)


def test_score_bundle_tampering_fails_immediately(tmp_path):
    from trace_agent.training.modules.score_bundle import write_score_bundle

    _, batch = make_batch(tmp_path)
    path = write_score_bundle(tmp_path / "bundle", prefix="demo", batch=batch, provenance={})
    np.save(path.parent / "demo_oof_probs.npy", np.zeros(batch.scores.shape))
    with pytest.raises(ValueError, match="score bundle hash mismatch"):
        load_module_score_bundle(path.parent, prefix="demo", task="prevalent")


def test_saved_feature_order_mismatch_fails(tmp_path):
    from trace_agent.training.modules.score_bundle import score_module_features

    features, _ = make_batch(tmp_path)
    model = ModuleModel("prevalent", "M1", "D1", ("a", "b"), ColumnEstimator())
    with pytest.raises(ValueError, match="feature order"):
        score_module_features(features, {("M1", "D1"): model}, task="prevalent")


def test_scoring_rejects_mis_keyed_model_identity(tmp_path):
    from trace_agent.training.modules.score_bundle import score_module_features

    features, _ = make_batch(tmp_path)
    model = ModuleModel("prevalent", "OTHER", "D1", ("b", "a"), ColumnEstimator())
    with pytest.raises(ValueError, match="identity"):
        score_module_features(features, {("M1", "D1"): model}, task="prevalent")
