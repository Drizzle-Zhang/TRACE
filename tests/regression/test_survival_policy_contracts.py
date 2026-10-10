"""Source-independent checks of frozen survival policy boundaries."""

from __future__ import annotations

import copy
import inspect
import json

import numpy as np
import pytest

from trace_agent.consistency.survival_policies import (
    StrategySpec,
    adjusted_score,
    apply_survival_policies,
    combined_score,
    evaluate_survival_policies,
    select_survival_policies,
    spec_to_json,
    write_survival_policies,
)
from trace_agent.contracts.scores import ModuleScoreBatch
from trace_agent.contracts.task import SplitRole, TaskType
from trace_agent.data.split_table import DatasetSplits
from trace_agent.data.training_dataset import TrainingDataset, TrainingTargets


def _artifact(spec, *, base_cutoff=0.5):
    baseline = StrategySpec("baseline_risk_only", "baseline", "baseline", "p_base_only")
    return {
        "schema": "trace.survival_consistency_policies.v1",
        "schema_version": 1,
        "task": "longitudinal",
        "horizon": "5y",
        "selection_source": "validation_only",
        "endpoints": ["D1"],
        "kg_q99": float(spec.params["kg_q99"]),
        "base_cutoffs": [base_cutoff],
        "rules": [
            {
                "endpoint": "D1",
                "endpoint_index": 0,
                "selected_strategy": spec.name,
                "use_consistency_cutoff": True,
            }
        ],
        "all_strategy_specs": {baseline.name: spec_to_json(baseline), spec.name: spec_to_json(spec)},
        "finalists": [baseline.name, spec.name],
        "selector_candidates": [baseline.name, spec.name],
    }


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_pure_weighted_policy_is_independent_of_unused_missing_score(weight):
    spec = StrategySpec(
        "pure",
        "combined",
        "hard_combined",
        "kg_only" if weight else "med_only",
        combined_cutoff=0.5,
        params={"kg_q99": 1.0, "combined_mode": "weighted_sum", "w": weight},
    )
    kg = np.array([[0.8 if weight else np.nan]], dtype=np.float32)
    med = np.array([[np.nan if weight else 0.8]], dtype=np.float32)

    assert combined_score(kg, med, spec)[0, 0] == pytest.approx(0.8)
    _, selected, _ = apply_survival_policies(
        _artifact(spec),
        predictions=np.array([[0.7]]),
        valid_mask=np.array([[True]]),
        kg_scores=kg,
        evidence_scores=med,
    )
    assert selected.tolist() == [[True]]


def test_adjusted_predictions_can_exceed_baseline_and_mask_extreme_missing_risk():
    spec = StrategySpec(
        "adjusted",
        "soft",
        "adjusted",
        "kg_only",
        transform="logit_soft_reweight",
        adjusted_cutoffs=np.array([0.0]),
        params={"kg_q99": 3.0, "a": 4.0, "b": 0.0},
    )
    artifact = _artifact(spec)
    before = copy.deepcopy(artifact)
    risk = np.array([[0.4], [1.0], [0.0], [np.nan]])
    kg = np.array([[3.0], [0.0], [3.0], [3.0]])
    evidence = np.full_like(risk, np.nan)
    transformed = adjusted_score(risk, kg, evidence, spec)
    assert np.isfinite(transformed[:3]).all()
    baseline, selected, _ = apply_survival_policies(
        artifact,
        predictions=risk,
        valid_mask=np.array([[1], [1], [1], [0]], dtype=bool),
        kg_scores=kg,
        evidence_scores=evidence,
    )

    assert baseline.tolist() == [[False], [True], [False], [False]]
    assert selected.tolist() == [[True], [True], [False], [False]]
    assert artifact == before
    assert "labels" not in inspect.signature(apply_survival_policies).parameters
    json.dumps(artifact, allow_nan=False)


def test_frozen_global_kg_scale_and_evaluation_are_independent_of_chunk_distribution():
    spec = StrategySpec(
        "joint",
        "combined",
        "hard_combined",
        "kg_med",
        combined_cutoff=0.5,
        params={"kg_q99": 10.0, "combined_mode": "weighted_sum", "w": 0.8},
    )
    artifact = _artifact(spec)
    risk = np.full((4, 1), 0.7)
    kg = np.array([[1.0], [8.0], [2.0], [9.0]])
    med = np.ones((4, 1))
    valid = np.ones((4, 1), dtype=bool)
    _, selected, _ = apply_survival_policies(
        artifact,
        predictions=risk,
        valid_mask=valid,
        kg_scores=kg,
        evidence_scores=med,
    )
    chunked = []
    for row in range(4):
        _, block, _ = apply_survival_policies(
            artifact,
            predictions=risk[row : row + 1],
            valid_mask=valid[row : row + 1],
            kg_scores=kg[row : row + 1],
            evidence_scores=med[row : row + 1],
        )
        chunked.append(block)
    assert selected.tolist() == [[False], [True], [False], [True]]
    np.testing.assert_array_equal(np.concatenate(chunked), selected)
    kwargs = {
        "predictions": risk,
        "labels": np.array([[0], [1], [0], [1]]),
        "label_mask": valid,
        "kg_scores": kg,
        "evidence_scores": med,
    }
    assert evaluate_survival_policies(artifact, **kwargs, chunk_size=1) == evaluate_survival_policies(
        artifact, **kwargs, chunk_size=4
    )


@pytest.mark.parametrize("role", ["train", "test", "inference"])
def test_survival_search_rejects_nonvalidation_selection(role):
    with pytest.raises(ValueError, match="validation"):
        select_survival_policies(
            predictions=np.array([[0.9], [0.1]]),
            labels=np.array([[1], [0]]),
            label_mask=np.ones((2, 1), bool),
            kg_scores=np.ones((2, 1)),
            evidence_scores=np.ones((2, 1)),
            endpoints=("D1",),
            split_role=role,
            horizon="5y",
            threshold_points=2,
            joint_grid_points=2,
            max_finalists=4,
        )


def test_all_missing_validation_labels_freeze_null_cutoff_and_no_positive_masks():
    values = np.array([[0.7], [0.9]])
    artifact = select_survival_policies(
        predictions=values,
        labels=np.full((2, 1), np.nan),
        label_mask=np.zeros((2, 1), bool),
        kg_scores=np.ones((2, 1)),
        evidence_scores=np.ones((2, 1)),
        endpoints=("D1",),
        split_role="validation",
        horizon="5y",
        threshold_points=2,
        joint_grid_points=2,
        max_finalists=4,
    )
    assert artifact["base_cutoffs"] == [None]
    baseline, selected, _ = apply_survival_policies(
        artifact,
        predictions=values,
        valid_mask=np.ones((2, 1), bool),
        kg_scores=np.ones((2, 1)),
        evidence_scores=np.ones((2, 1)),
    )
    assert not baseline.any() and not selected.any()
    json.dumps(artifact, allow_nan=False)


def test_cli_label_free_chunked_apply_and_test_report_accept_adjusted_expansion(tmp_path, monkeypatch):
    from trace_agent.cli import consistency_workflow as workflow

    n = 4100  # Cross the CLI's actual 4096-row chunk boundary.
    spec = StrategySpec(
        "adjusted",
        "soft",
        "adjusted",
        "kg_only",
        transform="logit_soft_reweight",
        adjusted_cutoffs=np.array([0.0]),
        params={"kg_q99": 3.0, "a": 4.0, "b": 0.0},
    )
    artifact = _artifact(spec)
    artifact["model_identity"] = "fixed-model"
    write_survival_policies(tmp_path / "consistency_rules.json", artifact)
    eids = np.arange(n)
    inference = {
        "task": "longitudinal",
        "horizon": "5y",
        "endpoints": ("D1",),
        "model_identity": "fixed-model",
        "participant_ids": eids,
        "predictions": np.full((n, 1), 0.4),
        "valid_mask": np.ones((n, 1), bool),
        "kg": np.full((n, 1), 3.0),
        "evidence": np.full((n, 1), np.nan),
        "input_sha256": "a" * 64,
    }
    monkeypatch.setattr(workflow, "_read_inference", lambda config: inference)

    def forbidden(*args, **kwargs):
        raise AssertionError("application opened target/split data")

    monkeypatch.setattr(workflow, "dataset_from_config", forbidden)
    config = {"output": {"directory": str(tmp_path)}, "consistency": {"horizon": "5y"}}
    path = workflow.apply_consistency(config)
    with np.load(path) as saved:
        assert not saved["baseline"].any()
        assert saved["filtered"].all()
    rows = {
        SplitRole.TRAIN: np.array([0]),
        SplitRole.VALIDATION: np.array([1]),
        SplitRole.TEST: np.arange(2, n),
    }
    dataset = TrainingDataset(
        scores=ModuleScoreBatch(
            participant_ids=eids,
            modules=("M1",),
            endpoints=("D1",),
            scores=np.zeros((n, 1, 1)),
            pred_mask=np.ones((n, 1, 1), bool),
            availability=np.ones((n, 1), bool),
            task=TaskType.LONGITUDINAL,
        ),
        targets=TrainingTargets(
            labels=np.ones((n, 1)), label_mask=np.ones((n, 1), bool), event_time=np.full((n, 1), 3.0)
        ),
        splits=DatasetSplits(
            table_path=tmp_path / "splits.csv",
            row_indices=rows,
            eid_values={role: eids[indices].astype(str) for role, indices in rows.items()},
            file_sha256="b" * 64,
            canonical_sha256="c" * 64,
        ),
    )
    report = workflow.evaluate_frozen_consistency(config, dataset)
    assert report["participants"] == n - 2
    assert report["baseline"]["micro"]["tp"] == 0
    assert report["filtered"]["micro"]["tp"] == n - 2
    assert report["strategy_search"]["selector_summary"]["tp"] == n - 2
