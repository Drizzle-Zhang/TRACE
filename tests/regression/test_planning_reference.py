from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trace_agent.consistency.feature import compute_feature_consistency
from trace_agent.contracts.scores import ModuleScoreBatch, RiskProfile
from trace_agent.contracts.task import TaskType
from trace_agent.data.burden import BurdenProfile
from trace_agent.integration.longitudinal_cox import LongitudinalRiskIntegrator
from trace_agent.planning.actions import build_action_features
from trace_agent.planning.candidate_ranker import KnowledgeCandidateRanker
from trace_agent.planning.engine import PlannerConfig, PlanningEngine
from trace_agent.planning.state import build_state_features

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = json.loads((ROOT / "tests/fixtures/planning_reference.json").read_text())


def test_reference_outputs_record_the_frozen_source_identity() -> None:
    """Reference source paths and hashes identify the expected planner outputs."""
    manifest = json.loads((ROOT / "tests/fixtures/reference_sources.json").read_text())
    recorded = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    for source in REFERENCE["sources"]:
        assert source["matches_frozen_manifest"]
        assert source["sha256"] == recorded[source["path"]]


@pytest.mark.parametrize("task", [TaskType.PREVALENT, TaskType.LONGITUDINAL])
def test_all_valid_state_and_action_features_match_frozen_outputs(task: TaskType) -> None:
    case = REFERENCE["feature_vectors"]
    values = np.asarray(case["values"], dtype=np.float32)[None, :]
    used = np.asarray(case["used_modules"], dtype=bool)
    profile = RiskProfile(
        participant_ids=np.array([1]),
        endpoints=("D1", "D2", "D3"),
        raw_scores=values,
        valid_mask=np.ones_like(values, dtype=bool),
        observed_modules=used[None, :],
        task=task,
        metadata={"logits" if task is TaskType.PREVALENT else "standardized_scores": values},
    )
    state = build_state_features(profile, used_modules=used, accumulated_burden=case["accumulated_burden"])
    np.testing.assert_allclose(state, case[task.value], rtol=1e-6, atol=1e-7)
    for action in case["actions"]:
        actual = build_action_features(action["module_index"], n_modules=3, burden=action["burden"])
        np.testing.assert_array_equal(actual, action["expected"])


@pytest.mark.parametrize("case", REFERENCE["candidate_cases"], ids=lambda case: case["name"])
def test_longitudinal_kg_candidate_order_matches_frozen_outputs(case: dict) -> None:
    task = TaskType.LONGITUDINAL
    signal = np.asarray(case["signal"], dtype=np.float32)[None, :]
    used = np.asarray(case["used_modules"], dtype=bool)
    profile = RiskProfile(
        participant_ids=np.array([1]),
        endpoints=("D1", "D2"),
        raw_scores=signal,
        valid_mask=np.asarray(case["valid_mask"], dtype=bool)[None, :],
        observed_modules=used[None, :],
        task=task,
        metadata={"logits" if task is TaskType.PREVALENT else "standardized_scores": signal},
    )
    ranker = KnowledgeCandidateRanker(
        kg_matrix=np.asarray(case["knowledge"]),
        burden=np.asarray(case["burden"]),
        urgency_threshold=0.8,
    )
    actual = ranker.rank(
        profile=profile,
        pred_mask_md=np.asarray(case["pred_mask"], dtype=bool),
        availability=np.ones(3, dtype=bool),
        used_modules=used,
        top_k=3,
    )
    assert actual.tolist() == case["expected_rank"]


@pytest.mark.parametrize("task", [TaskType.PREVALENT, TaskType.LONGITUDINAL])
@pytest.mark.parametrize("fallback", ["fully_covered", "negligible_gain"])
def test_cost_only_fallback_does_not_replace_prevalent_native_behavior(task, fallback) -> None:
    """Survival's frozen cost fallback is not prevalent's native KG fallback."""
    used = np.array([True, False, False])
    pred_mask = np.ones((3, 2), dtype=bool)
    knowledge = np.array([[1, 1], [100, 100], [1, 1]], dtype=np.float32)
    if fallback == "negligible_gain":
        pred_mask[0] = [True, False]
        knowledge *= 1e-15
    profile = RiskProfile(
        participant_ids=np.array([1]),
        endpoints=("D1", "D2"),
        raw_scores=np.zeros((1, 2), dtype=np.float32),
        valid_mask=np.ones((1, 2), dtype=bool),
        observed_modules=used[None, :],
        task=task,
    )
    ranker = KnowledgeCandidateRanker(kg_matrix=knowledge, burden=np.array([1, 4, 2]))
    actual = ranker.rank(
        profile=profile,
        pred_mask_md=pred_mask,
        availability=np.ones(3, dtype=bool),
        used_modules=used,
        top_k=2,
    )
    assert actual.tolist() == ([1, 2] if task is TaskType.PREVALENT else [2, 1])


class ZeroRiskFusion:
    """Controlled fusion returning zero log-risk for each endpoint."""

    def predict(self, scores, *, pred_mask_nmd, used_mask_nm, standardized):
        return np.zeros((scores.shape[0], scores.shape[2]), dtype=np.float32)

    def standardize(self, risk):
        return risk


class RecordedActionValueModel:
    def predict(self, rows):
        case = REFERENCE["trajectory"]
        action = np.asarray(rows)[:, -5:]
        return action[:, :3] @ np.asarray(case["q_add_values"]) + action[:, -1] * case["q_stop_value"]


def _plan(stop_value_mode: str):
    case = REFERENCE["candidate_cases"][0]
    mask = np.asarray(case["pred_mask"], dtype=bool)[None, :, :]
    batch = ModuleScoreBatch(
        participant_ids=np.array([1]),
        modules=("M1", "M2", "M3"),
        endpoints=("D1", "D2"),
        scores=np.where(mask, 0.0, np.nan),
        pred_mask=mask,
        availability=np.ones((1, 3), dtype=bool),
        task=TaskType.LONGITUDINAL,
    )
    burden = BurdenProfile(modules=batch.modules, monetary=np.asarray(case["burden"]), time=np.zeros(3))
    planner = PlanningEngine(
        integrator=LongitudinalRiskIntegrator(fusion=ZeroRiskFusion()),
        q_model=RecordedActionValueModel(),
        candidate_ranker=KnowledgeCandidateRanker(
            kg_matrix=np.asarray(case["knowledge"]), burden=burden.composite, urgency_threshold=0.8
        ),
        burden_profile=burden,
        config=PlannerConfig(max_steps=4, candidate_actions=1, stop_value_mode=stop_value_mode),
    )
    return planner.plan(batch, initial_used_modules=np.asarray(case["used_modules"], dtype=bool))


def test_add_then_model_stop_trajectory_matches_reference() -> None:
    expected = REFERENCE["trajectory"]
    result = _plan("model")

    assert [
        step.action.module_index if step.action.module_index is not None else -1 for step in result.trajectory
    ] == expected["expected_actions"]
    assert np.flatnonzero(result.final_used_modules).tolist() == expected["expected_used_modules"]
    assert result.accumulated_burden == expected["expected_burden"]
    np.testing.assert_allclose(result.final_profile.raw_scores[0], expected["expected_raw_risk"])
    assert result.trajectory[-1].metadata["stop_reason"] == "stop_value"


def test_fixed_zero_stop_follows_candidate_exhaustion() -> None:
    result = _plan("fixed_zero")

    assert [step.action.module_index for step in result.trajectory] == [2, 1, None]
    assert result.final_used_modules.all()
    assert result.trajectory[-1].metadata["stop_reason"] == "no_candidates"


@pytest.mark.parametrize("case", REFERENCE["mcs_cases"], ids=lambda case: case["name"])
def test_feature_consistency_matches_frozen_mcs_outputs(case: dict) -> None:
    result = compute_feature_consistency(
        contributions=np.asarray(case["contributions"], dtype=np.float32),
        medical_weights=np.asarray(case["medical_weights"], dtype=np.float32),
    )
    np.testing.assert_allclose(result.scores, case["expected_scores"], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result.raw_scores, case["expected_raw"], rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(result.matched_counts, case["expected_matched_counts"])
