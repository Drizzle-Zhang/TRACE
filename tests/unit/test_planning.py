from __future__ import annotations

import numpy as np
import pytest

from trace_agent.contracts.scores import ModuleScoreBatch, RiskProfile
from trace_agent.contracts.task import TaskType
from trace_agent.data.burden import BurdenProfile
from trace_agent.planning.candidate_ranker import KnowledgeCandidateRanker
from trace_agent.planning.engine import PlannerConfig, PlanningEngine
from trace_agent.planning.rewards.longitudinal import (
    build_horizon_labels,
    multi_horizon_bce_gain,
)
from trace_agent.planning.rewards.prevalent import prevalent_bce_gain


class MeanIntegrator:
    def integrate(self, batch, observed_modules):
        observed = np.asarray(observed_modules, dtype=bool)
        active = batch.pred_mask & observed[:, :, None]
        scores = np.nanmean(np.where(active, batch.scores, np.nan), axis=1)
        return RiskProfile(
            participant_ids=batch.participant_ids,
            endpoints=batch.endpoints,
            raw_scores=scores,
            calibrated_scores=scores,
            valid_mask=np.any(active, axis=1),
            observed_modules=observed,
            task=batch.task,
        )


class AddPositiveQ:
    def predict(self, rows):
        matrix = np.asarray(rows)
        # The final four columns are [module_0, module_1, burden, is_stop].
        action_features = matrix[:, -4:]
        return np.where(np.sum(action_features[:, :2], axis=1) > 0, 1.0, 0.0)


def _planning_batch() -> ModuleScoreBatch:
    return ModuleScoreBatch(
        participant_ids=np.array([7]),
        modules=("M1", "M2"),
        endpoints=("D1",),
        scores=np.array([[[0.2], [0.8]]], dtype=np.float32),
        pred_mask=np.ones((1, 2, 1), dtype=bool),
        availability=np.ones((1, 2), dtype=bool),
        task=TaskType.PREVALENT,
    )


def test_knowledge_ranker_excludes_used_modules_and_adjusts_for_burden() -> None:
    profile = RiskProfile(
        participant_ids=np.array([7]),
        endpoints=("D1",),
        raw_scores=np.array([[0.5]], dtype=np.float32),
        valid_mask=np.ones((1, 1), dtype=bool),
        observed_modules=np.array([[1, 0, 0]], dtype=bool),
        task=TaskType.PREVALENT,
    )
    ranker = KnowledgeCandidateRanker(
        kg_matrix=np.array([[1.0], [0.9], [0.8]], dtype=np.float32),
        burden=np.array([1.0, 9.0, 2.0], dtype=np.float32),
        cost_gamma=1.0,
    )

    ranked = ranker.rank(
        profile=profile,
        pred_mask_md=np.ones((3, 1), dtype=bool),
        availability=np.ones(3, dtype=bool),
        used_modules=np.array([1, 0, 0], dtype=bool),
        top_k=2,
    )

    assert ranked.tolist() == [2, 1]


def test_planning_engine_adds_module_then_stops_with_complete_trajectory() -> None:
    ranker = KnowledgeCandidateRanker(
        kg_matrix=np.ones((2, 1), dtype=np.float32),
        burden=np.array([1.0, 2.0], dtype=np.float32),
        cost_gamma=0.0,
    )
    engine = PlanningEngine(
        integrator=MeanIntegrator(),
        q_model=AddPositiveQ(),
        candidate_ranker=ranker,
        burden_profile=BurdenProfile(
            modules=("M1", "M2"),
            monetary=np.array([1.0, 2.0]),
            time=np.zeros(2),
        ),
        config=PlannerConfig(max_steps=0, candidate_actions=2, stop_value_mode="fixed_zero"),
    )

    result = engine.plan(_planning_batch(), initial_used_modules=np.array([1, 0], dtype=bool))

    assert result.final_used_modules.tolist() == [True, True]
    assert result.accumulated_burden == pytest.approx(3.0)
    assert [step.action.kind.value for step in result.trajectory] == ["add", "stop"]
    assert result.trajectory[0].action.module_index == 1
    np.testing.assert_allclose(result.final_profile.decision_scores, [[0.5]])


def test_max_steps_caps_additions_but_still_records_terminal_stop() -> None:
    burden = BurdenProfile(
        modules=("M1", "M2"),
        monetary=np.array([1.0, 2.0]),
        time=np.zeros(2),
    )
    engine = PlanningEngine(
        integrator=MeanIntegrator(),
        q_model=AddPositiveQ(),
        candidate_ranker=KnowledgeCandidateRanker(
            kg_matrix=np.ones((2, 1), dtype=np.float32),
            burden=burden.composite,
            cost_gamma=0.0,
        ),
        burden_profile=burden,
        config=PlannerConfig(max_steps=1, candidate_actions=2),
    )

    result = engine.plan(
        _planning_batch(),
        initial_used_modules=np.array([1, 0], dtype=bool),
    )

    assert [step.action.kind.value for step in result.trajectory] == [
        "add",
        "stop",
    ]
    assert result.trajectory[-1].metadata["stop_reason"] == "max_steps"


def test_prevalent_and_longitudinal_rewards_measure_loss_reduction() -> None:
    gain = prevalent_bce_gain(
        current_probability=np.array([0.4, 0.6]),
        updated_probability=np.array([0.8, 0.2]),
        labels=np.array([1.0, 0.0]),
        label_mask=np.array([1, 1], dtype=bool),
    )
    assert gain > 0

    labels, known = build_horizon_labels(
        event=np.array([1, 0, 0]),
        time=np.array([5.0, 12.0, 6.0]),
        label_mask=np.ones(3, dtype=bool),
        horizons=np.array([10.0]),
    )
    assert labels[:, 0].tolist() == [1.0, 0.0, 0.0]
    assert known[:, 0].tolist() == [True, True, False]

    longitudinal_gain = multi_horizon_bce_gain(
        current_logits=np.array([[0.0], [0.0], [0.0]]),
        updated_logits=np.array([[2.0], [-2.0], [0.0]]),
        labels=labels,
        known_mask=known,
        horizon_weights=np.array([1.0]),
    )
    assert longitudinal_gain > 0
