from __future__ import annotations

import numpy as np
import pytest

from trace_agent.agents.prevalent import PrevalentAgent
from trace_agent.agents.trace_agent import TRACEAgent
from trace_agent.consistency.validator import ConsistencyValidator
from trace_agent.contracts.consistency import ConsistencyThresholds
from trace_agent.contracts.scores import ModuleScoreBatch, RiskProfile
from trace_agent.contracts.task import TaskType
from trace_agent.data.burden import BurdenProfile
from trace_agent.planning.candidate_ranker import KnowledgeCandidateRanker
from trace_agent.planning.engine import PlannerConfig, PlanningEngine


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


class PositiveAddQ:
    def predict(self, rows):
        matrix = np.asarray(rows)
        action = matrix[:, -4:]
        return np.where(np.any(action[:, :2] > 0, axis=1), 1.0, 0.0)


def _batch() -> ModuleScoreBatch:
    return ModuleScoreBatch(
        participant_ids=np.array([101]),
        modules=("basic", "blood"),
        endpoints=("endpoint_a", "endpoint_b"),
        scores=np.array([[[0.2, 0.3], [0.8, 0.9]]], dtype=np.float32),
        pred_mask=np.ones((1, 2, 2), dtype=bool),
        availability=np.ones((1, 2), dtype=bool),
        task=TaskType.PREVALENT,
    )


def test_trace_agent_returns_risk_path_burden_rationale_and_validation() -> None:
    burden = BurdenProfile(
        modules=("basic", "blood"),
        monetary=np.array([1.0, 2.0]),
        time=np.zeros(2),
    )
    planner = PlanningEngine(
        integrator=MeanIntegrator(),
        q_model=PositiveAddQ(),
        candidate_ranker=KnowledgeCandidateRanker(
            kg_matrix=np.ones((2, 2), dtype=np.float32),
            burden=burden.composite,
            cost_gamma=0.0,
        ),
        burden_profile=burden,
        config=PlannerConfig(max_steps=0, candidate_actions=2),
    )
    validator = ConsistencyValidator(
        kg_matrix=np.ones((2, 2), dtype=np.float32),
        thresholds=ConsistencyThresholds(
            prediction=0.5,
            kg=0.5,
            feature=0.5,
            evidence=0.5,
            combined=0.5,
        ),
    )
    agent = TRACEAgent(
        task_agents=[
            PrevalentAgent(
                planner=planner,
                consistency_validator=validator,
                decision_strategy="combined",
            )
        ]
    )

    result = agent.assess(_batch(), initial_used_modules=np.array([1, 0], dtype=bool))

    assert result.task is TaskType.PREVALENT
    assert result.selected_modules == ("basic", "blood")
    assert result.planning.accumulated_burden == 3.0
    assert result.prioritized_endpoints[0].endpoint == "endpoint_b"
    assert result.prioritized_endpoints[0].score == pytest.approx(0.6)
    assert result.rationale
    assert all(result.validation_checks.values())
    assert result.consistency is not None
