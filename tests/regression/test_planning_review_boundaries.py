from __future__ import annotations

import json

import numpy as np
import pytest

from trace_agent.agents.task_agent import TRACETaskAgent
from trace_agent.contracts.scores import ModuleScoreBatch, RiskProfile
from trace_agent.contracts.task import SplitRole, TaskType
from trace_agent.data.burden import BurdenProfile
from trace_agent.data.training_dataset import TrainingTargets, TrainingView
from trace_agent.integration.longitudinal_cox import LongitudinalRiskIntegrator
from trace_agent.planning.candidate_ranker import KnowledgeCandidateRanker
from trace_agent.planning.engine import PlannerConfig, PlanningEngine
from trace_agent.planning.rewards.longitudinal import build_horizon_labels, multi_horizon_bce_gain
from trace_agent.planning.state import build_state_features
from trace_agent.runtime.artifacts import build_trace_agent
from trace_agent.runtime.model_artifact import TraceModelArtifact, dump_trace_artifact
from trace_agent.runtime.serialization import agent_result_to_dict
from trace_agent.training.planner import PlannerTrainingConfig, build_planner_examples


@pytest.mark.parametrize("task", [TaskType.PREVALENT, TaskType.LONGITUDINAL])
def test_state_construction_preserves_unobserved_prior_signal(task: TaskType) -> None:
    signal = np.array([[0.0, 10.0]], dtype=np.float32)
    key = "logits" if task is TaskType.PREVALENT else "standardized_scores"
    profile = RiskProfile(
        participant_ids=np.array([1]),
        endpoints=("D1", "D2"),
        raw_scores=signal.copy(),
        valid_mask=np.array([[True, False]]),
        observed_modules=np.array([[True, False]]),
        task=task,
        metadata={key: signal},
    )

    state = build_state_features(profile, used_modules=np.array([True, False]), accumulated_burden=1.0)

    assert state[1] == 0.0
    np.testing.assert_array_equal(profile.metadata[key], [[0.0, 10.0]])


class PriorRiskFusion:
    def predict(self, scores, *, pred_mask_nmd, used_mask_nm, standardized):
        return np.repeat(np.array([[0.0, 10.0]], dtype=np.float32), scores.shape[0], axis=0)

    def standardize(self, risk):
        return risk


class PositiveAddQ:
    def predict(self, rows):
        return np.ones(np.asarray(rows).shape[0], dtype=np.float32)


class NoGainReward:
    def gain(self, *, current, updated, targets, row):
        return 0.0


def test_training_and_online_candidates_match_for_partially_observed_prior() -> None:
    mask = np.array([[[True, False], [False, False], [True, False], [False, True]]])
    batch = ModuleScoreBatch(
        participant_ids=np.array([1]),
        modules=("baseline", "empty", "D1_module", "D2_module"),
        endpoints=("D1", "D2"),
        scores=np.where(mask, 0.0, np.nan),
        pred_mask=mask,
        availability=np.ones((1, 4), dtype=bool),
        task=TaskType.LONGITUDINAL,
    )
    burden = BurdenProfile(modules=batch.modules, monetary=np.ones(4), time=np.zeros(4))
    ranker = KnowledgeCandidateRanker(kg_matrix=np.ones((4, 2)), burden=burden.composite)
    integrator = LongitudinalRiskIntegrator(fusion=PriorRiskFusion())
    view = TrainingView(
        role=SplitRole.TRAIN,
        row_indices=np.array([0]),
        scores=batch,
        targets=TrainingTargets(labels=np.zeros((1, 2)), label_mask=np.ones((1, 2), dtype=bool)),
    )
    # Seed 78 samples baseline + empty, so D1 is partially covered and D2
    # still has its finite prior signal. Both unused modules remain eligible.
    examples = build_planner_examples(
        view,
        integrator=integrator,
        burden=burden,
        ranker=ranker,
        reward=NoGainReward(),
        config=PlannerTrainingConfig(seed=78, states_per_participant=1, candidate_actions=1),
    )
    initial = np.array([True, True, False, False])
    result = PlanningEngine(
        integrator=integrator,
        q_model=PositiveAddQ(),
        candidate_ranker=ranker,
        burden_profile=burden,
        config=PlannerConfig(max_steps=1, candidate_actions=1),
    ).plan(batch, initial_used_modules=initial)

    np.testing.assert_array_equal(examples.features[0, -12:-8], initial)
    training_action = int(np.argmax(examples.features[0, -6:-2]))
    assert training_action == result.trajectory[0].action.module_index == 2


def test_masked_nan_horizon_cells_do_not_pollute_observed_gain() -> None:
    labels, known = build_horizon_labels(
        event=np.array([1, 0]),
        time=np.array([2.0, 1.0]),
        label_mask=np.array([True, False]),
        horizons=np.array([5.0]),
    )
    gain = multi_horizon_bce_gain(
        current_logits=np.array([[0.0], [np.nan]]),
        updated_logits=np.array([[2.0], [np.nan]]),
        labels=labels,
        known_mask=known,
        horizon_weights=np.ones(1),
    )

    assert gain == pytest.approx(np.log(2.0) - np.logaddexp(0.0, -2.0))


def test_no_observed_horizon_cells_have_zero_gain() -> None:
    gain = multi_horizon_bce_gain(
        current_logits=np.array([[np.nan]]),
        updated_logits=np.array([[np.nan]]),
        labels=np.zeros((1, 1)),
        known_mask=np.zeros((1, 1), dtype=bool),
        horizon_weights=np.ones(1),
    )

    assert gain == 0.0


class FiniteRiskIntegrator:
    def integrate(self, batch, observed_modules):
        return RiskProfile(
            participant_ids=batch.participant_ids,
            endpoints=batch.endpoints,
            raw_scores=np.zeros((1, 1), dtype=np.float32),
            valid_mask=np.any(batch.pred_mask & observed_modules[:, :, None], axis=1),
            observed_modules=observed_modules,
            task=batch.task,
        )


class ConstantQ:
    def __init__(self, value):
        self.value = value

    def predict(self, rows):
        return np.full(len(rows), self.value, dtype=np.float32)


@pytest.mark.parametrize("task", [TaskType.PREVALENT, TaskType.LONGITUDINAL])
@pytest.mark.parametrize("mode", ["fixed_zero", "model"])
def test_positive_model_stop_ties_preserve_task_action_order(task, mode) -> None:
    batch = ModuleScoreBatch(
        participant_ids=np.array([1]),
        modules=("M1",),
        endpoints=("D1",),
        scores=np.zeros((1, 1, 1), dtype=np.float32),
        pred_mask=np.ones((1, 1, 1), dtype=bool),
        availability=np.ones((1, 1), dtype=bool),
        task=task,
    )
    burden = BurdenProfile(modules=batch.modules, monetary=np.ones(1), time=np.zeros(1))
    result = PlanningEngine(
        integrator=FiniteRiskIntegrator(),
        q_model=ConstantQ(0.5),
        candidate_ranker=KnowledgeCandidateRanker(kg_matrix=np.ones((1, 1)), burden=burden.composite),
        burden_profile=burden,
        config=PlannerConfig(max_steps=1, candidate_actions=1, stop_value_mode=mode),
    ).plan(batch, initial_used_modules=np.zeros(1, dtype=bool))

    # Original prevalent puts STOP first; survival puts candidate ADDs first.
    expected_add = mode == "fixed_zero" or task is TaskType.LONGITUDINAL
    assert bool(result.final_used_modules[0]) is expected_add
    assert result.trajectory[-1].action.kind.value == "stop"


class ConstantFusion:
    def fuse(self, scores, used, pred_mask_md=None):
        return np.full(scores.shape[1], 0.5, dtype=np.float32)

    def fuse_logits(self, scores, used, pred_mask_md=None):
        return np.zeros(scores.shape[1], dtype=np.float32)

    def predict(self, scores, *, pred_mask_nmd, used_mask_nm, standardized):
        return np.zeros((scores.shape[0], scores.shape[2]), dtype=np.float32)

    def standardize(self, risk):
        return risk


class PreferModelStopQ:
    def predict(self, rows):
        return 1.0 + np.asarray(rows)[:, -1]


def _runtime_fixture(tmp_path, *, task, trained_mode=None, configured_mode=None):
    batch = ModuleScoreBatch(
        participant_ids=np.array([1]),
        modules=("M1",),
        endpoints=("D1",),
        scores=np.full((1, 1, 1), 0.5, dtype=np.float32),
        pred_mask=np.ones((1, 1, 1), dtype=bool),
        availability=np.ones((1, 1), dtype=bool),
        task=task,
    )
    for name, component, model in (
        ("fusion", "integration", ConstantFusion()),
        ("q", "planner", PreferModelStopQ()),
    ):
        training = (
            {"settings": {"stop_value_mode": trained_mode}}
            if component == "planner" and trained_mode is not None
            else {}
        )
        dump_trace_artifact(
            TraceModelArtifact(
                component=component,
                task=task,
                model=model,
                modules=batch.modules,
                endpoints=batch.endpoints,
                training=training,
            ),
            tmp_path / f"{name}.joblib",
        )
    np.save(tmp_path / "kg.npy", np.ones((1, 1), dtype=np.float32))
    (tmp_path / "burden.yaml").write_text("M1: 1.0\n", encoding="utf-8")
    config = {
        "task": task.value,
        "artifacts": {
            "fusion_model": str(tmp_path / "fusion.joblib"),
            "q_model": str(tmp_path / "q.joblib"),
            "knowledge_matrix": str(tmp_path / "kg.npy"),
            "burden_profile": str(tmp_path / "burden.yaml"),
        },
        "planner": {} if configured_mode is None else {"stop_value_mode": configured_mode},
    }
    return batch, config


@pytest.mark.parametrize("task", [TaskType.PREVALENT, TaskType.LONGITUDINAL])
def test_runtime_inherits_model_stop_mode_from_trained_artifact(tmp_path, task) -> None:
    batch, config = _runtime_fixture(tmp_path, task=task, trained_mode="model")
    result = build_trace_agent(config, batch=batch).assess(
        batch, initial_used_modules=np.zeros(1, dtype=bool)
    )

    assert result.selected_modules == ()
    assert result.planning.trajectory[-1].metadata["stop_reason"] == "stop_value"


@pytest.mark.parametrize("trained,configured", [("model", "fixed_zero"), ("fixed_zero", "model")])
def test_runtime_rejects_stop_mode_that_conflicts_with_training(tmp_path, trained, configured) -> None:
    batch, config = _runtime_fixture(
        tmp_path, task=TaskType.PREVALENT, trained_mode=trained, configured_mode=configured
    )

    with pytest.raises(ValueError, match="stop_value_mode.*training|training.*stop_value_mode"):
        build_trace_agent(config, batch=batch)


@pytest.mark.parametrize("mode", [None, "fixed_zero", "model"])
def test_artifact_without_stop_training_settings_keeps_runtime_mode(tmp_path, mode) -> None:
    batch, config = _runtime_fixture(tmp_path, task=TaskType.PREVALENT, configured_mode=mode)
    result = build_trace_agent(config, batch=batch).assess(
        batch, initial_used_modules=np.zeros(1, dtype=bool)
    )

    assert result.selected_modules == (() if mode == "model" else ("M1",))


class MeanHorizonIntegrator:
    def integrate(self, batch, observed_modules):
        active = batch.pred_mask & observed_modules[:, :, None]
        total = np.sum(np.where(active, batch.scores, 0.0), axis=1)
        counts = np.sum(active, axis=1)
        risk = np.full(total.shape, np.nan, dtype=np.float32)
        np.divide(total, counts, out=risk, where=counts > 0)
        return RiskProfile(
            participant_ids=batch.participant_ids,
            endpoints=batch.endpoints,
            raw_scores=risk,
            valid_mask=counts > 0,
            observed_modules=observed_modules,
            task=batch.task,
            horizon_risks={"5y": risk / 2.0},
        )


def test_exported_trajectory_preserves_full_risk_axis_and_action_state() -> None:
    mask = np.array([[[True, False], [False, True]]])
    batch = ModuleScoreBatch(
        participant_ids=np.array([1]),
        modules=("M1", "M2"),
        endpoints=("D1", "D2"),
        scores=np.array([[[0.2, np.nan], [np.nan, 0.8]]], dtype=np.float32),
        pred_mask=mask,
        availability=np.ones((1, 2), dtype=bool),
        task=TaskType.LONGITUDINAL,
    )
    burden = BurdenProfile(modules=batch.modules, monetary=np.array([1.0, 2.0]), time=np.zeros(2))
    planner = PlanningEngine(
        integrator=MeanHorizonIntegrator(),
        q_model=PositiveAddQ(),
        candidate_ranker=KnowledgeCandidateRanker(kg_matrix=np.ones((2, 2)), burden=burden.composite),
        burden_profile=burden,
        config=PlannerConfig(max_steps=1, candidate_actions=1),
    )
    result = TRACETaskAgent(task=batch.task, planner=planner).assess(
        batch, initial_used_modules=np.array([True, False])
    )
    exported = agent_result_to_dict(result, top_endpoints=1)

    assert len(exported["endpoints"]) == 1
    before_add = exported["trajectory"][0]["risk_profile"]
    assert before_add["endpoint_order"] == ["D1", "D2"]
    assert before_add["raw_scores"][0] == pytest.approx(0.2)
    assert before_add["raw_scores"][1] is None
    assert before_add["decision_scores"] == before_add["raw_scores"]
    assert before_add["valid_mask"] == [True, False]
    assert before_add["observed_modules"] == [True, False]
    assert before_add["horizon_risks"]["5y"][0] == pytest.approx(0.1)
    assert before_add["horizon_risks"]["5y"][1] is None
    for step in exported["trajectory"]:
        state_burden = burden.accumulated(np.array(step["risk_profile"]["observed_modules"]))
        action_burden = burden.composite[step["module_index"]] if step["action"] == "add" else 0.0
        assert step["accumulated_burden"] == pytest.approx(state_burden + action_burden)
    stopped = exported["trajectory"][-1]["risk_profile"]
    assert stopped["observed_modules"] == [True, True]
    assert stopped["raw_scores"] == pytest.approx([0.2, 0.8])
    assert stopped["horizon_risks"]["5y"] == pytest.approx([0.1, 0.4])
    json.dumps(exported, allow_nan=False)
