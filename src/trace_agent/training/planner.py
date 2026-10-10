from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from lightgbm import Booster, LGBMRegressor

from trace_agent.contracts.planning import ActionKind
from trace_agent.contracts.scores import ModuleScoreBatch, RiskProfile
from trace_agent.contracts.task import SplitRole, TaskType
from trace_agent.data.arrays import select_participant
from trace_agent.data.burden import BurdenProfile
from trace_agent.data.training_dataset import TrainingTargets, TrainingView
from trace_agent.planning.actions import build_action_features
from trace_agent.planning.candidate_ranker import KnowledgeCandidateRanker
from trace_agent.planning.engine import PlannerConfig, PlanningEngine
from trace_agent.planning.q_model import QModel
from trace_agent.planning.rewards.longitudinal import build_horizon_labels
from trace_agent.planning.state import build_state_features
from trace_agent.training.common import masked_classification_metrics, require_view_role


class RiskIntegrator(Protocol):
    def integrate(
        self,
        batch: ModuleScoreBatch,
        observed_modules: np.ndarray,
    ) -> RiskProfile: ...


class PlannerReward(Protocol):
    def gain(
        self,
        *,
        current: RiskProfile,
        updated: RiskProfile,
        targets: TrainingTargets,
        row: int,
    ) -> float: ...


@dataclass(frozen=True)
class PlannerTrainingConfig:
    seed: int = 7
    states_per_participant: int = 2
    candidate_actions: int = 6
    lambda_cost: float = 0.01
    stop_value_mode: str = "fixed_zero"
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 20
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if self.states_per_participant < 1:
            raise ValueError("states_per_participant must be positive")
        if self.candidate_actions < 1:
            raise ValueError("candidate_actions must be positive")
        if not np.isfinite(self.lambda_cost) or self.lambda_cost < 0:
            raise ValueError("lambda_cost must be finite and non-negative")
        if self.stop_value_mode not in {"fixed_zero", "model"}:
            raise ValueError("stop_value_mode must be 'fixed_zero' or 'model'")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.num_leaves < 2:
            raise ValueError("num_leaves must be at least two")
        if self.min_child_samples < 1:
            raise ValueError("min_child_samples must be positive")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> PlannerTrainingConfig:
        allowed = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(values).difference(allowed))
        if unknown:
            raise ValueError(f"unknown planner training settings: {unknown}")
        return cls(**dict(values))


@dataclass(frozen=True)
class PlannerExamples:
    features: np.ndarray
    targets: np.ndarray
    participant_rows: np.ndarray

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        targets = np.asarray(self.targets, dtype=np.float32).reshape(-1)
        participant_rows = np.asarray(
            self.participant_rows,
            dtype=np.int64,
        ).reshape(-1)
        if features.ndim != 2:
            raise ValueError("planner features must be two-dimensional")
        if features.shape[0] != targets.size or targets.size != participant_rows.size:
            raise ValueError("planner features, targets, and participant rows must align")
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError("planner examples must contain only finite values")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "participant_rows", participant_rows)


@dataclass
class ActionValueModel:
    estimator: Any
    feature_width: int

    def predict(self, rows: np.ndarray) -> np.ndarray:
        values = np.asarray(rows, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.feature_width:
            raise ValueError(f"planner features must have width {self.feature_width}")
        booster = getattr(self.estimator, "booster_", None)
        prediction_settings = (
            {"num_threads": booster.params.get("num_threads") or 1} if isinstance(booster, Booster) else {}
        )
        predicted = (
            booster.predict(values, **prediction_settings)
            if booster is not None
            else self.estimator.predict(values)
        )
        result = np.asarray(predicted, dtype=np.float32).reshape(-1)
        if result.shape != (values.shape[0],) or not np.isfinite(result).all():
            raise ValueError("planner estimator returned invalid predictions")
        return result


@dataclass(frozen=True)
class PlannerTrainingResult:
    model: ActionValueModel
    metrics: Mapping[str, Any]
    example_counts: Mapping[str, int]


def _sample_state(
    availability: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    candidates = np.flatnonzero(np.asarray(availability, dtype=bool))
    used = np.zeros(np.asarray(availability).shape, dtype=bool)
    if candidates.size <= 1:
        return used
    count = int(rng.integers(0, candidates.size))
    if count:
        used[rng.choice(candidates, size=count, replace=False)] = True
    return used


def _build_planner_examples(
    view: TrainingView,
    *,
    integrator: RiskIntegrator,
    burden: BurdenProfile,
    ranker: KnowledgeCandidateRanker,
    reward: PlannerReward,
    config: PlannerTrainingConfig,
) -> PlannerExamples:
    if view.scores.modules != burden.modules:
        raise ValueError("burden modules must match the training score bundle")
    rng = np.random.default_rng(config.seed)
    maximum_burden = float(np.max(burden.composite)) if burden.composite.size else 0.0
    feature_rows: list[np.ndarray] = []
    target_values: list[float] = []
    participant_rows: list[int] = []

    for row in range(view.scores.n_participants):
        if not view.targets.label_mask[row].any():
            continue
        participant = select_participant(view.scores, row)
        for _ in range(config.states_per_participant):
            used = _sample_state(
                participant.availability[0],
                rng=rng,
            )
            current = integrator.integrate(participant, used[None, :])
            accumulated = burden.accumulated(used)
            state = build_state_features(
                current,
                used_modules=used,
                accumulated_burden=accumulated,
            )
            candidates = ranker.rank(
                profile=current,
                pred_mask_md=participant.pred_mask[0],
                availability=participant.availability[0],
                used_modules=used,
                top_k=config.candidate_actions,
            )
            for module_index in candidates.tolist():
                updated_used = used.copy()
                updated_used[module_index] = True
                updated = integrator.integrate(
                    participant,
                    updated_used[None, :],
                )
                raw_gain = float(
                    reward.gain(
                        current=current,
                        updated=updated,
                        targets=view.targets,
                        row=row,
                    )
                )
                normalized_burden = (
                    float(burden.composite[module_index]) / maximum_burden if maximum_burden > 0 else 0.0
                )
                target = raw_gain - config.lambda_cost * normalized_burden
                feature_rows.append(
                    np.concatenate(
                        [
                            state,
                            build_action_features(
                                module_index,
                                n_modules=len(view.scores.modules),
                                burden=float(burden.composite[module_index]),
                            ),
                        ]
                    ).astype(np.float32, copy=False)
                )
                target_values.append(float(target))
                participant_rows.append(int(view.row_indices[row]))
            if config.stop_value_mode == "model":
                feature_rows.append(
                    np.concatenate(
                        [
                            state,
                            build_action_features(
                                None,
                                n_modules=len(view.scores.modules),
                                burden=0.0,
                            ),
                        ]
                    ).astype(np.float32, copy=False)
                )
                target_values.append(0.0)
                participant_rows.append(int(view.row_indices[row]))

    if not feature_rows:
        raise ValueError("planner example construction produced no candidate actions")
    return PlannerExamples(
        features=np.vstack(feature_rows),
        targets=np.asarray(target_values, dtype=np.float32),
        participant_rows=np.asarray(participant_rows, dtype=np.int64),
    )


def build_planner_examples(
    view: TrainingView,
    *,
    integrator: RiskIntegrator,
    burden: BurdenProfile,
    ranker: KnowledgeCandidateRanker,
    reward: PlannerReward,
    config: PlannerTrainingConfig,
) -> PlannerExamples:
    require_view_role(view, SplitRole.TRAIN)
    return _build_planner_examples(
        view,
        integrator=integrator,
        burden=burden,
        ranker=ranker,
        reward=reward,
        config=config,
    )


def build_validation_planner_examples(
    view: TrainingView,
    *,
    integrator: RiskIntegrator,
    burden: BurdenProfile,
    ranker: KnowledgeCandidateRanker,
    reward: PlannerReward,
    config: PlannerTrainingConfig,
) -> PlannerExamples:
    require_view_role(view, SplitRole.VALIDATION)
    return _build_planner_examples(
        view,
        integrator=integrator,
        burden=burden,
        ranker=ranker,
        reward=reward,
        config=config,
    )


def _regression_metrics(
    model: ActionValueModel,
    examples: PlannerExamples,
) -> dict[str, float]:
    prediction = model.predict(examples.features)
    residual = prediction - examples.targets
    return {
        "mean_squared_error": float(np.mean(np.square(residual))),
        "mean_absolute_error": float(np.mean(np.abs(residual))),
    }


def evaluate_validation_trajectories(
    validation: TrainingView,
    *,
    model: QModel,
    integrator: RiskIntegrator,
    burden: BurdenProfile,
    ranker: KnowledgeCandidateRanker,
    config: PlannerTrainingConfig,
) -> dict[str, Any]:
    """Evaluate frozen Q-model rollouts using the native empty initial state.

    These rollouts use training candidate/STOP settings and the engine's default
    threshold and step limit. They do not select or refit the model, or claim to
    reproduce a separately configured inference policy.
    """
    require_view_role(validation, SplitRole.VALIDATION)
    if validation.scores.n_participants == 0:
        raise ValueError("validation trajectories require at least one participant")
    engine_config = PlannerConfig(
        candidate_actions=config.candidate_actions,
        stop_value_mode=config.stop_value_mode,
    )
    engine = PlanningEngine(
        integrator=integrator,
        q_model=model,
        candidate_ranker=ranker,
        burden_profile=burden,
        config=engine_config,
    )
    participant_metrics: list[dict[str, Any]] = []
    final_profiles: list[RiskProfile] = []
    stop_actions = 0
    total_actions = 0
    for row in range(validation.scores.n_participants):
        result = engine.plan(
            select_participant(validation.scores, row),
            initial_used_modules=np.zeros(len(validation.scores.modules), dtype=bool),
        )
        final_profiles.append(result.final_profile)
        additions = [
            step.action.module_name for step in result.trajectory if step.action.kind is ActionKind.ADD
        ]
        stops = [step for step in result.trajectory if step.action.kind is ActionKind.STOP]
        stop_actions += len(stops)
        total_actions += len(result.trajectory)
        participant_id = result.participant_id
        if isinstance(participant_id, np.generic):
            participant_id = participant_id.item()
        participant_metrics.append(
            {
                "participant_id": participant_id,
                "row_index": int(validation.row_indices[row]),
                "final_burden": float(result.accumulated_burden),
                "added_modules": additions,
                "n_added_modules": len(additions),
                "steps": len(result.trajectory),
                "stopped": bool(stops),
                "stop_reason": stops[-1].metadata.get("stop_reason") if stops else None,
            }
        )

    valid = np.concatenate([profile.valid_mask for profile in final_profiles], axis=0)
    final_risk: dict[str, Any] = {"endpoint_order": list(validation.scores.endpoints)}
    if validation.scores.task is TaskType.PREVALENT:
        probability = np.concatenate([profile.decision_scores for profile in final_profiles], axis=0)
        final_risk["classification"] = masked_classification_metrics(
            probability,
            validation.targets.labels,
            validation.targets.label_mask & valid,
        )
    else:
        if validation.targets.event_time is None:
            raise ValueError("longitudinal validation trajectories require event_time")
        horizon_names = tuple(final_profiles[0].horizon_risks)
        if any(tuple(profile.horizon_risks) != horizon_names for profile in final_profiles):
            raise ValueError("validation trajectory horizon order must remain consistent")
        horizons: dict[str, Any] = {}
        for name in horizon_names:
            probability = np.concatenate([profile.horizon_risks[name] for profile in final_profiles], axis=0)
            labels, known = build_horizon_labels(
                event=validation.targets.labels,
                time=validation.targets.event_time,
                label_mask=validation.targets.label_mask,
                horizons=np.asarray([float(name.removesuffix("y"))]),
            )
            horizons[name] = masked_classification_metrics(
                probability,
                labels.reshape(validation.targets.labels.shape),
                known.reshape(validation.targets.label_mask.shape) & valid,
            )
        final_risk["horizons"] = horizons
    return {
        "selection_source": "validation_only",
        "rollout_config": {"initial_modules": [], **dataclasses.asdict(engine_config)},
        "participants": participant_metrics,
        "summary": {
            "n_participants": len(participant_metrics),
            "mean_final_burden": float(np.mean([row["final_burden"] for row in participant_metrics])),
            "mean_added_modules": float(np.mean([row["n_added_modules"] for row in participant_metrics])),
            "mean_steps": float(np.mean([row["steps"] for row in participant_metrics])),
            "stop_action_fraction": float(stop_actions / total_actions) if total_actions else 0.0,
            "participant_stop_rate": float(np.mean([row["stopped"] for row in participant_metrics])),
        },
        "final_risk": final_risk,
    }


def fit_action_value_model(
    train_examples: PlannerExamples,
    validation_examples: PlannerExamples,
    *,
    config: PlannerTrainingConfig,
) -> PlannerTrainingResult:
    if train_examples.features.shape[1] != validation_examples.features.shape[1]:
        raise ValueError("train and validation planner feature widths must match")
    estimator = LGBMRegressor(
        objective="regression",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples,
        random_state=config.seed,
        n_jobs=config.n_jobs,
        verbosity=-1,
    )
    estimator.fit(train_examples.features, train_examples.targets)
    model = ActionValueModel(
        estimator=estimator,
        feature_width=int(train_examples.features.shape[1]),
    )
    return PlannerTrainingResult(
        model=model,
        metrics=_regression_metrics(model, validation_examples),
        example_counts={
            "train": int(train_examples.targets.size),
            "validation": int(validation_examples.targets.size),
        },
    )
