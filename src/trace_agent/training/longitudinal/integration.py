from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from trace_agent.contracts.task import SplitRole, TaskType
from trace_agent.data.training_dataset import TrainingView
from trace_agent.integration.absolute_risk import CoxAbsoluteRiskMapper
from trace_agent.planning.rewards.longitudinal import build_horizon_labels
from trace_agent.training.common import (
    masked_classification_metrics,
    require_view_role,
    sample_observed_masks,
)


def _horizon_name(value: float) -> str:
    return f"{int(value)}y" if float(value).is_integer() else f"{value:g}y"


@dataclass(frozen=True)
class LongitudinalIntegrationConfig:
    seed: int = 7
    subsets_per_participant: int = 4
    min_events: int = 5
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 6
    n_jobs: int = 1
    early_stopping_rounds: int = 0
    horizons: tuple[float, ...] = (5.0, 10.0)

    def __post_init__(self) -> None:
        horizons = tuple(float(value) for value in self.horizons)
        if self.subsets_per_participant < 0:
            raise ValueError("subsets_per_participant cannot be negative")
        if self.min_events < 1:
            raise ValueError("min_events must be positive")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if (
            isinstance(self.early_stopping_rounds, bool)
            or not isinstance(self.early_stopping_rounds, (int, np.integer))
            or self.early_stopping_rounds < 0
        ):
            raise ValueError("early_stopping_rounds must be a non-negative integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.max_depth < 1:
            raise ValueError("max_depth must be positive")
        if not horizons or np.any(~np.isfinite(horizons)) or np.any(np.asarray(horizons) <= 0):
            raise ValueError("horizons must be finite and positive")
        if len(set(horizons)) != len(horizons):
            raise ValueError("horizons must be unique")
        object.__setattr__(self, "horizons", horizons)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> LongitudinalIntegrationConfig:
        normalized = dict(values)
        if "horizons" in normalized:
            normalized["horizons"] = tuple(float(value) for value in normalized["horizons"])
        allowed = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(normalized).difference(allowed))
        if unknown:
            raise ValueError(f"unknown longitudinal integration settings: {unknown}")
        return cls(**normalized)


@dataclass
class ConstantRiskEstimator:
    value: float = 0.0

    def predict(self, rows: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(rows).shape[0], self.value, dtype=np.float32)


def survival_endpoint_features(
    scores_nmd: np.ndarray,
    pred_mask_nmd: np.ndarray,
    used_mask_nm: np.ndarray,
    endpoint: int,
) -> np.ndarray:
    scores = np.asarray(scores_nmd, dtype=np.float32)
    prediction_mask = np.asarray(pred_mask_nmd, dtype=bool)
    used = np.asarray(used_mask_nm, dtype=bool)
    if scores.ndim != 3 or prediction_mask.shape != scores.shape:
        raise ValueError("scores and pred_mask must share an N-by-M-by-D shape")
    if used.shape != scores.shape[:2]:
        raise ValueError("used_mask must have shape N-by-M")
    if endpoint < 0 or endpoint >= scores.shape[2]:
        raise IndexError(endpoint)
    active = prediction_mask[:, :, endpoint] & used
    values = np.where(active, scores[:, :, endpoint], 0.0)
    return np.concatenate(
        [values, active.astype(np.float32)],
        axis=1,
    ).astype(np.float32, copy=False)


def _predict_estimator(estimator: Any, features: np.ndarray) -> np.ndarray:
    if isinstance(estimator, ConstantRiskEstimator):
        predicted = estimator.predict(features)
    else:
        predicted = estimator.predict(features, output_margin=True)
    result = np.asarray(predicted, dtype=np.float32).reshape(-1)
    if result.shape != (features.shape[0],) or not np.isfinite(result).all():
        raise ValueError("longitudinal endpoint estimator returned invalid predictions")
    return result


@dataclass
class LongitudinalFusionModel:
    modules: tuple[str, ...]
    diseases: tuple[str, ...]
    estimators: tuple[Any, ...]
    risk_loc: np.ndarray
    risk_scale: np.ndarray
    baseline_survival: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        self.modules = tuple(str(value) for value in self.modules)
        self.diseases = tuple(str(value) for value in self.diseases)
        self.estimators = tuple(self.estimators)
        self.risk_loc = np.asarray(self.risk_loc, dtype=np.float32).reshape(-1)
        self.risk_scale = np.asarray(self.risk_scale, dtype=np.float32).reshape(-1)
        self.baseline_survival = {
            str(name): np.asarray(values, dtype=np.float64).reshape(-1)
            for name, values in self.baseline_survival.items()
        }
        if len(self.estimators) != self.D:
            raise ValueError("endpoint estimator count must match diseases")
        if self.risk_loc.shape != (self.D,) or self.risk_scale.shape != (self.D,):
            raise ValueError("risk location and scale must contain one value per disease")
        if np.any(~np.isfinite(self.risk_loc)) or np.any(self.risk_scale <= 0):
            raise ValueError("risk location and scale must be finite with positive scale")
        for name, values in self.baseline_survival.items():
            if values.shape != (self.D,) or np.any(~np.isfinite(values)):
                raise ValueError(f"baseline survival {name!r} does not align with diseases")
            if np.any((values <= 0) | (values > 1)):
                raise ValueError("baseline survival must be in (0, 1]")

    @property
    def M(self) -> int:
        return len(self.modules)

    @property
    def D(self) -> int:
        return len(self.diseases)

    def predict(
        self,
        scores_nmd: np.ndarray,
        *,
        pred_mask_nmd: np.ndarray,
        used_mask_nm: np.ndarray,
        standardized: bool = False,
    ) -> np.ndarray:
        scores = np.asarray(scores_nmd, dtype=np.float32)
        if scores.ndim != 3 or scores.shape[1:] != (self.M, self.D):
            raise ValueError(f"scores must have shape N-by-{self.M}-by-{self.D}, got {scores.shape}")
        raw = np.column_stack(
            [
                _predict_estimator(
                    estimator,
                    survival_endpoint_features(
                        scores,
                        pred_mask_nmd,
                        used_mask_nm,
                        endpoint,
                    ),
                )
                for endpoint, estimator in enumerate(self.estimators)
            ]
        ).astype(np.float32, copy=False)
        return self.standardize(raw) if standardized else raw

    def standardize(self, risk_nd: np.ndarray) -> np.ndarray:
        values = np.asarray(risk_nd, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.D:
            raise ValueError(f"risk scores must have {self.D} columns")
        return ((values - self.risk_loc) / self.risk_scale).astype(
            np.float32,
            copy=False,
        )


@dataclass(frozen=True)
class LongitudinalIntegrationResult:
    model: LongitudinalFusionModel
    metrics: Mapping[str, Any]
    train_row_indices: np.ndarray
    validation_row_indices: np.ndarray


def breslow_cumulative_hazard(
    time: np.ndarray,
    event: np.ndarray,
    log_risk: np.ndarray,
    *,
    horizon: float,
) -> float:
    durations = np.asarray(time, dtype=np.float64).reshape(-1)
    observed = np.asarray(event, dtype=bool).reshape(-1)
    risk = np.asarray(log_risk, dtype=np.float64).reshape(-1)
    if durations.shape != observed.shape or risk.shape != durations.shape:
        raise ValueError("time, event, and log_risk must share one shape")
    valid = np.isfinite(durations) & np.isfinite(risk) & (durations >= 0)
    durations = durations[valid]
    observed = observed[valid]
    risk = np.clip(risk[valid], -50.0, 50.0)
    if durations.size == 0:
        return 0.0
    cumulative = 0.0
    event_times = np.unique(durations[observed & (durations <= float(horizon))])
    relative_risk = np.exp(risk)
    for event_time in event_times:
        events = int(np.sum(observed & (durations == event_time)))
        denominator = float(np.sum(relative_risk[durations >= event_time]))
        if denominator > 0:
            cumulative += events / denominator
    return float(cumulative)


def _fit_endpoint(
    features: np.ndarray,
    signed_time: np.ndarray,
    event: np.ndarray,
    *,
    config: LongitudinalIntegrationConfig,
    endpoint: int,
    validation_features: np.ndarray | None = None,
    validation_signed_time: np.ndarray | None = None,
) -> Any:
    if int(np.sum(event > 0.5)) < config.min_events:
        return ConstantRiskEstimator()
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "Cox fusion training requires trace-agent[longitudinal]; "
            "install it with python -m pip install -e '.[longitudinal]'"
        ) from exc
    use_validation = (
        config.early_stopping_rounds > 0
        and validation_features is not None
        and validation_signed_time is not None
        and bool(np.any(validation_signed_time > 0))
    )
    estimator = XGBRegressor(
        objective="survival:cox",
        eval_metric="cox-nloglik",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        random_state=config.seed + endpoint,
        n_jobs=config.n_jobs,
        tree_method="hist",
        **({"early_stopping_rounds": config.early_stopping_rounds} if use_validation else {}),
    )
    if use_validation:
        estimator.fit(
            features,
            signed_time,
            eval_set=[(validation_features, validation_signed_time)],
            verbose=False,
        )
    else:
        estimator.fit(features, signed_time)
    return estimator


def _predict_all(
    estimators: list[Any] | tuple[Any, ...],
    scores: np.ndarray,
    pred_mask: np.ndarray,
    used_mask: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            _predict_estimator(
                estimator,
                survival_endpoint_features(
                    scores,
                    pred_mask,
                    used_mask,
                    endpoint,
                ),
            )
            for endpoint, estimator in enumerate(estimators)
        ]
    ).astype(np.float32, copy=False)


def _baseline_survival(
    train: TrainingView,
    raw_risk: np.ndarray,
    horizons: tuple[float, ...],
) -> dict[str, np.ndarray]:
    if train.targets.event_time is None:
        raise ValueError("longitudinal training requires event_time")
    result: dict[str, np.ndarray] = {}
    for horizon in horizons:
        baseline = np.ones(len(train.scores.endpoints), dtype=np.float64)
        for endpoint in range(len(train.scores.endpoints)):
            selected = train.targets.label_mask[:, endpoint]
            cumulative = breslow_cumulative_hazard(
                train.targets.event_time[selected, endpoint],
                train.targets.labels[selected, endpoint],
                raw_risk[selected, endpoint],
                horizon=horizon,
            )
            baseline[endpoint] = float(np.exp(-cumulative))
        result[_horizon_name(horizon)] = baseline
    return result


def _validation_metrics(
    model: LongitudinalFusionModel,
    validation: TrainingView,
) -> dict[str, Any]:
    if validation.targets.event_time is None:
        raise ValueError("longitudinal validation requires event_time")
    raw = model.predict(
        validation.scores.scores,
        pred_mask_nmd=validation.scores.pred_mask,
        used_mask_nm=validation.scores.availability,
    )
    horizons: dict[str, Any] = {}
    probabilities = CoxAbsoluteRiskMapper(
        endpoints=model.diseases, baseline_survival=model.baseline_survival
    ).transform(raw, validation.scores.endpoints)
    for name, probability in probabilities.items():
        horizon = float(name.removesuffix("y"))
        labels = np.zeros_like(validation.targets.labels, dtype=np.float32)
        known = np.zeros_like(validation.targets.label_mask, dtype=bool)
        for endpoint in range(len(validation.scores.endpoints)):
            endpoint_labels, endpoint_known = build_horizon_labels(
                event=validation.targets.labels[:, endpoint],
                time=validation.targets.event_time[:, endpoint],
                label_mask=validation.targets.label_mask[:, endpoint],
                horizons=np.asarray([horizon]),
            )
            labels[:, endpoint] = endpoint_labels[:, 0]
            known[:, endpoint] = endpoint_known[:, 0] & np.any(
                validation.scores.pred_mask[:, :, endpoint] & validation.scores.availability, axis=1
            )
        horizons[name] = masked_classification_metrics(
            probability,
            labels,
            known,
        )
    return {
        "participants": validation.scores.n_participants,
        "events": int(np.sum(validation.targets.labels[validation.targets.label_mask] > 0.5)),
        "horizons": horizons,
    }


def train_longitudinal_integration(
    train: TrainingView,
    validation: TrainingView,
    *,
    config: LongitudinalIntegrationConfig,
) -> LongitudinalIntegrationResult:
    require_view_role(train, SplitRole.TRAIN)
    require_view_role(validation, SplitRole.VALIDATION)
    if train.scores.task is not TaskType.LONGITUDINAL:
        raise ValueError("longitudinal integration requires longitudinal data")
    if validation.scores.task is not TaskType.LONGITUDINAL:
        raise ValueError("longitudinal validation requires longitudinal data")
    if train.scores.modules != validation.scores.modules:
        raise ValueError("train and validation module order must match")
    if train.scores.endpoints != validation.scores.endpoints:
        raise ValueError("train and validation endpoint order must match")
    if train.targets.signed_time is None or train.targets.event_time is None:
        raise ValueError("longitudinal training requires signed_time and event_time")
    if config.early_stopping_rounds and validation.targets.signed_time is None:
        raise ValueError("longitudinal validation requires signed_time")

    source_rows, observed_masks = sample_observed_masks(
        train.scores.availability,
        subsets_per_participant=config.subsets_per_participant,
        seed=config.seed,
    )
    estimators: list[Any] = []
    selected_rounds: dict[str, int] = {}
    for endpoint in range(len(train.scores.endpoints)):
        selected = train.targets.label_mask[source_rows, endpoint]
        selected &= np.isfinite(train.targets.signed_time[source_rows, endpoint])
        features = survival_endpoint_features(
            train.scores.scores[source_rows],
            train.scores.pred_mask[source_rows],
            observed_masks,
            endpoint,
        )[selected]
        signed_time = train.targets.signed_time[source_rows, endpoint][selected]
        event = train.targets.labels[source_rows, endpoint][selected]
        validation_features = None
        validation_signed_time = None
        if config.early_stopping_rounds and validation.targets.signed_time is not None:
            validation_selected = (
                validation.targets.label_mask[:, endpoint]
                & np.isfinite(validation.targets.signed_time[:, endpoint])
                & np.any(
                    validation.scores.pred_mask[:, :, endpoint] & validation.scores.availability,
                    axis=1,
                )
            )
            validation_features = survival_endpoint_features(
                validation.scores.scores,
                validation.scores.pred_mask,
                validation.scores.availability,
                endpoint,
            )[validation_selected]
            validation_signed_time = validation.targets.signed_time[validation_selected, endpoint]
        estimator = _fit_endpoint(
            features,
            signed_time,
            event,
            config=config,
            endpoint=endpoint,
            validation_features=validation_features,
            validation_signed_time=validation_signed_time,
        )
        estimators.append(estimator)
        selected_rounds[train.scores.endpoints[endpoint]] = (
            0
            if isinstance(estimator, ConstantRiskEstimator)
            else int(getattr(estimator, "best_iteration", config.n_estimators - 1)) + 1
        )

    train_raw = _predict_all(
        estimators,
        train.scores.scores,
        train.scores.pred_mask,
        train.scores.availability,
    )
    risk_loc = np.mean(train_raw, axis=0).astype(np.float32)
    risk_scale = np.maximum(np.std(train_raw, axis=0), 1e-6).astype(np.float32)
    model = LongitudinalFusionModel(
        modules=train.scores.modules,
        diseases=train.scores.endpoints,
        estimators=tuple(estimators),
        risk_loc=risk_loc,
        risk_scale=risk_scale,
        baseline_survival=_baseline_survival(
            train,
            train_raw,
            config.horizons,
        ),
    )
    return LongitudinalIntegrationResult(
        model=model,
        metrics={**_validation_metrics(model, validation), "selected_rounds": selected_rounds},
        train_row_indices=train.row_indices.copy(),
        validation_row_indices=validation.row_indices.copy(),
    )
