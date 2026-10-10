from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from lightgbm import Booster, LGBMClassifier

from trace_agent.contracts.task import SplitRole, TaskType
from trace_agent.data.training_dataset import TrainingView
from trace_agent.training.common import (
    masked_classification_metrics,
    require_view_role,
    sample_observed_masks,
)


@dataclass(frozen=True)
class PrevalentIntegrationConfig:
    seed: int = 7
    subsets_per_participant: int = 4
    min_class_count: int = 10
    n_estimators: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 31
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if self.subsets_per_participant < 0:
            raise ValueError("subsets_per_participant cannot be negative")
        if self.min_class_count < 1:
            raise ValueError("min_class_count must be positive")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.num_leaves < 2:
            raise ValueError("num_leaves must be at least two")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> PrevalentIntegrationConfig:
        allowed = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(values).difference(allowed))
        if unknown:
            raise ValueError(f"unknown prevalent integration settings: {unknown}")
        return cls(**dict(values))


@dataclass
class ConstantProbabilityEstimator:
    probability: float

    def predict_proba(self, rows: np.ndarray) -> np.ndarray:
        n_rows = int(np.asarray(rows).shape[0])
        positive = np.full(n_rows, float(self.probability), dtype=np.float64)
        return np.column_stack([1.0 - positive, positive])


def endpoint_features(
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
    probability = np.clip(scores[:, :, endpoint], 1e-6, 1.0 - 1e-6)
    logits = np.log(probability / (1.0 - probability))
    values = np.where(active, logits, 0.0)
    return np.concatenate(
        [values, active.astype(np.float32)],
        axis=1,
    ).astype(np.float32, copy=False)


def _positive_probability(estimator: Any, rows: np.ndarray) -> np.ndarray:
    booster = getattr(estimator, "booster_", None)
    if booster is not None:
        prediction_settings = (
            {"num_threads": booster.params.get("num_threads") or 1} if isinstance(booster, Booster) else {}
        )
        probability = np.asarray(booster.predict(rows, **prediction_settings), dtype=np.float64).reshape(-1)
        if probability.shape != (rows.shape[0],):
            raise ValueError("prevalent endpoint estimator returned an invalid shape")
        return np.clip(probability, 0.0, 1.0).astype(np.float32)
    probability = np.asarray(estimator.predict_proba(rows), dtype=np.float64)
    if probability.shape != (rows.shape[0], 2):
        raise ValueError("prevalent endpoint estimator returned an invalid shape")
    return np.clip(probability[:, 1], 0.0, 1.0).astype(np.float32)


@dataclass
class PrevalentFusionModel:
    modules: tuple[str, ...]
    diseases: tuple[str, ...]
    estimators: tuple[Any, ...]

    def __post_init__(self) -> None:
        self.modules = tuple(str(value) for value in self.modules)
        self.diseases = tuple(str(value) for value in self.diseases)
        self.estimators = tuple(self.estimators)
        if len(self.estimators) != len(self.diseases):
            raise ValueError("endpoint estimator count must match diseases")

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
    ) -> np.ndarray:
        scores = np.asarray(scores_nmd, dtype=np.float32)
        if scores.shape[1:] != (self.M, self.D):
            raise ValueError(f"scores must have trailing shape {(self.M, self.D)}, got {scores.shape}")
        return np.column_stack(
            [
                _positive_probability(
                    estimator,
                    endpoint_features(
                        scores,
                        pred_mask_nmd,
                        used_mask_nm,
                        endpoint,
                    ),
                )
                for endpoint, estimator in enumerate(self.estimators)
            ]
        ).astype(np.float32, copy=False)

    def fuse(
        self,
        probs_md: np.ndarray,
        mask_m: np.ndarray,
        pred_mask_md: np.ndarray | None = None,
    ) -> np.ndarray:
        scores = np.asarray(probs_md, dtype=np.float32)
        prediction_mask = (
            np.isfinite(scores) if pred_mask_md is None else np.asarray(pred_mask_md, dtype=bool)
        )
        result = self.predict(
            scores[None, :, :],
            pred_mask_nmd=prediction_mask[None, :, :],
            used_mask_nm=np.asarray(mask_m, dtype=bool)[None, :],
        )
        return result[0]

    def fuse_logits(
        self,
        probs_md: np.ndarray,
        mask_m: np.ndarray,
        pred_mask_md: np.ndarray | None = None,
    ) -> np.ndarray:
        probability = np.clip(
            self.fuse(probs_md, mask_m, pred_mask_md),
            1e-7,
            1.0 - 1e-7,
        )
        return np.log(probability / (1.0 - probability)).astype(np.float32)


@dataclass(frozen=True)
class PrevalentIntegrationResult:
    model: PrevalentFusionModel
    metrics: Mapping[str, Any]
    train_row_indices: np.ndarray
    validation_row_indices: np.ndarray


def _fit_endpoint(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    config: PrevalentIntegrationConfig,
    endpoint: int,
) -> Any:
    negative = int(np.sum(labels == 0))
    positive = int(np.sum(labels == 1))
    if min(negative, positive) < config.min_class_count:
        probability = float(np.mean(labels)) if labels.size else 0.5
        return ConstantProbabilityEstimator(probability=probability)
    estimator = LGBMClassifier(
        objective="binary",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        random_state=config.seed + endpoint,
        n_jobs=config.n_jobs,
        verbosity=-1,
    )
    estimator.fit(features, labels.astype(np.int8, copy=False))
    return estimator


def train_prevalent_integration(
    train: TrainingView,
    validation: TrainingView,
    *,
    config: PrevalentIntegrationConfig,
) -> PrevalentIntegrationResult:
    require_view_role(train, SplitRole.TRAIN)
    require_view_role(validation, SplitRole.VALIDATION)
    if train.scores.task is not TaskType.PREVALENT:
        raise ValueError("prevalent integration requires prevalent score data")
    if validation.scores.task is not TaskType.PREVALENT:
        raise ValueError("prevalent validation requires prevalent score data")
    if train.scores.modules != validation.scores.modules:
        raise ValueError("train and validation module order must match")
    if train.scores.endpoints != validation.scores.endpoints:
        raise ValueError("train and validation endpoint order must match")

    source_rows, observed_masks = sample_observed_masks(
        train.scores.availability,
        subsets_per_participant=config.subsets_per_participant,
        seed=config.seed,
    )
    augmented_scores = train.scores.scores[source_rows]
    augmented_prediction_mask = train.scores.pred_mask[source_rows]
    estimators: list[Any] = []
    for endpoint in range(len(train.scores.endpoints)):
        selected = train.targets.label_mask[source_rows, endpoint]
        selected &= np.isfinite(train.targets.labels[source_rows, endpoint])
        features = endpoint_features(
            augmented_scores,
            augmented_prediction_mask,
            observed_masks,
            endpoint,
        )[selected]
        labels = train.targets.labels[source_rows, endpoint][selected]
        estimators.append(
            _fit_endpoint(
                features,
                labels,
                config=config,
                endpoint=endpoint,
            )
        )

    model = PrevalentFusionModel(
        modules=train.scores.modules,
        diseases=train.scores.endpoints,
        estimators=tuple(estimators),
    )
    validation_probability = model.predict(
        validation.scores.scores,
        pred_mask_nmd=validation.scores.pred_mask,
        used_mask_nm=validation.scores.availability,
    )
    return PrevalentIntegrationResult(
        model=model,
        metrics=masked_classification_metrics(
            validation_probability,
            validation.targets.labels,
            validation.targets.label_mask
            & np.any(validation.scores.pred_mask & validation.scores.availability[:, :, None], axis=1),
        ),
        train_row_indices=train.row_indices.copy(),
        validation_row_indices=validation.row_indices.copy(),
    )
