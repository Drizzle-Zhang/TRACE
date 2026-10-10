"""Validated settings shared by native module trainers."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from trace_agent.contracts.task import TaskType


class InsufficientModuleData(ValueError):
    """A pair cannot support the requested train-only fitting procedure."""


@dataclass(frozen=True)
class ModuleFit:
    estimator: Any
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ModuleTrainingSettings:
    seed: int = 7
    min_positive: int = 50
    min_positive_rate: float = 0.001
    insufficient_data_policy: str = "skip"
    n_estimators: int = 100
    learning_rate: float = 0.1
    n_jobs: int = 1
    num_leaves: int = 31
    min_child_samples: int = 20
    max_depth: int = 6
    early_stopping_rounds: int = 50
    inner_validation_fraction: float = 0.2

    def __post_init__(self) -> None:
        for name in (
            "seed",
            "min_positive",
            "n_estimators",
            "n_jobs",
            "num_leaves",
            "min_child_samples",
            "max_depth",
            "early_stopping_rounds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")  # noqa: TRY004 -- config contract
        if self.min_positive < 1 or self.n_estimators < 1 or self.n_jobs < 1:
            raise ValueError("training counts and n_jobs must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.num_leaves < 2 or self.min_child_samples < 1 or self.max_depth < 1:
            raise ValueError("tree size settings are invalid")
        if self.early_stopping_rounds < 1:
            raise ValueError("early stopping patience must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not 0 <= self.min_positive_rate <= 1:
            raise ValueError("min_positive_rate must be in [0,1]")
        if not 0 < self.inner_validation_fraction < 1:
            raise ValueError("inner_validation_fraction must be in (0,1)")
        if self.insufficient_data_policy not in {"skip", "error"}:
            raise ValueError("insufficient_data_policy must be skip or error")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], *, task: TaskType | str) -> "ModuleTrainingSettings":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values).difference(allowed))
        if unknown:
            raise ValueError(f"unknown module training settings: {unknown}")
        defaults = {"n_estimators": 1000} if TaskType.parse(task) is TaskType.LONGITUDINAL else {}
        return cls(**{**defaults, **dict(values)})


def check_binary_training(labels: np.ndarray, settings: ModuleTrainingSettings, *, min_class: int) -> None:
    values = np.asarray(labels, dtype=float).reshape(-1)
    if not np.isfinite(values).all() or not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("module fit labels must be finite and binary")
    positive = int(np.sum(values == 1))
    negative = int(np.sum(values == 0))
    if positive < settings.min_positive or negative < min_class or positive < min_class:
        raise InsufficientModuleData("insufficient positive/event or class count")
    if positive / values.size < settings.min_positive_rate:
        raise InsufficientModuleData("positive/event rate below configured threshold")
