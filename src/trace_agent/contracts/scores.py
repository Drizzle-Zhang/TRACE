from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from trace_agent.contracts.task import TaskType


def _names(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    if len(set(names)) != len(names):
        raise ValueError(f"{label} names must be unique")
    if any(not name for name in names):
        raise ValueError(f"{label} names must be non-empty")
    return names


@dataclass(frozen=True)
class ModuleScoreBatch:
    participant_ids: np.ndarray
    modules: tuple[str, ...]
    endpoints: tuple[str, ...]
    scores: np.ndarray
    pred_mask: np.ndarray
    availability: np.ndarray
    task: TaskType
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        participant_ids = np.asarray(self.participant_ids).reshape(-1)
        modules = _names(tuple(self.modules), label="module")
        endpoints = _names(tuple(self.endpoints), label="endpoint")
        scores = np.asarray(self.scores, dtype=np.float32)
        pred_mask = np.asarray(self.pred_mask, dtype=bool)
        availability = np.asarray(self.availability, dtype=bool)
        expected = (participant_ids.size, len(modules), len(endpoints))
        if scores.shape != expected:
            raise ValueError(f"scores shape must be {expected}, got {scores.shape}")
        if pred_mask.shape != expected:
            raise ValueError(f"pred_mask shape must be {expected}, got {pred_mask.shape}")
        if availability.shape != expected[:2]:
            raise ValueError(f"availability shape must be {expected[:2]}, got {availability.shape}")
        if np.any(pred_mask & ~availability[:, :, None]):
            raise ValueError("pred_mask cannot mark an unavailable module as valid")
        if np.any(~pred_mask & ~np.isnan(scores)):
            raise ValueError("masked-out scores must be NaN")
        if np.any(pred_mask & ~np.isfinite(scores)):
            raise ValueError("valid module scores must be finite")
        object.__setattr__(self, "participant_ids", participant_ids)
        object.__setattr__(self, "modules", modules)
        object.__setattr__(self, "endpoints", endpoints)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "pred_mask", pred_mask)
        object.__setattr__(self, "availability", availability)
        object.__setattr__(self, "task", TaskType.parse(self.task))

    @property
    def n_participants(self) -> int:
        return int(self.participant_ids.size)


@dataclass(frozen=True)
class RiskProfile:
    participant_ids: np.ndarray
    endpoints: tuple[str, ...]
    raw_scores: np.ndarray
    valid_mask: np.ndarray
    observed_modules: np.ndarray
    task: TaskType
    calibrated_scores: np.ndarray | None = None
    horizon_risks: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        participant_ids = np.asarray(self.participant_ids).reshape(-1)
        endpoints = _names(tuple(self.endpoints), label="endpoint")
        expected = (participant_ids.size, len(endpoints))
        raw_scores = np.asarray(self.raw_scores, dtype=np.float32)
        valid_mask = np.asarray(self.valid_mask, dtype=bool)
        observed_modules = np.asarray(self.observed_modules, dtype=bool)
        if raw_scores.shape != expected:
            raise ValueError(f"raw_scores shape must be {expected}, got {raw_scores.shape}")
        if valid_mask.shape != expected:
            raise ValueError(f"valid_mask shape must be {expected}, got {valid_mask.shape}")
        if observed_modules.ndim != 2 or observed_modules.shape[0] != participant_ids.size:
            raise ValueError("observed_modules must be a two-dimensional participant-by-module matrix")
        calibrated = None
        if self.calibrated_scores is not None:
            calibrated = np.asarray(self.calibrated_scores, dtype=np.float32)
            if calibrated.shape != expected:
                raise ValueError(f"calibrated_scores shape must be {expected}, got {calibrated.shape}")
        horizons: dict[str, np.ndarray] = {}
        for horizon, values in self.horizon_risks.items():
            array = np.asarray(values, dtype=np.float32)
            if array.shape != expected:
                raise ValueError(f"horizon risk {horizon!r} shape must be {expected}, got {array.shape}")
            horizons[str(horizon)] = array
        object.__setattr__(self, "participant_ids", participant_ids)
        object.__setattr__(self, "endpoints", endpoints)
        object.__setattr__(self, "raw_scores", raw_scores)
        object.__setattr__(self, "valid_mask", valid_mask)
        object.__setattr__(self, "observed_modules", observed_modules)
        object.__setattr__(self, "task", TaskType.parse(self.task))
        object.__setattr__(self, "calibrated_scores", calibrated)
        object.__setattr__(self, "horizon_risks", horizons)

    @property
    def n_participants(self) -> int:
        return int(self.participant_ids.size)

    @property
    def decision_scores(self) -> np.ndarray:
        if self.calibrated_scores is not None:
            return self.calibrated_scores
        return self.raw_scores
