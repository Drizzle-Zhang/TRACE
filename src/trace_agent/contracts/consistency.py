from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from trace_agent.contracts.task import SplitRole


@dataclass(frozen=True)
class FeatureConsistencyResult:
    scores: np.ndarray
    raw_scores: np.ndarray
    matched_counts: np.ndarray


@dataclass(frozen=True)
class EvidenceConsistencyResult:
    scores: np.ndarray
    support_scores: np.ndarray
    no_conflict_scores: np.ndarray
    matched_counts: np.ndarray


@dataclass(frozen=True)
class ConsistencyThresholds:
    prediction: float
    kg: float
    feature: float
    evidence: float
    combined: float
    source_split: SplitRole = SplitRole.VALIDATION
    metric: str = "f1"

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.prediction,
                self.kg,
                self.feature,
                self.evidence,
                self.combined,
            ],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("consistency thresholds must be finite")
        if SplitRole.parse(self.source_split) is not SplitRole.VALIDATION:
            raise ValueError("consistency thresholds must originate from validation")
        object.__setattr__(self, "source_split", SplitRole.parse(self.source_split))


@dataclass(frozen=True)
class ConsistencyResult:
    kg_scores: np.ndarray
    feature_scores: np.ndarray
    evidence_scores: np.ndarray
    combined_scores: np.ndarray
    strategy_masks: Mapping[str, np.ndarray]
    thresholds: ConsistencyThresholds
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kg = np.asarray(self.kg_scores, dtype=np.float32)
        feature = np.asarray(self.feature_scores, dtype=np.float32)
        evidence = np.asarray(self.evidence_scores, dtype=np.float32)
        combined = np.asarray(self.combined_scores, dtype=np.float32)
        if not (kg.shape == feature.shape == evidence.shape == combined.shape):
            raise ValueError("all consistency score arrays must share one shape")
        masks: dict[str, np.ndarray] = {}
        for name, values in self.strategy_masks.items():
            mask = np.asarray(values, dtype=bool)
            if mask.shape != combined.shape:
                raise ValueError(f"strategy mask {name!r} does not match score shape")
            masks[str(name)] = mask
        object.__setattr__(self, "kg_scores", kg)
        object.__setattr__(self, "feature_scores", feature)
        object.__setattr__(self, "evidence_scores", evidence)
        object.__setattr__(self, "combined_scores", combined)
        object.__setattr__(self, "strategy_masks", masks)
