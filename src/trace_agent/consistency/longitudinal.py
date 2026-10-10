from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LongitudinalConsistencyScale:
    """Frozen positive relative-risk q99, represented in log space."""

    log_reference: float
    score_kind: str = "log_risk"

    def __post_init__(self) -> None:
        if self.score_kind not in {"log_risk", "relative_risk"}:
            raise ValueError("module_score_kind must be log_risk or relative_risk")
        if not np.isfinite(self.log_reference):
            raise ValueError("longitudinal scale must have a finite log reference")

    def transform(self, values: np.ndarray) -> np.ndarray:
        scores = np.asarray(values, dtype=np.float64)
        if self.score_kind == "relative_risk":
            with np.errstate(divide="ignore", invalid="ignore"):
                scores = np.log(np.maximum(scores, 0))
        return np.exp(np.minimum(scores - self.log_reference, 0)).astype(np.float32)


def fit_longitudinal_scale(
    scores: np.ndarray,
    pred_mask: np.ndarray,
    validation_rows: np.ndarray,
    *,
    score_kind: str = "log_risk",
    chunk_size: int = 8192,
    max_values: int = 2_000_000,
) -> LongitudinalConsistencyScale:
    """Use only validation scores; preserve the old q99 sampling/linear quantile."""
    LongitudinalConsistencyScale(0.0, score_kind)
    rows = np.asarray(validation_rows, dtype=np.int64).reshape(-1)
    if rows.size == 0:
        raise ValueError("longitudinal scale requires validation rows")
    if scores.ndim != 3 or pred_mask.shape != scores.shape:
        raise ValueError("longitudinal scores/mask must have matching N,M,D shapes")
    if np.any(rows < 0) or np.any(rows >= scores.shape[0]):
        raise ValueError("validation rows are out of range")

    def chunks():
        for start in range(0, rows.size, max(1, chunk_size)):
            selected = rows[start : start + max(1, chunk_size)]
            values = np.asarray(scores[selected], dtype=np.float32)
            valid = np.asarray(pred_mask[selected], dtype=bool) & np.isfinite(values)
            if score_kind == "relative_risk":
                valid &= values > 0
            yield values[valid]

    count = sum(values.size for values in chunks())
    if count == 0:
        return LongitudinalConsistencyScale(0.0, score_kind)
    step = max(1, int(np.ceil(count / max(1, max_values))))
    sampled = np.concatenate([values[::step].copy() for values in chunks()]).astype(np.float64)
    logs = np.log(sampled) if score_kind == "relative_risk" else sampled
    logs.sort()
    position = (logs.size - 1) * 0.99
    lower = int(np.floor(position))
    fraction = position - lower
    log_reference = float(logs[lower])
    if fraction > 0:
        log_reference = float(
            np.logaddexp(logs[lower] + np.log1p(-fraction), logs[lower + 1] + np.log(fraction))
        )
    # The old HR implementation clamps the denominator at 1e-12.
    return LongitudinalConsistencyScale(max(log_reference, float(np.log(1e-12))), score_kind)


def load_longitudinal_scale(
    path: str | Path, *, modules: tuple[str, ...], endpoints: tuple[str, ...]
) -> LongitudinalConsistencyScale:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("schema") != "trace.longitudinal_consistency_scale.v1":
        raise ValueError("unsupported longitudinal consistency scale schema")
    if tuple(record.get("modules", [])) != modules or tuple(record.get("endpoints", [])) != endpoints:
        raise ValueError("longitudinal consistency scale module/endpoint order differs")
    if record.get("source_split") != "validation":
        raise ValueError("longitudinal consistency scale must be fitted on validation")
    return LongitudinalConsistencyScale(float(record["log_reference"]), record["module_score_kind"])
