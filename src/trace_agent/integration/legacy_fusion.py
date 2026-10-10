"""Portable inference state for the selected historical LightGBM and Cox fusions.

These classes preserve the frozen predictors' feature and calibration contracts.
They contain no original project wrapper, source directory, or fitting inputs.
"""

from __future__ import annotations

import ctypes
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def _raw_score(model: Any, values: np.ndarray) -> np.ndarray | None:
    booster = getattr(model, "booster_", None)
    if booster is not None:
        try:
            n_jobs = getattr(model, "n_jobs", None)
            settings = {"num_threads": int(n_jobs)} if n_jobs is not None and int(n_jobs) > 0 else {}
            return np.asarray(booster.predict(values, raw_score=True, **settings), dtype=np.float32).reshape(
                -1
            )
        # Some frozen wrappers reject booster raw scoring; retain their fallback.
        except Exception:  # noqa: BLE001, S110
            pass
    try:
        return np.asarray(model.predict(values, raw_score=True), dtype=np.float32).reshape(-1)
    except TypeError:
        return None


@dataclass
class LegacyPrevalentFusion:
    """Historical selected-module LightGBM features and probability calibration.

    Calibrators use integer endpoint indices. Calibration is included in both
    returned probabilities and the logits consumed by the historical Q model.
    """

    modules: tuple[str, ...]
    diseases: tuple[str, ...]
    models: tuple[Any | None, ...]
    model_feature_indices: tuple[np.ndarray | None, ...] | None = None
    model_input_space: str = "logit"
    calibrators: Mapping[int, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.modules = tuple(str(value) for value in self.modules)
        self.diseases = tuple(str(value) for value in self.diseases)
        self.models = tuple(self.models)
        if len(self.models) != self.D:
            raise ValueError("models must contain one predictor per endpoint")
        if self.model_feature_indices is not None:
            self.model_feature_indices = tuple(
                None if indices is None else np.asarray(indices, dtype=np.int64).reshape(-1).copy()
                for indices in self.model_feature_indices
            )
            if len(self.model_feature_indices) != self.D:
                raise ValueError("model feature indices must contain one entry per endpoint")
        space = str(self.model_input_space).strip().lower()
        self.model_input_space = "prob" if space in {"prob", "probability", "probabilities"} else "logit"
        self.calibrators = dict(self.calibrators)

    @property
    def M(self) -> int:
        return len(self.modules)

    @property
    def D(self) -> int:
        return len(self.diseases)

    def fuse_logits(
        self,
        probs_md: np.ndarray,
        mask_m: np.ndarray,
        pred_mask_md: np.ndarray | None = None,
        eps: float = 1e-6,
    ) -> np.ndarray:
        scores = np.asarray(probs_md, dtype=np.float32)
        observed = np.asarray(mask_m, dtype=np.float32).reshape(self.M)
        prediction_mask = (
            (~np.isnan(scores)).astype(np.float32)
            if pred_mask_md is None
            else np.asarray(pred_mask_md, dtype=np.float32)
        )
        if scores.shape != (self.M, self.D) or prediction_mask.shape != scores.shape:
            raise ValueError(f"scores and prediction mask must have shape ({self.M}, {self.D})")
        transformed = np.zeros((self.M, self.D), dtype=np.float32)
        valid = prediction_mask > 0
        clipped = np.clip(scores[valid], eps, 1.0 - eps).astype(np.float32, copy=False)
        transformed[valid] = clipped if self.model_input_space == "prob" else _logit(clipped)
        logits = np.zeros(self.D, dtype=np.float32)
        for endpoint, model in enumerate(self.models):
            if model is None:
                continue
            indices = None if self.model_feature_indices is None else self.model_feature_indices[endpoint]
            if indices is None:
                indices = np.arange(self.M, dtype=np.int64)
            values = np.full(indices.size, np.nan, dtype=np.float32)
            active = (prediction_mask[indices, endpoint] > 0) & (observed[indices] > 0)
            values[active] = transformed[indices[active], endpoint]
            row = values.reshape(1, -1)
            raw = _raw_score(model, row)
            if raw is not None:
                logits[endpoint] = float(raw[0])
            else:
                probability = np.asarray(model.predict_proba(row))[:, 1]
                logits[endpoint] = _logit(np.clip(probability, eps, 1.0 - eps))[0]
        probabilities = _sigmoid(logits)
        calibrated = np.array(probabilities, copy=True)
        for endpoint, calibrator in self.calibrators.items():
            index = int(endpoint)
            if 0 <= index < self.D:
                value = np.asarray([probabilities[index]], dtype=np.float32)
                calibrated[index] = float(calibrator.predict(value)[0])
        return _logit(calibrated)

    def fuse(
        self,
        probs_md: np.ndarray,
        mask_m: np.ndarray,
        pred_mask_md: np.ndarray | None = None,
        eps: float = 1e-6,
    ) -> np.ndarray:
        return _sigmoid(self.fuse_logits(probs_md, mask_m, pred_mask_md=pred_mask_md, eps=eps))


def cox_margin_to_legacy_hr(values: np.ndarray) -> np.ndarray:
    """Preserve saved CPU Cox HR values, including float32 expf rounding."""
    from xgboost.core import _LIB

    try:
        exponential = _LIB.expf
    except AttributeError as error:
        raise RuntimeError("legacy CPU Cox conversion requires XGBoost's expf symbol") from error
    exponential.argtypes = [ctypes.c_float]
    exponential.restype = ctypes.c_float
    margins = np.asarray(values, dtype=np.float32)
    return np.fromiter(
        (exponential(float(value)) for value in margins.flat), dtype=np.float32, count=margins.size
    ).reshape(margins.shape)


@dataclass
class LegacyCoxFusion:
    """Selected HR features with native log-risk output and saved HR scaling."""

    modules: tuple[str, ...]
    diseases: tuple[str, ...]
    models: tuple[Any | None, ...]
    active_module_indices: tuple[np.ndarray, ...]
    risk_loc: np.ndarray
    risk_scale: np.ndarray

    def __post_init__(self) -> None:
        self.modules = tuple(str(value) for value in self.modules)
        self.diseases = tuple(str(value) for value in self.diseases)
        self.models = tuple(self.models)
        self.active_module_indices = tuple(
            np.asarray(indices, dtype=np.int64).reshape(-1).copy() for indices in self.active_module_indices
        )
        self.risk_loc = np.asarray(self.risk_loc, dtype=np.float32).reshape(-1).copy()
        self.risk_scale = np.asarray(self.risk_scale, dtype=np.float32).reshape(-1).copy()
        if len(self.models) != self.D or len(self.active_module_indices) != self.D:
            raise ValueError("models and active module indices must contain one entry per endpoint")
        if self.risk_loc.shape != (self.D,) or self.risk_scale.shape != (self.D,):
            raise ValueError("saved risk location and scale must contain one value per endpoint")

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
        pred_mask_nmd: np.ndarray | None = None,
        used_mask_nm: np.ndarray | None = None,
        standardized: bool = False,
    ) -> np.ndarray:
        scores = np.asarray(scores_nmd, dtype=np.float32)
        if scores.ndim != 3 or scores.shape[1:] != (self.M, self.D):
            raise ValueError("legacy Cox bridge input axis mismatch")
        hr = cox_margin_to_legacy_hr(scores)
        if np.any(np.isfinite(scores) & ~np.isfinite(hr)):
            raise FloatingPointError(
                "legacy HR feature scale overflowed; cannot preserve saved fusion inputs"
            )
        output = np.full((scores.shape[0], self.D), np.nan, dtype=np.float32)
        for endpoint, model in enumerate(self.models):
            indices = self.active_module_indices[endpoint]
            if model is None or indices.size == 0:
                continue
            values = hr[:, indices, endpoint].copy()
            if pred_mask_nmd is not None:
                values[~np.asarray(pred_mask_nmd[:, indices, endpoint], dtype=bool)] = np.nan
            if used_mask_nm is not None:
                values[~np.asarray(used_mask_nm[:, indices], dtype=bool)] = np.nan
            output[:, endpoint] = model.predict(values, output_margin=True)
        return self.standardize(output) if standardized else output

    def standardize(self, margins: np.ndarray) -> np.ndarray:
        hr = cox_margin_to_legacy_hr(margins)
        scale = np.where(
            np.isfinite(self.risk_scale) & (np.abs(self.risk_scale) > 1e-6), self.risk_scale, 1.0
        ).astype(np.float32, copy=False)
        return (hr - self.risk_loc) / scale
