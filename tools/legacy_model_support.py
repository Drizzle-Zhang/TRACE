"""Explicit inference-only adapters for the frozen, packaged legacy weights.

These adapters are audit tools, not newly fitted production artifacts. Original
Python dependencies must be on PYTHONPATH; original files are never rewritten.
"""

from __future__ import annotations

import ctypes
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def bind_calibrated_class(code: Path):
    name = "frozen_legacy_calibration"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, code / "scripts/calibrate_fusion_model.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    cls = sys.modules[name].CalibratedFusionModel
    # The original command-line script serialized this class in __main__.
    sys.modules["__main__"].CalibratedFusionModel = cls
    return cls


def configure_threads(model: Any) -> None:
    if hasattr(model, "models") and model.models is not None:
        for estimator in model.models:
            if estimator is not None:
                configure_threads(estimator)
    elif hasattr(model, "model"):
        configure_threads(model.model)
    elif hasattr(model, "set_params"):
        model.set_params(n_jobs=1)
        if hasattr(model, "get_booster"):
            model.get_booster().set_param({"nthread": 1})


def load_original(path: Path, code: Path):
    bind_calibrated_class(code)
    model = joblib.load(path, mmap_mode="r" if path.name == "q_model.joblib" else None)
    configure_threads(model)
    return model


def cox_margin_to_legacy_hr(values):
    """Use the saved CPU Cox predictor's expf, including float32 rounding.

    NumPy's SIMD exponential is close numerically but can cross tree cuts in a
    fusion trained on exact HR values. This audit conversion intentionally uses
    the expf symbol resolved from the loaded XGBoost runtime. A runtime that
    does not export it cannot claim this legacy compatibility path.
    """
    from xgboost.core import _LIB

    try:
        exponential = _LIB.expf
    except AttributeError as error:
        raise RuntimeError("legacy CPU Cox conversion requires XGBoost's expf symbol") from error
    exponential.argtypes = [ctypes.c_float]
    exponential.restype = ctypes.c_float
    margins = np.asarray(values, dtype=np.float32)
    return np.fromiter(
        (exponential(float(value)) for value in margins.reshape(-1)), dtype=np.float32, count=margins.size
    ).reshape(margins.shape)


@dataclass
class LegacyPrevalentFusion:
    """Stable pickle wrapper delegating to the actual frozen calibration class."""

    fusion_model: Any
    calibrators: dict
    diseases: list[str]
    code_root: str

    @property
    def M(self):
        return self.fusion_model.M

    @property
    def D(self):
        return self.fusion_model.D

    def _original(self):
        return bind_calibrated_class(Path(self.code_root))(self.fusion_model, self.calibrators, self.diseases)

    def fuse_logits(self, *args, **kwargs):
        return self._original().fuse_logits(*args, **kwargs)

    def fuse(self, *args, **kwargs):
        return self._original().fuse(*args, **kwargs)


@dataclass
class LegacyCoxFusion:
    """Convert native log-risk inputs to the legacy fusion's HR feature scale.

    The learned Cox fusion still sees HR. Its output is obtained as a native
    log margin. Old Q standardization is preserved by exponentiating only at
    that boundary. Missing learned fusion models remain NaN.
    """

    original: Any

    @property
    def M(self):
        return self.original.M

    @property
    def D(self):
        return self.original.D

    def predict(self, scores_nmd, *, pred_mask_nmd=None, used_mask_nm=None, standardized=False):
        scores = np.asarray(scores_nmd, dtype=np.float32)
        if scores.ndim != 3 or scores.shape[1:] != (self.M, self.D):
            raise ValueError("legacy Cox bridge input axis mismatch")
        hr = cox_margin_to_legacy_hr(scores)
        if np.any(np.isfinite(scores) & ~np.isfinite(hr)):
            raise FloatingPointError(
                "legacy HR feature scale overflowed; cannot preserve the old fusion input"
            )
        out = np.full((scores.shape[0], self.D), np.nan, dtype=np.float32)
        for d, model in enumerate(self.original.models):
            if model is None:
                continue
            indices = self.original.active_module_indices[d]
            if len(indices) == 0:
                continue
            x = hr[:, indices, d].copy()
            if pred_mask_nmd is not None:
                x[~np.asarray(pred_mask_nmd[:, indices, d], bool)] = np.nan
            if used_mask_nm is not None:
                x[~np.asarray(used_mask_nm[:, indices], bool)] = np.nan
            out[:, d] = model.predict(x, output_margin=True)
        return self.standardize(out) if standardized else out

    def standardize(self, margins):
        hr = cox_margin_to_legacy_hr(margins)
        return self.original.standardize(hr)
