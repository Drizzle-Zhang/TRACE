from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def apply_endpoint_calibrators(
    scores: np.ndarray,
    endpoints: Sequence[str],
    calibrators: Mapping[str, Any] | None,
) -> np.ndarray | None:
    if not calibrators:
        return None
    array = np.asarray(scores, dtype=np.float32)
    calibrated = array.copy()
    for endpoint_index, endpoint in enumerate(endpoints):
        calibrator = calibrators.get(str(endpoint))
        if calibrator is None:
            continue
        valid = np.isfinite(array[:, endpoint_index])
        if not np.any(valid):
            continue
        predicted = np.asarray(
            calibrator.predict(array[valid, endpoint_index]),
            dtype=np.float32,
        ).reshape(-1)
        if predicted.shape != (int(np.sum(valid)),):
            raise ValueError(f"calibrator for {endpoint!r} returned an invalid shape")
        calibrated[valid, endpoint_index] = np.clip(predicted, 0.0, 1.0)
    return calibrated
