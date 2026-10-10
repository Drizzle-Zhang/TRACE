from __future__ import annotations

import numpy as np


def build_action_features(
    module_index: int | None,
    *,
    n_modules: int,
    burden: float,
) -> np.ndarray:
    """Build the legacy-compatible ``[one_hot, burden, is_stop]`` vector."""

    if n_modules <= 0:
        raise ValueError("n_modules must be positive")
    one_hot = np.zeros(int(n_modules), dtype=np.float32)
    index = -1 if module_index is None else int(module_index)
    is_stop = index < 0
    if not is_stop:
        if index >= n_modules:
            raise IndexError(f"module index {index} is outside [0, {n_modules})")
        one_hot[index] = 1.0
    return np.concatenate(
        [
            one_hot,
            np.asarray([float(burden), float(is_stop)], dtype=np.float32),
        ]
    )
