from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _unique_name_map(names: Sequence[str], *, label: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for index, value in enumerate(names):
        name = str(value).strip()
        if not name:
            raise ValueError(f"{label} contains an empty name")
        if name in mapping:
            raise ValueError(f"{label} contains duplicate name {name!r}")
        mapping[name] = index
    return mapping


def build_name_permutation(
    source_names: Sequence[str],
    target_names: Sequence[str],
) -> np.ndarray:
    source = _unique_name_map(source_names, label="source names")
    _unique_name_map(target_names, label="target names")
    missing = [str(name) for name in target_names if str(name) not in source]
    if missing:
        raise ValueError(f"missing target names in source: {missing}")
    return np.asarray([source[str(name)] for name in target_names], dtype=np.int64)


def align_named_axis(
    values: np.ndarray,
    source_names: Sequence[str],
    target_names: Sequence[str],
    *,
    axis: int,
) -> np.ndarray:
    array = np.asarray(values)
    normalized_axis = int(axis)
    if normalized_axis < 0:
        normalized_axis += array.ndim
    if normalized_axis < 0 or normalized_axis >= array.ndim:
        raise ValueError(f"axis {axis} is invalid for array with ndim={array.ndim}")
    if array.shape[normalized_axis] != len(source_names):
        raise ValueError(
            f"source-name count {len(source_names)} does not match axis size "
            f"{array.shape[normalized_axis]}"
        )
    permutation = build_name_permutation(source_names, target_names)
    return np.take(array, permutation, axis=normalized_axis)
