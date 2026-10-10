"""Shared, label-free outer folds covering train participants only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from trace_agent.data.split_table import normalize_eids, read_eid_table


def build_outer_folds(
    participant_ids: np.ndarray,
    train_rows: np.ndarray,
    *,
    seed: int = 7,
    path: str | Path | None = None,
    row_column: str = "row_index",
    eid_column: str = "eid",
    fold_column: str = "fold",
) -> np.ndarray:
    """Return fold IDs 1..5 in train-row order, never a new main split."""
    ids = normalize_eids(participant_ids)
    rows = np.asarray(train_rows, dtype=np.int64).reshape(-1)
    if rows.size < 5 or np.unique(rows).size != rows.size:
        raise ValueError("fivefold requires at least five unique train rows")
    if np.any(rows < 0) or np.any(rows >= ids.size):
        raise ValueError("train rows are outside participant axis")
    if path is None:
        assignment = np.zeros(rows.size, dtype=np.int64)
        for fold, (_, held) in enumerate(KFold(5, shuffle=True, random_state=seed).split(rows), 1):
            assignment[held] = fold
        return assignment
    frame = read_eid_table(path, eid_column=eid_column)
    required = {row_column, eid_column, fold_column}
    if len(required) != 3:
        raise ValueError("fold-table column names must be distinct")
    if not required.issubset(frame.columns) or frame[list(required)].isna().any(axis=None):
        raise ValueError("fold table requires non-missing row_index,eid,fold")
    numeric_rows = pd.to_numeric(frame[row_column], errors="raise").to_numpy(dtype=float)
    numeric_folds = pd.to_numeric(frame[fold_column], errors="raise").to_numpy(dtype=float)
    for values in (numeric_rows, numeric_folds):
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError("fold rows and fold IDs must be integers")
    source_rows = numeric_rows.astype(np.int64)
    source_ids = normalize_eids(frame[eid_column].to_numpy(), label="fold table")
    if np.unique(source_rows).size != source_rows.size or np.unique(source_ids).size != source_ids.size:
        raise ValueError("duplicate fold row or EID")
    if set(source_rows.tolist()) != set(rows.tolist()):
        raise ValueError("fold table must cover train rows only, exactly once")
    if not np.array_equal(source_ids, ids[source_rows]):
        raise ValueError("fold EID mismatch")
    folds = numeric_folds.astype(np.int64)
    if set(folds.tolist()) != {1, 2, 3, 4, 5}:
        raise ValueError("fold IDs must be 1..5 with all folds non-empty")
    lookup = dict(zip(source_rows.tolist(), folds.tolist()))
    return np.asarray([lookup[int(row)] for row in rows], dtype=np.int64)
